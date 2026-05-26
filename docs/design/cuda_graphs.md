# CUDA 图

本文介绍了 vLLM v1 中超越先前 [torch.compile 集成](torch_compile.md) 的新 CUDA 图模式。总结来说，我们：

1. 添加了灵活的 `cudagraph_mode` 配置
2. 使完整的 CUDA 图支持与编译正交
3. 引入了 CUDA 图调度器作为中央控制器，自动选择所需的运行时模式和每批次的 CUDA 图

在本文档中，我们将讨论：

* [动机](#motivation)
* [CUDA 图模式](#cudagraphmodes)
* [详细设计](#detailed-design)
* [不同 CUDA 图模式的示例用法](#usage-guide)
* [视觉编码器（ViT）CUDA 图](cuda_graphs_multimodal.md)

!!! note
    在本文档中，我们将纯解码（`max_query_len=1`）或推测解码（`max_query_len =1+num_spec_tokens`）批次称为**均匀解码**批次，相反的是**非均匀**批次（即预填充或混合预填充-解码批次）。

!!! note
    以下内容主要基于 <https://github.com/vllm-project/vllm/pull/20059> 的最新提交。

## 动机

最初的逐段编译被构建为允许逐段 CUDA 图捕获，排除不支持 CUDA 图的操作（主要是注意力）。这允许从 CUDA 图中获得一些加速，同时保持与所有注意力后端的兼容性。我们后来添加了对"完整 CUDA 图"的支持（通过不逐段编译），以便在注意力支持 CUDA 图的情况下进一步减少延迟。然而，编译和 CUDA 图捕获之间的这种紧密耦合导致了全有或全无的体验，缺乏灵活性。许多注意力后端也没有准备好统一的"完整"CUDA 图捕获（例如，目前只有 FlashAttention 3 支持）或者只支持纯解码批次的 CUDA 图（例如 Flashinfer、FlashMLA 和 Mamba 等）。这导致了令人困惑的性能/兼容性权衡、不一致的 CUDA 图支持以及日益复杂的代码结构。

这促使我们寻求更细粒度的 CUDA 图解决方案，具有以下特性：

* 明确感知预填充/混合或（均匀）解码批次的 CUDA 图，并分别捕获它们。
* 将 CUDA 图捕获逻辑与编译分离（尽可能）以实现功能正交性，这意味着：
    * 使用相同的编译图捕获逐段和完整 CUDA 图，以及
    * 无编译的完整 CUDA 图捕获。
* 在完全和逐段 CUDA 图之间根据批次组成在运行时进行分派。
* 集中控制 CUDA 图行为，降低代码复杂度并提高可扩展性。

这些特性为各种启动/性能权衡和功能支持的 CUDA 图捕获和编译提供了最大的灵活性。

## `CudagraphModes`

[CUDAGraphMode][vllm.config.compilation.CUDAGraphMode] 是您在 `CompilationConfig.cudagraph_mode` 中调节的单一旋钮：

* `NONE` — 关闭 CUDA 图。适合调试。
* `PIECEWISE` — 单一模式策略（也是过去的默认值）。它是最灵活的：注意力或其他不兼容 CUDA 图的操作保持即时模式，其他所有内容进入 CUDA 图。需要逐段编译。
* `FULL` — 单一模式策略，仅为非均匀批次捕获完整 CUDA 图，然后均匀解码批次重用相同 batch_size 的非均匀批次的 CUDA 图，因为它们是兼容的；适用于小模型或 prompt 较小的工作负载。
* `FULL_DECODE_ONLY` — 为均匀解码使用完整 CUDA 图，不为预填充/混合等使用 CUDA 图；适用于 P/D 设置中的解码实例，其中预填充不那么重要，这样可以节省 `PIECEWISE` CUDA 图所需的内存。
* `FULL_AND_PIECEWISE` — （默认模式）为均匀解码使用完整 CUDA 图，为其他情况使用逐段 CUDA 图；通常是最佳性能设置，特别适合低延迟的小模型或 MoE，但也需要最多内存和最长捕获时间。

默认值：如果您在 v1 上使用逐段编译，我们默认使用 `FULL_AND_PIECEWISE` 以获得更好的性能（对于池化模型，仍然是 `PIECEWISE`）。否则，例如如果逐段编译不可用，我们默认使用 `NONE`。

虽然 `NONE`、`PIECEWISE` 和 `FULL` 是单一模式配置，分别等同于即时执行、逐段 CUDA 图和完整 CUDA 图的过去实现，但 `FULL_DECODE_ONLY` 和 `FULL_AND_PIECEWISE` 是新添加的双模式配置，需要调度以根据运行时批次在具体运行时模式之间动态切换。

!!! note
    这里，单一模式 `NONE`、`PIECEWISE` 和 `FULL` 被视为 CUDA 图调度的运行时模式。如果使用双模式，调度器将根据批次组成始终分派到其成员模式之一（加上潜在的 `NONE`，如果没有合适的 CUDA 图可用）。

虽然级联注意力与 CUDA 图不兼容，但它现在与所有可能的 CUDA 图模式配置兼容。如果批次使用级联注意力，它总是被分派到 `PIECEWISE` 模式（如果可用，否则为 `NONE`）。

!!! note
    并非所有 CUDA 图模式都与每个注意力后端兼容。我们会自动"降级"模式到最接近的支持模式。例如，如果后端仅支持纯解码/均匀批次的 CUDA 图，我们在启用逐段编译时将 `FULL` 转换为 `FULL_AND_PIECEWISE`，否则转换为 `FULL_DECODE_ONLY`。

## 详细设计

### 概览

新的 CUDA 图逻辑建立在逐段编译之上，并支持双 CUDA 图运行时模式切换。该系统包含以下核心组件：

* [CUDAGraphWrapper][vllm.compilation.cuda_graph.CUDAGraphWrapper]：处理被包装可调用对象的 CUDA 图捕获和重放的包装器。
* [CudagraphDispatcher][vllm.v1.cudagraph_dispatcher.CudagraphDispatcher]：中央控制器，包含 CUDA 图的唯一真实来源，并处理它们之间的分派。
* [CUDAGraphMode][vllm.config.compilation.CUDAGraphMode]：描述支持的运行时模式的枚举（上面已介绍）。
* [BatchDescriptor][vllm.forward_context.BatchDescriptor]，作为用于分派的运行时批次的唯一表示。

请参阅下图，快速比较之前和当前的 CUDA 图与 Inductor 编译的设计模式。我们可以看到，以前 CUDA 图逻辑和编译逻辑紧密耦合在 vllm `PiecewiseBackend` 中，CUDA 图由 `batch_size` 隐式分派。现在，CUDA 图逻辑被分离到 `CUDAGraphWrapper` 类中，负责完整和逐段 CUDA 图能力，并且分派通过**运行时模式**加上 `BatchDescriptor` 作为**分派键**，通过 `CudagraphDispatcher` **显式**完成。

**以前：**

![以前的设计](../assets/design/cuda_graphs/previous_design.png)

**现在：**

![现在的设计](../assets/design/cuda_graphs/current_design.png)

### `BatchDescriptor`

[BatchDescriptor][vllm.forward_context.BatchDescriptor] 是 `ForwardContext` 中的一个组件，与 CUDA 图运行时模式一起，作为运行时分配键的核心结构。原型如下：

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int
    num_reqs: int
    uniform: bool = False
    has_lora: bool = False
```

其中 `num_tokens` 可以是填充后的 token 长度，`uniform` 表示所有请求是否具有相同的查询长度。许多注意力后端仅在批次均匀时支持完整 CUDA 图；纯解码批次是均匀的，但查询长度可能不是 1（即 `num_tokens == num_reqs`），这种情况出现在推测解码的验证传递中，其中"解码"批次的查询长度为 `1+num_spec_tokens`。

此结构的目标是用尽可能少的项唯一标识一个（填充后的）批次，对应于一个 CUDA 图条目。

!!! note
    `BatchDescriptor` 的原型将来可能会扩展以适应更一般的情况，例如，包含更多项，如 `uniform_query_len` 以支持多个不同的均匀解码长度设置（<https://github.com/vllm-project/vllm/pull/23679>），或支持输入不一定感知 token 长度的模型的 CUDA 图所需的其他修改（例如，一些多模态输入）。

### `CudagraphDispatcher`

[CudagraphDispatcher][vllm.v1.cudagraph_dispatcher.CudagraphDispatcher] 负责维护两组有效的分派键，一组用于 `FULL` 运行时模式，一组用于 `PIECEWISE` 运行时模式，并在执行模型前向传播之前分派正确的运行时模式和分派键。它将接收初始键（填充输入的粗略 batch_descriptor）并返回选定的运行时模式和最终的 batch_descriptor，然后通过前向上下文告诉 `CUDAGraphWrapper` 实例该决定。注意，`CudagraphDispatcher` 是可用 CUDA 图键的唯一真实来源，`CUDAGraphWrapper` 实例可以盲目信任前向上下文关于要分派到哪个 CUDA 图的信息。这让我们简化包装器代码并将逻辑集中在调度器中。

分派键通过调度器的 `initialize_cudagraph_keys` 方法初始化，该方法由 gpu_model_runner 在所有可能的注意力后端初始化后调用。这是未来我们可以更花哨并"准备"各种 CUDA 图组合的地方。目前，我们仅根据 `cudagraph_mode` 的 `decode_mode`/`mixed_mode` 的有效组合和编译配置中的 `cudagraph_capture_sizes` 附加可用键。

分派代码如下所示：

```python
batch_descriptor=BatchDescriptor(num_tokens=num_input_tokens, uniform_decode=...)
runtime_mode, batch_descriptor = cudagraphdispatcher.dispatch(batch_descriptor)
# 执行
with set_forward_context(
    ..., 
    cudagraph_runtime_mode=runtime_mode, 
    batch_descriptor=batch_descriptor,
):
     output = self.model(...)
```

在 `dispatch()` 方法内部，调度器将搜索合适的 CUDA 图运行时模式和现有的分派键以返回。我们基本上按照以下优先级搜索现有键：`FULL`>`PIECEWISE`>`None`。如果分派键不存在，则默认返回 `NONE` 模式以进行即时执行。实现可以在此[处](https://github.com/vllm-project/vllm/blob/main/vllm/v1/cudagraph_dispatcher.py#L91)找到。

以下是在模型执行器中运行时工作的简化说明：
![执行器运行时](../assets/design/cuda_graphs/executor_runtime.png)

### `CUDAGraphWrapper`

一个 [CUDAGraphWrapper][vllm.compilation.cuda_graph.CUDAGraphWrapper] 实例包装一个可运行对象，并简单地模拟具有附加 CUDA 图能力的可运行对象。每个包装器实例绑定到一个特定的 `runtime_mode`，仅限于 `PIECEWISE` 和 `FULL` 模式，并负责捕获/重放和传递（直接调用）可运行对象。在运行时，每个包装器将：

1. 从全局前向上下文中检查 runtime_mode 和 batch_descriptor（分派键）。
2. 如果 runtime_mode 是 `NONE` 或 runtime_mode 与包装器的模式不匹配，则直接调用可运行对象。
3. 否则，即 runtime_mode 与包装器的模式匹配，包装器将执行 CUDA 图捕获（如果键不存在，则创建新条目并缓存）或重放（如果键存在于缓存中）。

以上步骤基于 CUDA 图包装器将直接信任前向上下文中的内容（由调度器控制）的假设。这让我们简化并集中逻辑，降低复杂性以及包装器和调度器之间状态不匹配的风险。它还允许为 `FULL` 和 `PIECEWISE` 运行时模式重用包装器类。请参见此[处](https://github.com/vllm-project/vllm/blob/f751e50b7a2aae3110d83ed0d88202fc91b3e78a/vllm/compilation/cuda_graph.py#L106)的实现。

#### 嵌套包装器设计

使完整 CUDA 图和逐段 CUDA 图共存且兼容的核心机制是嵌套 CUDA 图包装器设计，建立在仅有单个逐段 FX 图的逐段编译之上。我们在整个模型外部包装一个 `FULL` 模式包装器，用于完整 CUDA 图功能；同时，每个逐段后端在编译内部通过 `PIECEWISE` 模式包装器包装。

下面的流程图应清楚地描述其工作原理。
![包装器流程](../assets/design/cuda_graphs/wrapper_flow.png)

因此，对于 `FULL` 运行时模式，由于逐段包装器未激活，捕获/重放完整 CUDA 图是安全的。对于 `PIECEWISE` 模式，情况类似，因为 `FULL` 模式包装器和 `PIECEWISE` 模式包装器之间没有冲突。对于 `NONE` 运行时模式，`FULL` 和 `PIECEWISE` 包装器都不会激活，因此我们简单地回退到即时执行。

### 完整 CUDA 图捕获和预热

CUDA 图捕获发生在运行器首次使用非 `NONE` 运行时模式调用模型前向传播时（使用 `_dummy_run`）。对于完整 CUDA 图捕获，我们通过适当设置注意力元数据来显式捕获不同情况（即预填充/混合批次或均匀解码批次），以确保底层注意力后端启动期望的内核例程。为了区分预填充/混合批次或均匀解码批次，最重要的属性是 attn_metadata 中的 `max_query_len`（对大多数注意力后端都是如此）。对于均匀解码，我们将其设置为期望的 `uniform_query_len`，否则对于非均匀解码批次，我们将其设置为 `num_tokens`。

CUDA 图包装器不再管理预热逻辑。预热过程现在由 GPU 模型运行器直接控制，其中分配 `NONE` 运行时模式以进行即时执行预热。在为完整 CUDA 图预热时，在预热 `dummy_run` 调用期间显式运行注意力也很重要。

## 注意力后端的 CUDA 图兼容性

为了指示注意力后端对 CUDA 图的兼容性，我们引入了一个新的枚举类型 [AttentionCGSupport][vllm.v1.attention.backend.AttentionCGSupport]，这是一个跟踪注意力后端支持 CUDA 图能力的枚举类型。该值按能力顺序排序，即 `ALWAYS` > `UNIFORM_BATCH` > `UNIFORM_SINGLE_TOKEN_DECODE` > `NEVER`。

```python
class AttentionCGSupport(enum.Enum):
    """注意力后端 CUDA 图支持的常量
    这里我们不考虑级联注意力，因为目前
    它从不支持 CUDA 图。"""

    ALWAYS = 3
    """始终支持 CUDA 图；支持混合预填充-解码"""
    UNIFORM_BATCH = 2
    """支持仅包含相同查询长度的批次的 CUDA 图，
    这可用于推测解码
        即"解码"是 1 + num_speculative_tokens"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """支持仅包含 query_len==1 解码的批次的 CUDA 图"""
    NEVER = 0
    """不支持 CUDA 图"""
```

假设我们有混合注意力后端（例如在 mamba mixer 模型中）。在这种情况下，我们寻求所有后端的最小能力来确定模型的最终能力，并且我们可能通过将模式降级到最合适的一个来解决不兼容的 CUDA 图模式。例如，如果最小能力是 `UNIFORM_BATCH`，将 `FULL` 模式降级为 `FULL_AND_PIECEWISE` 模式；如果最小能力是 `NEVER`（对于 -O3 编译模式），则降级为 `PIECEWISE` 模式。有关完整的回退策略，请参见代码[此处][vllm.v1.worker.gpu_model_runner.GPUModelRunner._check_and_update_cudagraph_mode]。

下表列出了在撰写本文时支持完整 CUDA 图的后端。

| 注意力后端 | cudagraph_support | 注释 |
| :---------------- | :---------------- | :------- |
| FlashAttention v2 | `UNIFORM_BATCH` | 实际上是 `ALWAYS` 但由于性能原因回退到 `FULL_AND_PIECEWISE` |
| FlashAttention v3 | `ALWAYS` | 对两种批次都有统一例程，因此 `FULL` 模式很好 |
| Triton Attention | `ALWAYS` | 首选 `FULL_AND_PIECEWISE`，因为它对预填充/混合和纯解码批次有不同的内核 |
| AITER FlashAttention | `UNIFORM_BATCH` | |
| FlashInfer | `UNIFORM_SINGLE_TOKEN_DECODE` | 在 Blackwell 上使用 TRTLLM 注意力时将设置为 `UNIFORM_BATCH` |
| FlashMLA | `UNIFORM_BATCH` | |
| FlashInferMLA | `UNIFORM_BATCH` | |
| FlashInferMLASparse | `UNIFORM_BATCH` | |
| AITER MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| CUTLASS MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| Mamba attention | `UNIFORM_SINGLE_TOKEN_DECODE` | |

未列出的后端都被声明为 `NEVER`。

## 使用指南

现在 CLI 直接使用 cudagraph_mode 的大写字符串作为 compilation_config：`--compilation-config '{"cudagraph_mode": "..."}'`，其中 `...` 应为 `NONE`、`PIECEWISE`、`FULL`、`FULL_DECODE_ONLY` 和 `FULL_AND_PIECEWISE` 之一。注意，所有 `PIECEWISE` 相关模式需要逐段编译，所有 `FULL` 相关模式需要注意力后端的 CUDA 图支持。例如：

```bash
vllm serve --model meta-llama/Llama-3.1-8B-Instruct --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

### Python 示例

```python
import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

import vllm
from vllm.config import CUDAGraphMode

compilation_config = {"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}
model = vllm.LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
    compilation_config=compilation_config,
)
sampling_params = vllm.SamplingParams(
    temperature=0,  # 贪心解码
    max_tokens=1024,
)
outputs = model.generate(
    ["My name is John and"],
    sampling_params=sampling_params,
)
```

### 逐段编译和完整图自定义传递（注意力融合、序列并行）

不幸的是，一些自定义编译传递必须看到整个图才能有效，因此与逐段编译不兼容。这包括 `AttnQuantFusionPass` 和 `SequenceParallelismPass`。作为短期解决方案，我们在启用注意力融合时自动禁用逐段编译（通过设置 `splitting_ops=[]`）。我们使用 CUDA 图模式 `FULL` 或 `FULL_DECODE_ONLY`（取决于后端支持）。然而，这导致了另一个优化不兼容性和令人困惑的性能权衡。

长期来看，我们增加了在 Inductor 中（而不是在 Dynamo 之后立即）分区图的能力。可以通过 `CompilationConfig.use_inductor_graph_partition=True` 启用，但目前是实验性的，仅适用于 `torch>=2.9`。这也增加了编译时间，因为它需要编译整个图，并且不能重用逐段编译产物。一旦 vLLM 支持 2.9，我们计划使这成为默认方法，因为它也将加速逐段 CUDA 图捕获。

## 关于性能

请参见以下链接获取示例：

* [20059#issuecomment-3160858458](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3160858458)
* [20059#issuecomment-3188735226](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3188735226)
* [20059#issuecomment-3219888738](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3219888738)
