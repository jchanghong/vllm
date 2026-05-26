# 架构概览

本文档概述了 vLLM 的架构。

[TOC]

## 入口点

vLLM 提供了多种与系统交互的入口点。下图展示了它们之间的关系。

![入口点图](../assets/design/arch_overview/entrypoints.excalidraw.png)

### LLM 类

`LLM` 类提供了执行离线推理的主要 Python 接口，即在不使用独立模型推理服务器的情况下与模型交互。

以下是 `LLM` 类的使用示例：

??? code

    ```python
    from vllm import LLM, SamplingParams

    # 定义输入提示列表
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The largest ocean is",
    ]

    # 定义采样参数
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # 使用 OPT-125M 模型初始化 LLM 引擎
    llm = LLM(model="facebook/opt-125m")

    # 为输入提示生成输出
    outputs = llm.generate(prompts, sampling_params)

    # 打印生成的输出
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

更多 API 详情请参阅 API 文档中的[离线推理](../api/README.md#offline-inference)部分。

`LLM` 类的代码位于 [vllm/entrypoints/llm.py](../../vllm/entrypoints/llm.py)。

### 在线服务

vLLM 的第二个主要接口是通过其在线服务器提供的。可以使用 `vllm serve` 命令启动该服务器。

```bash
vllm serve <model>
```

`vllm` CLI 的代码位于 [vllm/entrypoints/cli/main.py](../../vllm/entrypoints/cli/main.py)。

有时您可能会看到直接使用 API 服务器入口点，而不是通过 `vllm` CLI 命令。例如：

```bash
python -m vllm.entrypoints.openai.api_server --model <model>
```

!!! warning

    `python -m vllm.entrypoints.openai.api_server` 已被弃用，
    可能在未来的版本中不再受支持。

该代码位于 [vllm/entrypoints/openai/api_server.py](../../vllm/entrypoints/openai/api_server.py)。

有关 API 服务器的更多详情，请参阅[在线服务](../serving/online_serving/README.md)文档。

## V1 进程架构

vLLM V1 采用多进程架构来分离关注点并最大化吞吐量。理解此架构对于正确调整部署中的 CPU 资源至关重要。关键进程包括：

### API 服务器进程

API 服务器进程处理 HTTP 请求（例如 OpenAI 兼容 API）、执行输入处理（分词、多模态数据加载）并将结果流式返回给客户端。它通过 ZMQ 套接字与引擎核心进程通信。

默认情况下有 **1 个 API 服务器进程**，但当使用数据并行时，API 服务器数量会自动扩展以匹配数据并行大小。也可以通过 `--api-server-count` 标志手动配置。每个 API 服务器通过 ZMQ 以多对多拓扑连接到**所有**引擎核心，使任何 API 服务器都可以将请求路由到任何引擎核心。每个 API 服务器进程使用多个 CPU 线程进行媒体加载（由 `VLLM_MEDIA_LOADING_THREAD_COUNT` 控制，默认为 8）。

代码位于 [vllm/entrypoints/openai/api_server.py](../../vllm/entrypoints/openai/api_server.py) 和 [vllm/v1/utils.py](../../vllm/v1/utils.py)。

### 引擎核心进程

引擎核心进程运行调度器、管理 KV 缓存并协调跨 GPU 工作器的模型执行。它运行一个繁忙循环，持续调度请求并将工作分派给 GPU 工作器。

每个数据并行等级有 **1 个引擎核心进程**。例如，使用 `--data-parallel-size 4` 时，有 4 个引擎核心进程。

代码位于 [vllm/v1/engine/core.py](../../vllm/v1/engine/core.py) 和 [vllm/v1/engine/utils.py](../../vllm/v1/engine/utils.py)。

### GPU 工作器进程

每个 GPU 由一个专用工作器进程管理。工作器进程加载模型权重、执行前向传播并管理 GPU 内存。工作器与拥有它们的引擎核心进程通信。

每个 GPU 有 **1 个工作器进程**。每个引擎核心的 GPU 工作器进程总数等于 `tensor_parallel_size x pipeline_parallel_size`。

代码位于 [vllm/v1/executor/multiproc_executor.py](../../vllm/v1/executor/multiproc_executor.py) 和 [vllm/v1/worker/gpu_worker.py](../../vllm/v1/worker/gpu_worker.py)。

### DP 协调器进程（条件性）

当使用数据并行（`--data-parallel-size > 1`）时，一个额外的协调器进程负责管理跨 DP 等级的负载均衡，并协调 MoE 模型的同步前向传播。

有 **1 个 DP 协调器进程**（仅在启用数据并行时存在）。

代码位于 [vllm/v1/engine/coordinator.py](../../vllm/v1/engine/coordinator.py)。

### 进程数量汇总

对于一个具有 `N` 个 GPU、`TP` 张量并行大小、`DP` 数据并行大小和 `A` 个 API 服务器数量的部署：

| 进程类型 | 数量 | 说明 |
| - | - | - |
| API 服务器 | `A`（默认为 `DP`） | 处理 HTTP 请求和输入处理 |
| 引擎核心 | `DP`（默认为 1） | 调度器和 KV 缓存管理 |
| GPU 工作器 | `N`（= `DP x PP x TP`） | 每个 GPU 一个，执行模型前向传播 |
| DP 协调器 | 如果 `DP > 1` 则为 1，否则为 0 | 跨 DP 等级的负载均衡 |
| **总计** | **`A + DP + N`（如果 DP > 1 则 +1）** | |

例如，一个典型的单节点部署，4 个 GPU（`vllm serve -tp=4`）包含：

- 1 个 API 服务器 + 1 个引擎核心 + 4 个 GPU 工作器 = **6 个进程**

<figure markdown="1">
![V1 进程架构 - TP=4](../assets/design/arch_overview/v1_process_architecture_tp4.png)
</figure>

一个数据并行部署，8 个 GPU（`vllm serve -tp=2 -dp=4`）包含：

- 4 个 API 服务器 + 4 个引擎核心 + 8 个 GPU 工作器 + 1 个 DP 协调器 = **17 个进程**

<figure markdown="1">
![V1 进程架构 - TP=2, DP=4](../assets/design/arch_overview/v1_process_architecture_tp2_dp4.png)
</figure>

有关 CPU 资源大小调整建议，请参阅
[GPU 部署的 CPU 资源](../configuration/optimization.md#cpu-resources-for-gpu-deployments)。

## LLM 引擎

`LLMEngine` 和 `AsyncLLMEngine` 类是 vLLM 系统运行的核心，
负责模型推理和异步请求处理。

![LLMEngine 图](../assets/design/arch_overview/llm_engine.excalidraw.png)

### LLMEngine

`LLMEngine` 类是 vLLM 引擎的核心组件。它负责接收客户端请求并从模型生成输出。`LLMEngine` 包括输入处理、模型执行（可能分布在多个主机和/或 GPU 上）、调度和输出处理。

- **输入处理**：使用指定的分词器处理输入文本的分词。
- **调度**：选择每个步骤处理哪些请求。
- **模型执行**：管理语言模型的执行，包括跨多个 GPU 的分布式执行。
- **输出处理**：处理模型生成的输出，将语言模型的 token ID 解码为人类可读的文本。

`LLMEngine` 的代码位于 [vllm/engine/llm_engine.py](../../vllm/engine/llm_engine.py)。

### AsyncLLMEngine

`AsyncLLMEngine` 类是 `LLMEngine` 类的异步包装器。它使用 `asyncio` 创建一个后台循环，持续处理传入的请求。`AsyncLLMEngine` 专为在线服务设计，可以处理多个并发请求并将输出流式返回给客户端。

OpenAI 兼容的 API 服务器使用 `AsyncLLMEngine`。还有一个演示 API 服务器，作为更简单的示例位于 [vllm/entrypoints/api_server.py](../../vllm/entrypoints/api_server.py)。

`AsyncLLMEngine` 的代码位于 [vllm/engine/async_llm_engine.py](../../vllm/engine/async_llm_engine.py)。

## 工作器

工作器是运行模型推理的进程。vLLM 遵循常见的做法，即使用一个进程控制一个加速器设备，例如 GPU。例如，如果我们使用大小为 2 的张量并行和大小为 2 的流水线并行，我们将有 4 个工作器。工作器通过其 `rank` 和 `local_rank` 进行标识。`rank` 用于全局协调，而 `local_rank` 主要用于分配加速器设备和访问本地资源，如文件系统和共享内存。

## 模型运行器

每个工作器有一个模型运行器对象，负责加载和运行模型。模型执行逻辑的大部分位于此处，例如准备输入张量和捕获 CUDA 图。

## 模型

每个模型运行器对象有一个模型对象，即实际的 `torch.nn.Module` 实例。有关各种配置如何影响最终获得的类，请参见 [huggingface_integration](huggingface_integration.md)。

## 类层次结构

下图显示了 vLLM 的类层次结构：

![类层次结构](../assets/design/hierarchy.png)

此类层次结构背后有几个重要的设计选择：

1. **可扩展性**：层次结构中的所有类都接受一个包含所有必要信息的配置对象。[VllmConfig](https://github.com/vllm-project/vllm/blob/d1c6799b8870e513bf4f2305cbf6cda9fc3d773b/vllm/config.py#L2036) 类是传递的主要配置对象。类层次结构相当深，每个类都需要读取其感兴趣的配置。通过将所有配置封装在一个对象中，我们可以轻松地传递配置对象并访问所需的配置。假设我们想添加一个新功能（鉴于 LLM 推理领域发展迅速，这很常见），该功能仅涉及模型运行器。我们将不得不在 `VllmConfig` 类中添加一个新的配置选项。由于我们传递整个配置对象，我们只需将配置选项添加到 `VllmConfig` 类中，模型运行器就可以直接访问它。我们不需要更改引擎、工作器或模型类的构造函数来传递新的配置选项。

2. **统一性**：模型运行器需要一个统一的接口来创建和初始化模型。vLLM 支持 50 多种流行的开源模型。每个模型都有自己的初始化逻辑。如果构造函数签名因模型而异，模型运行器将不知道如何相应地调用构造函数，除非使用复杂且容易出错的检查逻辑。通过使模型类的构造函数统一，模型运行器可以轻松创建和初始化模型，而无需知道具体的模型类型。这对于组合模型也很有用。视觉语言模型通常由视觉模型和语言模型组成。通过使构造函数统一，我们可以轻松创建视觉模型和语言模型，并将它们组合成视觉语言模型。

!!! note
    为支持此更改，所有 vLLM 模型的签名已更新为：

    ```python
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
    ```

    为避免意外传递不正确的参数，构造函数现在是仅关键字参数。这确保如果传递了旧配置，构造函数将引发错误。vLLM 开发者已经为 vLLM 中的所有模型进行了此更改。对于树外注册的模型，开发者需要更新其模型，例如添加适配代码以将旧构造函数签名适配到新签名：

    ??? code

        ```python
        class MyOldModel(nn.Module):
            def __init__(
                self,
                config,
                cache_config: Optional[CacheConfig] = None,
                quant_config: Optional[QuantizationConfig] = None,
                lora_config: Optional[LoRAConfig] = None,
                prefix: str = "",
            ) -> None:
                ...

        from vllm.config import VllmConfig
        class MyNewModel(MyOldModel):
            def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
                config = vllm_config.model_config.hf_config
                cache_config = vllm_config.cache_config
                quant_config = vllm_config.quant_config
                lora_config = vllm_config.lora_config
                super().__init__(config, cache_config, quant_config, lora_config, prefix)

        from packaging import version
        if version.parse(__version__) >= version.parse("0.6.4"):
            MyModel = MyNewModel
        else:
            MyModel = MyOldModel
        ```

    这样，模型可以同时兼容旧版和新版 vLLM。

3. **初始化时进行分片和量化**：某些功能需要更改模型权重。例如，张量并行需要对模型权重进行分片，量化需要对模型权重进行量化。实现此功能有两种可能的方式。一种方式是在模型初始化后更改模型权重。另一种方式是在模型初始化期间更改模型权重。vLLM 选择后者。第一种方式对于大型模型不具备可扩展性。假设我们想用 16 个 H100 80GB GPU 运行一个 405B 模型（大约 810GB 权重）。理想情况下，每个 GPU 只应加载 50GB 权重。如果在模型初始化后更改模型权重，我们需要将完整的 810GB 权重加载到每个 GPU 上，然后进行分片，导致巨大的内存开销。相反，如果在模型初始化期间分片权重，每一层只会创建它需要的权重分片，导致更小的内存开销。同样的思路也适用于量化。注意，我们在模型的构造函数中添加了一个额外的参数 `prefix`，以便模型可以根据前缀以不同方式初始化自身。这对于非均匀量化很有用，其中模型的不同部分以不同方式量化。`prefix` 通常是顶级模型的空字符串，对于子模型则是像 `"vision"` 或 `"language"` 这样的字符串。通常，它与检查点文件中模块的 state dict 的名称匹配。

这种设计的一个缺点是，编写 vLLM 中单个组件的单元测试比较困难，因为每个组件都需要通过一个完整的配置对象来初始化。我们通过提供一个默认初始化函数来解决这个问题，该函数创建一个所有字段设置为 `None` 的默认配置对象。如果我们想要测试的组件只关心配置对象中的几个字段，我们可以创建一个默认配置对象并设置我们关心的字段。这样，我们就可以隔离测试该组件。请注意，vLLM 中的许多测试是测试整个系统的端到端测试，所以这不算大问题。

总之，完整的配置对象 `VllmConfig` 可以被视为一个引擎级别的全局状态，在所有 vLLM 类之间共享。
