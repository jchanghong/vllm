# 如何调试 vLLM-torch.compile 集成

TL;DR：

- 使用 tlparse 获取 torch.compile 日志。将这些日志包含在错误报告和/或支持请求中。
- vLLM-torch.compile 集成由多个组件组成。vLLM 提供了关闭每个组件的标志：

| 在线标志 | 离线标志 | 结果 |
| - | - | - |
| --enforce-eager | enforce_eager=True | 关闭 torch.compile 和 CUDA 图 |
| -cc.mode=0 | compilation_config=CompilationConfig(mode=CompilationMode.NONE) | 仅关闭 torch.compile |
| -cc.mode=1 | compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE) | 关闭 vLLM-compile 对 torch.compile 的修改 |
| -cc.cudagraph_mode=NONE | compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE) | 仅关闭 CUDA 图 |
| -cc.backend=eager | compilation_config=CompilationConfig(backend='eager') | 关闭 TorchInductor |
| -cc.ir_enable_torch_wrap=False | compilation_config=CompilationConfig(ir_enable_torch_wrap=False) | 关闭 vLLM IR 包装 |

## vLLM-torch.compile 概览

为了提升性能，vLLM 利用 torch.compile 和 CUDA 图来加速。torch.compile 为 PyTorch 代码生成优化内核，而 CUDA 图消除了开销。最值得注意的是，vLLM-compile 不是 torch.compile，它是使用 PyTorch 编译内部 API 构建的自定义编译器。

![vLLM-compile 图](../assets/design/debug_vllm_compile/design_diagram.png)

- 给定一个模型，我们通过 TorchDynamo 进行完整的图捕获，该图在批量大小（token 数）上是动态的。
- 然后 vLLM 可选地分割和/或特化此图，然后使用 TorchInductor 将每个图编译成编译产物。此步骤可能使用 vLLM 自定义 Inductor 传递进一步优化图。这包括 vLLM IR 降级以消除分派开销。
- 编译产物保存到 vLLM 的编译缓存中，以便将来加载。
- vLLM 应用 CUDA 图以减少 CPU 开销。

在这四个步骤中都可能出现问题。当出现问题时，请尝试隔离出问题的子系统——这将允许您关闭最少的功能以保持可靠性目标，同时最小化对性能的影响，并且在您提交错误报告时也有助于我们（vLLM）。

有关设计的更多详情，请参见以下资源：

- [vLLM-torch.compile 介绍博客文章](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [vLLM-torch.compile 集成设计](./torch_compile.md)
- [vLLM IR 设计](./vllm_ir.md)
- [vLLM 办公时间 #26](https://www.youtube.com/live/xLyxc7hxCJc?si=Xulo9pe53C6ywf0V&t=561)
- [PyTorch 大会 2025 演讲](https://youtu.be/1wV1ESbGrVQ?si=s1GqymUfwiwOrDTg&t=725)

## 使用 tlparse

使用 [tlparse](https://github.com/meta-pytorch/tlparse) 查看 torch.compile 日志。这些日志显示编译过程的所有阶段，包括 torch.compile 生成的融合内核。

安装 tlparse：

```sh
pip install tlparse
```

要启用 torch.compile 日志，可以设置环境变量 `TORCH_TRACE=<dir>`。在追踪期间，该目录中会为每个 rank 创建一个文件，每个文件包含编译过程中的产物。如果可以，我们建议将这些日志文件与错误报告一起发送——它们非常有帮助。

用法（离线推理）

```sh
TORCH_TRACE=~/trace_dir python my_script.py
tlparse ~/trace_dir/<rank_0_log_file>
```

用法（服务）

```sh
TORCH_TRACE=~/trace_dir vllm serve
# 退出服务器（ctrl-c）
tlparse ~/trace_dir/<rank_0_log_file>
```

给定其中一个日志文件，`tlparse` 命令会输出一些 HTML 文件（可能输出到例如 `./tl_out/index.html`）。打开它查看日志。它看起来像下面这样：

![tlparse 示例](../assets/design/debug_vllm_compile/tlparse_inductor.png)

## 关闭 vLLM-torch.compile 集成

传递 `--enforce-eager` 以关闭 vLLM-torch.compile 集成并完全在即时模式下运行。这包括关闭 CUDA 图。

```sh
# 在线
vllm serve --enforce-eager
```

```py
# 离线
LLM(model, enforce_eager=True)
```

仅关闭 torch.compile，在编译配置中传递 `mode = NONE`（`-cc` 是 `--compilation_config` 的简写）：

```sh
# 在线
vllm serve -cc.mode=0
```

```py
# 离线
from vllm.config.compilation import CompilationConfig, CompilationMode
LLM(model, compilation_config=CompilationConfig(mode=CompilationMode.NONE))
```

仅关闭 CUDA 图，传递 `cudagraph_mode = NONE`：

```sh
# 在线
vllm serve -cc.cudagraph_mode=NONE
```

```py
# 离线
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
LLM(model, compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE))
```

vLLM IR 大量使用编译管道，包括函数化、自定义融合和降级。要关闭此功能并捕获 vLLM IR 的即时模式分派行为，请使用 `ir_enable_torch_wrap=False` 运行。IR torch wrap 仅在使用 `mode=VLLM_COMPILE` 和 `backend="inductor"`（默认）时默认启用。

```sh
# 在线
vllm serve -cc.ir_enable_torch_wrap=False
```

```py
# 离线
from vllm.config.compilation import CompilationConfig
LLM(model, compilation_config=CompilationConfig(ir_enable_torch_wrap=False))
```

## 调试 TorchDynamo

vLLM 要求模型代码可通过 TorchDynamo（torch.compile 的前端）捕获为完整图。TorchDynamo 不支持所有 Python 特性。如果遇到不支持的特性（这有时被称为图断裂），它将在完整图模式下报错。

如果您遇到图断裂，请[向 pytorch/pytorch 提交问题](https://github.com/pytorch/pytorch)，以便 PyTorch 开发者可以优先处理。然后，请尽力重写代码以避免图断裂。更多信息，请参见此 [Dynamo 指南](https://docs.pytorch.org/docs/stable/compile/programming_model.dynamo_core_concepts.html)。

## 调试动态形状完整图捕获

vLLM 要求模型的前向传播可以被捕获为一个在批量大小（即 token 数）上是动态的完整图。它（默认情况下）将这一个图编译成一个产物，并对所有批量大小使用此产物。

如果您的代码无法使用动态形状捕获，您可能会遇到静默不正确、显式错误或 CUDA 非法内存访问。例如，以下内容无法捕获为单个图：

```py
if data.size[0] % 128 == 0:
    foo(...)
else:
    bar(...)
```

这个问题很容易诊断。使用 tlparse 并点击 `compilation_metrics`：它将告诉您批量大小上的符号约束。如果存在任何限制批量大小的约束，那么我们就遇到了问题。

![不良 tlparse 示例](../assets/design/debug_vllm_compile/dynamic_shapes.png)

为避免此问题，请执行以下任一操作：

1. 避免对 token 数量进行分支
2. 将分支逻辑包装到自定义算子中。TorchDynamo 不会追踪自定义算子。

## 调试约束违反和动态形状守卫问题

动态形状守卫是 Dynamo 守卫的一个特定类别。它们是 `torch.compile` 附加到动态维度（例如 `seq_len`）的约束，以确保编译后的产物保持有效。这些守卫通常出现在框架代码、自定义传递或用户代码根据动态形状值进行分支时。

**示例：**

```python
if x > 10:
    # 路径 A
else:
    # 路径 B
```

这创建了一个守卫 `x > 10` 或 `x <= 10`，取决于追踪了哪个路径。

**vLLM 的假设：**
vLLM 假设 torch.compile 添加的所有守卫都是安全丢弃的，并且不会将编译后的图约束到特定的输入形状。当此假设被违反时，可能会导致用户需要调试的问题。表明此假设被违反的一些副作用是运行时错误或 `ConstraintViolationErrors`。

如果动态形状被约束为单个值，将抛出 `ConstraintViolationErrors`。如果您遇到约束违反错误或怀疑动态形状守卫被错误添加，您可以使用更严格的动态形状模式来帮助调试问题：

```sh
# 在线 - 使用 unbacked 模式
vllm serve meta-llama/Llama-3.2-1B -cc.dynamic_shapes_config.type=unbacked

# 在线 - 使用 backed_size_oblivious 模式
vllm serve meta-llama/Llama-3.2-1B -cc.dynamic_shapes_config.type=backed_size_oblivious
```

```py
# 离线 - 使用 unbacked 模式
from vllm.config.compilation import CompilationConfig, DynamicShapesConfig, DynamicShapesType
LLM(model, compilation_config=CompilationConfig(
    dynamic_shapes_config=DynamicShapesConfig(type=DynamicShapesType.UNBACKED)
))

# 离线 - 使用 backed_size_oblivious 模式
from vllm.config.compilation import CompilationConfig, DynamicShapesConfig, DynamicShapesType
LLM(model, compilation_config=CompilationConfig(
    dynamic_shapes_config=DynamicShapesConfig(type=DynamicShapesType.BACKED_SIZE_OBLIVIOUS)
))
```

这些模式更严格，减少或消除了动态形状守卫的需要，这有助于隔离问题：

- `unbacked`：使用无后盾的 symint，不允许守卫，更容易识别守卫被错误添加的位置
- `backed_size_oblivious`：使用对守卫更严格的模式

关于动态形状模式的更多详情，请参见[动态形状与 vLLM 守卫丢弃](torch_compile.md#dynamic-shapes-and-vllm-guard-dropping)。

### 打印守卫

要查看编译过程中添加的所有守卫，可以使用 `TORCH_LOGS=+dynamic`：

```sh
TORCH_LOGS=+dynamic vllm serve meta-llama/Llama-3.2-1B
```

在日志中查找 `[guard added]`，以查看守卫被添加的位置。这有助于识别哪些操作导致守卫被错误添加。

## 调试 TorchInductor

TorchInductor 接收捕获的图，然后将其编译为一些 Python 代码，这些代码可能调用 1 个或多个 triton 内核。在罕见（但不幸）的情况下，它可能产生不正确的 triton 内核。这可能表现为静默不正确、CUDA 非法内存访问或显式错误。

### Inductor 运行时断言

默认情况下（在 torch < 2.12 上），vLLM 禁用 Inductor 的运行时断言（`assert_size_stride`、`assert_alignment`）以避免大型模型上每次前向传播约 2ms 的开销。设置 `VLLM_LOGGING_LEVEL=DEBUG` 会自动重新启用它们，以便调试会话获得完整的形状/步长验证：

```sh
VLLM_LOGGING_LEVEL=DEBUG vllm serve <model>
```

您也可以通过 `--compilation-config` 显式覆盖它们：

```sh
vllm serve <model> -cc.inductor_compile_config='{"size_asserts": true, "alignment_asserts": true, "scalar_asserts": true}'
```

在 torch >= 2.12 上，PyTorch 使用了高效的断言一次策略，vLLM 不再抑制这些标志。

要调试是否是 TorchInductor 的问题，可以通过在编译配置中传递 `backend='eager'` 来禁用它：

```sh
# 在线
vllm serve -cc.backend=eager
```

```py
# 离线
LLM(compilation_config=CompilationConfig(backend='eager'))
```

如果是 Inductor 的问题，[向 PyTorch 提交 bug](https://github.com/pytorch/pytorch)。如果您有冒险精神，可以在 Inductor 输出代码中调试 triton 内核（您可以通过 tlparse 定位到这些代码）。

![tlparse 示例](../assets/design/debug_vllm_compile/tlparse_inductor.png)

您也可以使用 `TORCH_LOGS=output_code <command>` 来打印 Inductor 输出代码。

### 可编辑的 TorchInductor 代码

您可以通过设置 `VLLM_COMPILE_CACHE_SAVE_FORMAT=unpacked` 或传递 `-cc.compile_cache_save_format=unpacked` 来编辑 TorchInductor 运行的代码。默认值是 `binary`，这意味着不可编辑。

这是一个有用的技巧：您可以在输出代码中设置断点（例如 `torch.distributed.breakpoint()`）和打印语句。

## 调试 vLLM-compile 缓存

vLLM 构建了自己的 torch.compile 产物缓存。其思路是产物可以编译一次，然后在编译后重复使用。这是建立在 [torch.compile 的编译器缓存](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)之上的一个层。

虽然 torch.compile 的编译器缓存非常稳定，但 vLLM 的编译器缓存不幸地并非总是正确。您可以通过设置 `VLLM_DISABLE_COMPILE_CACHE=1` 来禁用它。

您也可以手动删除此缓存。

- 使用 `rm -rf ~/.cache/vllm` 删除 vLLM 的编译缓存（查看日志以确认位置是否改变）
- 使用 `rm -rf /tmp/torchinductor_$(whoami)` 删除 torch.compile 的内置缓存

vLLM 的缓存是从缓存键到编译产物的映射。vLLM 通过组合多个因素（例如配置标志和模型名称）来计算缓存键。如果 vLLM 的编译缓存出错，这通常意味着缺少某个因素。请参见[此示例](https://github.com/vllm-project/vllm/blob/18b39828d90413d05d770dfd2e2f48304f4ca0eb/vllm/config/model.py#L310)，了解 vLLM 如何计算部分缓存键。

vLLM 的编译缓存要求被编译的代码必须是可序列化的。如果不是这种情况，将在保存时报错。通常的修复方法是：

- 重写不可序列化的部分（可能很困难，因为目前很难判断什么是可序列化的，什么不是）
- 提交错误报告
- 通过设置 `VLLM_DISABLE_COMPILE_CACHE=1` 忽略错误（注意这将使热服务器启动慢得多）。

## 调试 CUDA 图

CUDA 图是一个功能，允许：

- 将调用 1 个以上 CUDA 内核的可调用对象捕获到 CUDA 图中
- 重放 CUDA 图

捕获的 CUDA 图包含捕获过程中使用的所有内存。CUDA 图的重放读取和写入完全相同的内存区域。

这带来了一些限制：

1. 为了在新数据上使用 CUDA 图，您需要将数据复制到 CUDA 图正在读取的缓冲区中
2. CUDA 图只捕获 CUDA 内核，不捕获 CPU 上完成的工作。

vLLM 使用原始 CUDA 图 API，如果使用不当，这是不安全的。

仅关闭 CUDA 图，传递 `cudagraph_mode = NONE`：

```sh
# 在线
vllm serve -cc.cudagraph_mode=NONE
```

```py
# 离线
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
LLM(model, compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE))
```
