# `torch.compile` 集成

在 vLLM 的 V1 架构中，`torch.compile` 默认启用，是该框架的关键组成部分。本文档提供了一个简单的逐步示例，展示如何理解 `torch.compile` 的使用。

在整个示例中，我们将运行一个常见的 Llama 模型，并开启调试级日志以显示所有细节。使用的命令是 `VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B`。

!!! note
    更多关于 `torch.compile` 集成的信息和最新进展，请参见这篇[博客文章](https://blog.vllm.ai/2025/08/20/torch-compile.html)。

## 编译缓存

在非常详细的日志中，我们可以看到：

```console
INFO 03-07 03:06:55 [backends.py:409] Using cache directory: ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0 for vLLM's torch.compile
```

vLLM 将考虑所有可用因素，并决定一个目录来存储所有编译产物。这意味着，您可以直接将整个 `~/.cache/vllm/torch_compile_cache` 目录复制到您的部署场景中，以节省大量编译时间，从而加速 vLLM 实例的启动时间。

考虑的因素包括：

- 所有相关配置（请参见 [config 文件夹](../../vllm/config)中各自配置的 `compute_hash` 函数）
- PyTorch 配置（请参见 [compiler_interface.py](../../vllm/compilation/compiler_interface.py) 中的 `compute_hash` 函数）
- 模型的前向函数以及前向函数调用的相关函数（见下文）

考虑了所有这些因素后，通常我们可以保证缓存使用是安全的，不会导致任何意外行为。因此，缓存默认启用。如果您想调试编译过程，或者怀疑缓存导致了某些问题，可以通过设置环境变量 `VLLM_DISABLE_COMPILE_CACHE=1` 禁用它。

vLLM 的 `torch.compile` 集成的一个独特方面是，我们保证在处理任何请求之前完成所有编译。不会有请求触发新的编译。否则，引擎将被该请求阻塞，响应时间将出现意外的尖峰。

默认情况下，缓存将编译产物保存为二进制文件。如果您想与生成的代码进行交互以进行调试，请在编译配置中设置 `compile_cache_save_format=unpacked`，或省略此设置并设置环境变量 `VLLM_COMPILE_CACHE_SAVE_FORMAT=unpacked`。

## 动态形状与 vLLM 守卫丢弃

`torch.compile` 设计为在需要时毫不犹豫地对动态形状进行守卫。这与 vLLM 的 `torch.compile` 方法相矛盾，后者会丢弃守卫，因为许多守卫可能是实质性的。

`torch.compile` 提供两种动态形状：`backed` 和 `unbacked`。
`torch.compile` 对 `backed` 动态形状进行守卫，并且不保证不会向它们添加守卫。用户代码、dynamo、inductor 和 autograd 都可以添加守卫。此外，对于 0/1 特化，即使没有遇到这些范围的分支，backed 符号也会无条件地特化为 0、1 或 >=2。

相反，`unbacked` 动态形状保证不会被守卫，并且不会进行 0/1 特化。然而，当遇到需要其值的分支时，如果没有定义显式的 unbacked 处理，可能会抛出数据依赖错误。框架正趋于一种不会抛出 DDE 而是选择通用路径的状态。使用 unbacked 的一个缺点是由于性能错误或选择通用路径而错失优化机会，此外还使用固定的非示例输入提示（这将在不久的将来通过 override_hint API 修复）。选择通用路径的一个示例是，当无法通过引入克隆的更改符号化证明连续性时，在函数中调用 `contiguous()` 和 `reshape()` 时假设输入不连续。

`backed_size_oblivious` 是一个标志，允许在定义了显式 unbacked 处理的地方将 backed 符号视为 unbacked。使用此模式，框架代码中大多避免了 0/1 特化，并且默认的 0/1 特化不会发生。然而，仍然无法保证 torch.compile 不会进行守卫，特别是由于用户代码或自定义传递。`backed_size_oblivious` 在 PyTorch 编译中是实验性的，可能会被弃用。尽管如此，它是比 `backed` 更安全的选择，并且降低性能的概率低于 `unbacked`。

### 配置动态形状

`DynamicShapesConfig` 允许您通过设置 `type` 字段来控制动态形状行为。您可以在三种模式之间选择：`BACKED`（默认）、`UNBACKED` 和 `BACKED_SIZE_OBLIVIOUS`。

#### 离线推理示例（使用 LLM 类）

使用 `LLM` 类进行离线推理时，可以通过 `compilation_config` 参数配置动态形状：

```python
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, DynamicShapesConfig, DynamicShapesType

# 示例：使用 backed_size_oblivious（实验性，比 backed 更安全）
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    compilation_config=CompilationConfig(
        dynamic_shapes_config=DynamicShapesConfig(
            type=DynamicShapesType.BACKED_SIZE_OBLIVIOUS
        )
    )
)

# 示例：使用 unbacked（最强的防护保证）
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    compilation_config=CompilationConfig(
        dynamic_shapes_config=DynamicShapesConfig(
            type=DynamicShapesType.UNBACKED
        )
    )
)

# 生成输出
prompts = ["Hello, my name is", "The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
```

#### 在线服务示例（使用 vllm serve）

使用 `vllm serve` 进行在线服务时，可以通过 `--compilation-config` 标志配置动态形状：

```bash
# 示例：使用 unbacked
vllm serve meta-llama/Llama-3.2-1B \
  --compilation-config '{"dynamic_shapes_config": {"type": "unbacked"}}'


# 替代方式：使用点符号（对单个值更简单）
vllm serve meta-llama/Llama-3.2-1B -cc.dynamic_shapes_config.type=unbacked
```

#### 选择合适的模式

- **BACKED**（默认）：当您愿意接受潜在的不安全守卫丢弃以获得最大性能时使用。守卫可能被不合理地添加然后被忽略。

- **UNBACKED**：当您需要最强的防止守卫的保证时使用。这是最保守的选项，但可能错过一些优化机会。

- **BACKED_SIZE_OBLIVIOUS**：当您希望在避免守卫和性能之间取得平衡时使用。这种实验性模式比 BACKED 更安全，但仍不如 UNBACKED 保守。

## Python 代码编译

在非常详细的日志中，我们可以看到：

??? console "日志"

      ```text
      DEBUG 03-07 03:06:52 [decorators.py:203] Start compiling function <code object forward at 0x7f08acf40c90, file "xxx/vllm/model_executor/models/llama.py", line 339>

      DEBUG 03-07 03:06:54 [backends.py:370] Traced files (to be considered for compilation cache):
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/_dynamo/polyfills/builtins.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/container.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/module.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/attention/layer.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/communication_op.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/parallel_state.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/custom_op.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/activation.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/layernorm.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/linear.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/rotary_embedding.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/vocab_parallel_embedding.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/models/llama.py

      DEBUG 03-07 03:07:07 [backends.py:462] Computation graph saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py
      DEBUG 03-07 03:07:07 [wrapper.py:105] Dynamo transformed code saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py
      ```

这是关于 Python 代码编译的，即 Dynamo 的图捕获。它尝试使用 `xxx/vllm/model_executor/models/llama.py:339` 处的代码追踪函数，这是我们编译的模型的 `forward` 函数。在前向传播过程中，还有其他函数被 Dynamo 调用和内联，如日志所示，包括来自 `xxx/torch/nn/modules/module.py` 的一些 PyTorch 函数（由 PyTorch `nn.Module` 使用，因为模块属性访问会触发函数调用）、来自 vLLM 的一些通信/注意力/激活函数。所有被追踪的文件将在我们决定缓存目录时被考虑。这样，上述文件中任何代码更改都将触发编译缓存未命中，从而导致重新编译。

Dynamo 编译的结果是一个新函数，存储在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py` 中。通常，此函数从模块中解包张量，然后将其传递给追踪的计算图。计算图存储在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py` 中。

## 计算图处理

计算图为每个张量带有形状标注。输入是 input ids、position ids、模型的权重和缓冲区，输出是最终的隐藏状态。请注意，lm head 投影和采样操作不在图中考虑。

计算图的大多数输入具有静态形状，因为它们是模型权重和缓冲区，在模型生命周期内不会改变。只有 input ids 和 position ids 具有符号形状，即形状可以在不同批次之间变化。然而，它们将共享相同的符号形状。也就是说，计算图的唯一可变大小是批量大小（当前前向传播中处理的 token 数）。

注意力操作很复杂，并且需要与 KV 缓存交互，具有复杂的形状。幸运的是，注意力操作的输出与注意力操作的输入查询共享相同的形状。因此，我们将整个注意力操作包装成一个 PyTorch 自定义算子 `torch.ops.vllm.unified_attention_with_output`，这样 Dynamo 将不会尝试检查任何内部操作。这样，尽管注意力操作很复杂，我们仍然可以从 Dynamo 的角度将模型的计算图作为一个完整图来捕获。

计算图进一步被 `splitting_ops`（通常是注意力操作）分割成多个片段。因此，在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py` 文件中，我们可以看到许多子模块，每个子模块是分割后的图片段：

- 注意力操作本身是一个子模块。
- 从一次注意力操作到下一次注意力操作之间的计算图部分是一个子模块。

每个子模块可以通过其索引识别，并将被单独处理。

## 计算图编译

在非常详细的日志中，我们还可以看到：

```console
DEBUG 03-07 03:52:37 [backends.py:134] store the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
DEBUG 03-07 03:52:39 [backends.py:134] store the 1-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
...
DEBUG 03-07 03:52:45 [backends.py:134] store the 15-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
DEBUG 03-07 03:52:45 [backends.py:134] store the 16-th graph for shape None from inductor via handle ('fvj3ccoi7m34f3dnr4itmu55mmun44l5xymwhrjlwisylsk7q6jy', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/tf/ctfftkglj7b4lcttq5cymx6cew372uoauupqn6ldsvpiucavqcjc.py')
```

这意味着计算图的第一片段（符号形状为 `None`）由 Inductor 编译（带有键 `fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw`）。编译后的内核存储在 `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py` 中。您可以打开该文件查看 Inductor 最终运行的代码。

还有一个细节：您可以看到第 1 个图和第 15 个图具有相同的键，而第 0 个图和第 16 个图不同。这是预期的，因为我们按注意力操作分割图后，得到 3 个唯一的子图：

- 注意力之前的第一个层
- 从一次注意力操作到下一次注意力操作的每个中间层
- 注意力之后的最后一个层

如果我们已经有了缓存目录（例如第二次运行相同的代码），我们将看到以下日志：

```console
DEBUG 03-07 04:00:45 [backends.py:86] Directly load the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
```

这次，Inductor 编译被完全绕过，我们将从磁盘加载上次得到的编译产物。

上述示例仅使用 Inductor 为通用形状（即符号形状）进行编译。我们也可以使用 Inductor 为某些特定形状进行编译，例如：

```bash
vllm serve meta-llama/Llama-3.2-1B \
  --compilation_config '{"compile_sizes": [1, 2, 4, 8]}'
```

然后它还将为批量大小 `1, 2, 4, 8` 编译特定的内核。此时，计算图中的所有形状都是静态和已知的，我们将开启自动调优以获得最大性能。这在首次运行时可能很慢，但下次运行时，我们可以直接绕过调优并运行调优后的内核。

当所有形状已知时，`torch.compile` 可以比较不同的配置，并经常找到一些更好的配置来运行内核。例如，我们可以看到以下日志：

??? console "日志"

    ```
    AUTOTUNE mm(8x2048, 2048x3072)
      triton_mm_4 0.0130 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
      triton_mm_8 0.0134 ms 97.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
      triton_mm_12 0.0148 ms 87.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=4, num_warps=4
      mm 0.0160 ms 81.6%
      triton_mm_16 0.0165 ms 78.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=8
      triton_mm_3 0.0199 ms 65.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
      triton_mm_1 0.0203 ms 64.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=2, num_warps=2
      triton_mm_7 0.0203 ms 64.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
      triton_mm_2 0.0208 ms 62.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
      triton_mm_11 0.0215 ms 60.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 2.0428 seconds and 7.5727 seconds precompiling
    ```

这意味着，对于形状为 `8x2048x3072` 的矩阵乘法，`torch.compile` 尝试了各种配置的 triton 模板，并且比默认代码（分派到 cublas 库）快得多。

不幸的是，因为自动调优需要相当长的时间（从几秒到几分钟，取决于模型大小和批量大小），即使它可以缓存供以后使用，但为了用户友好，我们默认将其关闭。如果您想要最大性能，建议尝试它，通过编译特定的形状。

## CUDA 图捕获

vLLM 的 V1 架构使用与逐段编译对齐的逐段 CUDA 图。完整计算图被分割，如上所述，我们只为注意力操作之间的图片段捕获 CUDA 图（包括任何注意力操作之前的第一个图和所有注意力操作之后的最后一个图）。这是基于一个常见观察：注意力之间的计算通常是 token 级别的，易于 CUDA 图处理；而注意力操作本身很难与 CUDA 图兼容。因此，通过在即时模式下运行注意力操作，同时在 CUDA 图中运行其余操作，我们保持了注意力操作的灵活性。

逐段 CUDA 图还具有细粒度的内存管理。其目的是只将注意力内核排除在 CUDA 图之外，同时将所有其余模块和内存分配操作保留在 CUDA 图中。这就是 V1 中注意力操作将输出张量作为注意力输入的原因。

CUDA 图由编译器后端捕获和管理，当批量大小有对应的 CUDA 图捕获时进行重放。模型的调用者（模型运行器）只需确保正确管理输入缓冲区。所有中间缓冲区由编译器后端自动管理。

默认情况下，vLLM 将尝试确定一组大小来捕获 CUDA 图。您也可以使用配置 `cudagraph_capture_sizes` 覆盖它：

```bash
vllm serve meta-llama/Llama-3.2-1B \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8]}'
```

然后它将仅为指定的大小捕获 CUDA 图。这对于对 CUDA 图捕获进行细粒度控制非常有用。

### 完整 CUDA 图捕获

如果使用兼容 CUDA 图的注意力后端，可以将注意力包含为 CUDA 图的一部分。这可以在某些情况下提高性能，例如较小模型或 MoE 的解码速度。更多详情请参见 [CUDA 图](cuda_graphs.md)。
