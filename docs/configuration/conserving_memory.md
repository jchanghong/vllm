# 节省内存

大型模型可能会导致机器内存不足（OOM）。以下是一些有助于缓解此问题的选项。

## 张量并行（TP）

张量并行（`tensor_parallel_size` 选项）可用于将模型拆分到多个 GPU 上。

以下代码将模型拆分到 2 个 GPU 上。

```python
from vllm import LLM

llm = LLM(model="ibm-granite/granite-3.1-8b-instruct", tensor_parallel_size=2)
```

!!! warning
    为确保 vLLM 正确初始化 CUDA，您应避免在初始化 vLLM 之前调用相关函数（例如 [torch.accelerator.set_device_index][]）。否则，您可能会遇到类似 `RuntimeError: Cannot re-initialize CUDA in forked subprocess` 的错误。

    要控制使用哪些设备，请改为设置 `CUDA_VISIBLE_DEVICES` 环境变量。

!!! note
    启用张量并行后，每个进程都会读取整个模型并将其分块，这使得磁盘读取时间更长（与张量并行的大小成正比）。

    您可以使用[示例/功能/分片状态/离线加载分片状态.py](../../examples/features/sharded_state/load_sharded_state_offline.py)将模型检查点转换为分片检查点。转换过程可能需要一些时间，但之后您可以更快地加载分片检查点。无论张量并行大小如何，模型加载时间应保持恒定。

## 量化

量化模型以较低精度为代价，占用更少内存。

静态量化模型可以从 HF Hub 下载（一些流行的模型可在 [Red Hat AI](https://huggingface.co/RedHatAI) 获取）并直接使用，无需额外配置。

通过 `quantization` 选项也支持动态量化——详情请参见[此处](../features/quantization/README.md)。

## 上下文长度和批处理大小

您可以通过限制模型的上下文长度（`max_model_len` 选项）和最大批处理大小（`max_num_seqs` 选项）来进一步减少内存使用。

```python
from vllm import LLM

llm = LLM(model="adept/fuyu-8b", max_model_len=2048, max_num_seqs=2)
```

## 减少 CUDA 图

默认情况下，我们使用 CUDA 图来优化模型推理，这会在 GPU 上占用额外的内存。

您可以通过调整 `compilation_config` 来更好地平衡推理速度和内存使用：

??? code

    ```python
    from vllm import LLM
    from vllm.config import CompilationConfig, CompilationMode

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            # 默认情况下，它最多达到 max_num_seqs
            cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        ),
    )
    ```

您可以通过 `enforce_eager` 标志完全禁用图捕获：

```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enforce_eager=True)
```

## 调整缓存大小

如果 CPU RAM 耗尽，请尝试以下选项：

- （仅限多模态模型）您可以通过设置 `mm_processor_cache_gb` 引擎参数（默认为 4 GiB）来调整多模态缓存的大小。
- （仅限 CPU 后端）您可以使用 `VLLM_CPU_KVCACHE_SPACE` 环境变量（默认为 4 GiB）设置 KV 缓存的大小。

## 多模态输入限制

您可以允许每个提示使用更少数量的多模态项目，以减少模型的内存占用：

```python
from vllm import LLM

# 每个提示最多接受 3 张图片和 1 个视频
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"image": 3, "video": 1},
)
```

您可以更进一步，通过将其限制设置为零来完全禁用未使用的模态。
例如，如果您的应用程序仅接受图像输入，则无需为视频分配任何内存。

```python
from vllm import LLM

# 接受任意数量的图像，但不接受视频
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"video": 0},
)
```

您甚至可以让多模态模型仅进行文本推理：

```python
from vllm import LLM

# 不接受图像。仅文本。
llm = LLM(
    model="google/gemma-3-27b-it",
    limit_mm_per_prompt={"image": 0},
)
```

### 可配置选项

`limit_mm_per_prompt` 还接受每个模态的可配置选项。在可配置形式中，您仍然需要指定 `count`，并且可以选择提供大小提示，这些提示控制 vLLM 如何分析和预留多模态输入的内存。这有助于您根据期望的实际媒体调整内存，而不是使用模型的绝对最大值。

按模态划分的可配置选项：

- `image`：`{"count": int, "width": int, "height": int}`
- `video`：`{"count": int, "num_frames": int, "width": int, "height": int}`
- `audio`：`{"count": int, "length": int}`

详细信息请参见 [`ImageDummyOptions`][vllm.config.multimodal.ImageDummyOptions]、[`VideoDummyOptions`][vllm.config.multimodal.VideoDummyOptions] 和 [`AudioDummyOptions`][vllm.config.multimodal.AudioDummyOptions]。

示例：

```python
from vllm import LLM

# 每个提示最多 5 张图片，以 512x512 进行分析。
# 每个提示最多 1 个视频，以 640x640 的 32 帧进行分析。
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={
        "image": {"count": 5, "width": 512, "height": 512},
        "video": {"count": 1, "num_frames": 32, "width": 640, "height": 640},
    },
)
```

为了向后兼容，传递整数时的行为与之前相同，将被解释为 `{"count": <int>}`。例如：

- `limit_mm_per_prompt={"image": 5}` 等同于 `limit_mm_per_prompt={"image": {"count": 5}}`
- 您可以混合格式：`limit_mm_per_prompt={"image": 5, "video": {"count": 1, "num_frames": 32, "width": 640, "height": 640}}`

!!! note
    - 大小提示仅影响内存分析。它们会影响用于计算预留激活大小的虚拟输入。它们不会改变推理时实际处理输入的方式。
    - 如果提示超过模型能接受的范围，vLLM 会将其限制为模型的有效最大值，并可能会记录警告。

!!! warning
    这些大小提示目前仅影响激活内存分析。编码器缓存大小由运行时的实际输入决定，不受这些提示限制。

## 多模态处理器参数

对于某些模型，您可以调整多模态处理器参数以减小处理后的多模态输入的大小，从而节省内存。

以下是一些示例：

```python
from vllm import LLM

# 适用于 Qwen2-VL 系列模型
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    mm_processor_kwargs={"max_pixels": 768 * 768},  # 默认为 1280 * 28 * 28
)

# 适用于 InternVL 系列模型
llm = LLM(
    model="OpenGVLab/InternVL2-2B",
    mm_processor_kwargs={"max_dynamic_patch": 4},  # 默认为 12
)
```
