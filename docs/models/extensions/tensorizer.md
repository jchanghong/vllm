# 使用 CoreWeave 的 Tensorizer 加载模型

vLLM 支持使用 [CoreWeave 的 Tensorizer](https://docs.coreweave.com/coreweave-machine-learning-and-ai/inference/tensorizer) 加载模型。
已序列化到磁盘、HTTP/HTTPS 端点或 S3 端点的 vLLM 模型张量可以在运行时极快地直接反序列化到 GPU，
从而显著缩短 Pod 启动时间并降低 CPU 内存使用。还支持张量加密。

vLLM 将 Tensorizer 完全集成到其模型加载机制中。以下内容将简要介绍如何在 vLLM 上开始使用 Tensorizer。

## 安装 Tensorizer

要安装 `tensorizer`，请运行 `pip install vllm[tensorizer]`。

## 基础

要使用 Tensorizer 加载模型，首先需要使用 Tensorizer 对模型进行序列化。
[示例脚本](../../../examples/features/tensorize_vllm_model.py) 负责处理此过程。

让我们通过一个基本示例来演练：使用该脚本序列化 `facebook/opt-125m`，然后加载它进行推理。

## 使用 Tensorizer 序列化 vLLM 模型

要使用 Tensorizer 序列化模型，请使用必要的 CLI 参数调用示例脚本。脚本本身的文档字符串详细解释了 CLI 参数
及其正确使用方法。我们直接使用文档字符串中的一个示例，假设我们要将模型序列化并保存到 S3 存储桶 `s3://my-bucket`：

```bash
python examples/features/tensorize_vllm_model.py \
   --model facebook/opt-125m \
   serialize \
   --serialized-directory s3://my-bucket \
   --suffix v1
```

这会将模型张量保存到 `s3://my-bucket/vllm/facebook/opt-125m/v1`。如果您打算将 LoRA 适配器应用于张量化模型，可以在上述命令中传入 LoRA 适配器的 HF ID，相关工件也会保存到该位置：

```bash
python examples/features/tensorize_vllm_model.py \
   --model facebook/opt-125m \
   --lora-path <lora_id> \
   serialize \
   --serialized-directory s3://my-bucket \
   --suffix v1
```

## 使用 Tensorizer 提供模型服务

一旦模型被序列化到所需位置，您就可以使用 `vllm serve` 或 `LLM` 入口点加载模型。您可以将保存模型的目录传递给 `LLM()` 和 `vllm serve` 的 `model` 参数。例如，要提供之前保存的带有 LoRA 适配器的张量化模型，您可以执行：

```bash
vllm serve s3://my-bucket/vllm/facebook/opt-125m/v1 \
    --load-format tensorizer \
    --enable-lora 
```

或者，使用 `LLM()`：

```python
from vllm import LLM
llm = LLM(
    "s3://my-bucket/vllm/facebook/opt-125m/v1", 
    load_format="tensorizer",
    enable_lora=True,
)
```

## Tensorizer 配置选项

`tensorizer` 中负责序列化和反序列化模型的核心对象分别是 `TensorSerializer` 和 `TensorDeserializer`。要向其传递任意 kwargs 以配置序列化和反序列化过程，您可以通过 `model_loader_extra_config` 分别使用 `serialization_kwargs` 和 `deserialization_kwargs` 作为键来提供。上述对象所有参数的完整文档字符串可以在 `tensorizer` 的 [serialization.py](https://github.com/coreweave/tensorizer/blob/main/tensorizer/serialization.py) 文件中找到。

例如，在使用 `tensorizer` 序列化时，可以通过 `TensorSerializer` 初始化器中的 `limit_cpu_concurrency` 参数限制 CPU 并发数。要设置 `limit_cpu_concurrency` 为某个任意值，您可以在序列化时这样操作：

```bash
python examples/features/tensorize_vllm_model.py \
   --model facebook/opt-125m \
   --lora-path <lora_id> \
   serialize \
   --serialized-directory s3://my-bucket \
   --serialization-kwargs '{"limit_cpu_concurrency": 2}' \
   --suffix v1
```

例如，在自定义通过 `TensorDeserializer` 的加载过程时，您可以通过 `model_loader_extra_config` 在初始化器中设置 `num_readers` 参数来限制反序列化期间的并发读取器数量，如下所示：

```bash
vllm serve s3://my-bucket/vllm/facebook/opt-125m/v1 \
    --load-format tensorizer \
    --enable-lora \
    --model-loader-extra-config '{"deserialization_kwargs": {"num_readers": 2}}'
```

或者使用 `LLM()`：

```python
from vllm import LLM
llm = LLM(
    "s3://my-bucket/vllm/facebook/opt-125m/v1", 
    load_format="tensorizer",
    enable_lora=True,
    model_loader_extra_config={"deserialization_kwargs": {"num_readers": 2}},
)
```
