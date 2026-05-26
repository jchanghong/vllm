# BitsAndBytes

vLLM 现在支持 [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) 以实现更高效的模型推理。
BitsAndBytes 通过量化模型来减少内存使用并提升性能，同时不会显著损失精度。
与其他量化方法相比，BitsAndBytes 无需使用输入数据对量化模型进行校准。

以下是在 vLLM 中使用 BitsAndBytes 的步骤。

```bash
pip install bitsandbytes>=0.49.2
```

vLLM 会读取模型的配置文件，并同时支持即时量化和预量化检查点。

你可以在 [Hugging Face](https://huggingface.co/models?search=bitsandbytes) 上找到 bitsandbytes 量化模型。
通常，这些仓库会包含一个包含 `quantization_config` 部分的 `config.json` 文件。

## 读取量化检查点

对于预量化的检查点，vLLM 会尝试从配置文件中推断量化方法，因此你无需显式指定量化参数。

```python
from vllm import LLM
import torch
# unsloth/tinyllama-bnb-4bit 是一个预量化检查点。
model_id = "unsloth/tinyllama-bnb-4bit"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
```

## 即时量化：以 4 位量化加载

对于使用 BitsAndBytes 的即时 4 位量化，你需要显式指定量化参数。

```python
from vllm import LLM
import torch
model_id = "huggyllama/llama-7b"
llm = LLM(
    model=model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization="bitsandbytes",
)
```

## OpenAI 兼容服务器

在模型参数后追加以下内容以进行 4 位即时量化：

```bash
--quantization bitsandbytes
```
