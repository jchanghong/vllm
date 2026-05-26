# 上下文扩展

!!! note
    vLLM 旧版本中使用的 `--rope-scaling` 参数已不再支持。请改用带有 `rope_parameters` 的 `--hf-overrides` 方法。
本目录包含使用 vLLM 扩展模型上下文长度的示例。

## 离线推理示例

[`context_extension.py`](../../examples/features/context_extension/context_extension_offline.py) 脚本演示了如何使用 YARN 方法（rope_parameters）扩展 Qwen 模型的上下文长度，并运行一个简单的聊天示例。

### 用法

```bash
python examples/features/context_extension/context_extension_offline.py
```

## OpenAI 在线方法

您也可以使用 vLLM 的 OpenAI 兼容 API 来提供具有扩展上下文长度的模型服务。

### 用法

使用以下命令运行 vLLM 服务器，通过 YARN 扩展上下文长度：

```bash
vllm serve Qwen/Qwen3-0.6B \
  --hf-overrides '{"rope_parameters": {"factor": 4.0, "original_max_position_embeddings": 32768, "rope_theta": 1000000, "rope_type": "yarn"}}' \
  --max-model-len 131072
```

### 客户端示例

启动服务器后，您可以使用 OpenAI Python 客户端与之交互：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"  # 虚拟 API 密钥，客户端要求提供
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"}
    ],
    max_tokens=128,
    temperature=0.8,
    top_p=0.95
)

print(response.choices[0].message.content)
```

### 关键参数

可用参数取决于您选择的 `rope_type`。有关所有支持的 RoPE 类型及其特定参数的详细信息，请参阅 [Hugging Face Transformers RoPE 文档](https://huggingface.co/docs/transformers/main/en/internal/rope_utils#transformers.RopeParameters)。

常见参数包括：

- `rope_type`：RoPE 实现的类型（例如 "yarn"、"linear"、"dynamic"）
- `factor`：扩展上下文长度的因子
- `original_max_position_embeddings`：模型的原始最大位置嵌入

以下参数特定于 vLLM：

- `max_model_len`：扩展后的新最大序列长度（原始长度 * 因子）。
  用于 KV 缓存的预分配和服务时的请求限制。
