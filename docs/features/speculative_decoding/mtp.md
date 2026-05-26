# MTP（多 token 预测）

MTP 是一种投机解码方法，其中目标模型包含原生的多 token 预测能力。与基于草稿模型的方法不同，您无需提供单独的草稿模型。

在以下情况下，MTP 非常有用：

- 您的模型原生支持 MTP。
- 您希望以最少的额外配置实现基于模型的投机解码。

## Gemma 4 辅助模型

Gemma 4 辅助检查点使用 vLLM 的 Gemma 4 MTP 路径。它们不是通用的草稿模型，尽管它们是通过 `--speculative-config` 中的 `model` 字段传入的。

在服务带辅助检查点的 Gemma 4 时使用 `"method": "mtp"`：

```bash
vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --speculative-config '{"method":"mtp","model":"gg-hf-am/gemma-4-E2B-it-assistant","num_speculative_tokens":1}'
```

E2B、E4B、26B-A4B 和 31B Gemma 4 IT 辅助检查点在其配置使用 `model_type: gemma4_assistant` 时均受支持。vLLM 会将这些检查点映射到 `Gemma4MTPModel`，并将辅助层与目标模型共享 KV 缓存。

如果较旧的 vLLM 版本针对 Gemma 4 辅助检查点记录 `SpeculativeConfig(method='draft_model', ...)`，则说明该版本将辅助检查点视为通用草稿模型，并且可能在对多模态 Gemma 4 目标进行初始化时失败。请升级到具有 Gemma 4 MTP 支持的版本。

## 离线示例

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## 在线示例

```bash
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --tensor-parallel-size 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## 说明

- MTP 仅适用于 vLLM 中支持 MTP 的模型家族。
- `num_speculative_tokens` 控制推测深度。像 `1` 这样的小值是良好的起始默认值。
- 如果您的模型不支持 MTP，请使用其他方法，例如 EAGLE 或草稿模型推测。
