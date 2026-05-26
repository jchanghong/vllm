# 批处理不变性

!!! note
    批处理不变性目前处于 beta 阶段。某些功能仍在积极开发中。
    跟踪进展和计划改进：<https://github.com/vllm-project/vllm/issues/27433>

本文档介绍如何在 vLLM 中启用批处理不变性。批处理不变性确保模型的输出是确定性的，并且独立于批处理大小或批处理中请求的顺序。

## 动机

批处理不变性对以下几个用例至关重要：

- **框架调试**：确定性输出使得调试推理框架中的问题更加容易，因为相同的输入无论批处理如何，总是产生相同的输出。
- **模型调试**：通过确保在不同批处理配置下行为一致，帮助识别模型实现中的问题。
- **强化学习（RL）**：RL 训练通常需要确定性回放以实现可重现性和稳定训练。
- **大规模推理系统**：使用 vLLM 作为组件的系统受益于确定性行为，用于测试、验证和一致性保证。

## 硬件要求

批处理不变性目前需要计算能力 9.0 或更高的 NVIDIA GPU：

- **H 系列**：H100、H200
- **B 系列**：B100、B200

## 启用批处理不变性

可以通过将 `VLLM_BATCH_INVARIANT` 环境变量设置为 `1` 来启用批处理不变性：

```bash
export VLLM_BATCH_INVARIANT=1
```

### 在线推理（服务器模式）

启动启用批处理不变性的 vLLM 服务器：

```bash
VLLM_BATCH_INVARIANT=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

然后使用 OpenAI 兼容客户端：

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# 这些请求将产生确定性输出
# 无论批处理大小或顺序如何
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.7,
    seed=42,
)

print(response.choices[0].text)
```

### 离线推理

用于具有批处理不变性的离线批处理推理：

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Machine learning enables",
    "Deep learning models can",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100,
    seed=42,
)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
)

# 输出将是确定性的，无论批处理大小如何
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}")
    print(f"生成: {generated_text!r}\n")
```

## 已测试模型

批处理不变性已在以下模型上测试并验证：

- **DeepSeek 系列**：`deepseek-ai/DeepSeek-V3`、`deepseek-ai/DeepSeek-V3-0324`、`deepseek-ai/DeepSeek-R1`、`deepseek-ai/DeepSeek-V3.1`
- **Qwen3（密集）**：`Qwen/Qwen3-1.7B`、`Qwen/Qwen3-8B`、`Qwen/Qwen3-4B-AWQ`、`Qwen/Qwen3-8B-AWQ`
- **Qwen3（MoE）**：`Qwen/Qwen3-30B-A3B`、`Qwen/Qwen3-Next-80B-A3B-Instruct`、`Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`
- **Qwen2.5**：`Qwen/Qwen2.5-0.5B-Instruct`、`Qwen/Qwen2.5-1.5B-Instruct`、`Qwen/Qwen2.5-3B-Instruct`、`Qwen/Qwen2.5-7B-Instruct`、`Qwen/Qwen2.5-14B-Instruct`、`Qwen/Qwen2.5-32B-Instruct`
- **Llama 3**：`meta-llama/Llama-3.1-8B-Instruct`、`meta-llama/Llama-3.2-1B-Instruct`
- **GPT-OSS**：`openai/gpt-oss-20b`、`openai/gpt-oss-120b`
- **Mistral**：`mistralai/Mistral-7B-v0.3`

其他模型也可能有效，但这些已经过明确验证。如果您在使用特定模型时遇到问题，请在 [GitHub 问题跟踪器](https://github.com/vllm-project/vllm/issues/new/choose) 上报告。

## 实现细节

启用批处理不变性时，vLLM 将：

1. 使用确定性的内核实现进行注意力和其他操作
2. 确保在不同批处理大小间数值行为一致
3. 禁用可能引入非确定性的某些优化（例如张量并行模式中的自定义全规约操作）

!!! note
    与默认的非确定性模式相比，启用批处理不变性可能影响性能。这种权衡是为了保证可重现性而有意为之的。

## 未来改进

批处理不变性功能正在积极开发中。计划的改进包括：

- 支持更多 GPU 架构
- 扩展模型覆盖范围
- 性能优化
- 更多测试和验证

有关最新状态和贡献想法，请参见[跟踪问题](https://github.com/vllm-project/vllm/issues/27433)。
