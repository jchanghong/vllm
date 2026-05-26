# 奖励用途

奖励模型 (RM) 旨在评估和评分语言模型生成的输出质量，作为人类偏好的代理。

## 摘要

- 模型用途：奖励
- 池化任务：

| 模型类型 | 池化任务 |
|------------------------------------|----------------|
| (序列) (结果) 奖励模型 | classify |
| token (结果) 奖励模型 | token_classify |
| 过程奖励模型 | token_classify |

- 离线 API：
    - `LLM.encode(..., pooling_task="...")`
- 在线 API：
    - Pooling API (`/pooling`)

## 支持的模型

### 奖励模型

使用序列分类模型作为 (序列) (结果) 奖励模型，其用法和支持的功能与普通的[分类模型](classify.md)相同。

--8<-- [start:supported-sequence-reward-models]

| 架构 | 模型 | 示例 HF 模型 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `JambaForSequenceClassification` | Jamba | `ai21labs/Jamba-tiny-reward-dev` 等 | ✅︎ | ✅︎ |
| `Qwen3ForSequenceClassification`<sup>C</sup> | 基于 Qwen3 | `Skywork/Skywork-Reward-V2-Qwen3-0.6B` 等 | ✅︎ | ✅︎ |
| `LlamaForSequenceClassification`<sup>C</sup> | 基于 Llama | `Skywork/Skywork-Reward-V2-Llama-3.2-1B` 等 | ✅︎ | ✅︎ |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup> 等 | 生成式模型 | N/A | \* | \* |

<sup>C</sup> 通过 `--convert classify` 自动转换为分类模型。([详细信息](./README.md#model-conversion))  

如果您的模型不在上述列表中，我们将尝试使用 [as_seq_cls_model][vllm.model_executor.models.adapters.as_seq_cls_model] 自动转换模型。默认情况下，类别概率从最后一个 token 对应的 softmax 化隐藏状态中提取。

--8<-- [end:supported-sequence-reward-models]

### Token 奖励模型

(序列) 分类和 token 分类之间的关键区别在于它们的输出粒度：(序列) 分类为整个输入序列生成单个结果，而 token 分类为序列中的每个单独 token 生成结果。

使用 token 分类模型作为 token (结果) 奖励模型，其用法和支持的功能与普通的[token 分类模型](token_classify.md)相同。

--8<-- [start:supported-token-reward-models]

| 架构 | 模型 | 示例 HF 模型 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `InternLM2ForRewardModel` | 基于 InternLM2 | `internlm/internlm2-1_8b-reward`, `internlm/internlm2-7b-reward` 等 | ✅︎ | ✅︎ |
| `Qwen2ForRewardModel` | 基于 Qwen2 | `Qwen/Qwen2.5-Math-RM-72B` 等 | ✅︎ | ✅︎ |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup> 等 | 生成式模型 | N/A | \* | \* |

<sup>C</sup> 通过 `--convert classify` 自动转换为分类模型。([详细信息](./README.md#model-conversion))  

如果您的模型不在上述列表中，我们将尝试使用 [as_seq_cls_model][vllm.model_executor.models.adapters.as_seq_cls_model] 自动转换模型。

--8<-- [end:supported-token-reward-models]

### 过程奖励模型

用于评估中间步骤的过程奖励模型对于实现期望的结果至关重要。

| 架构 | 模型 | 示例 HF 模型 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `LlamaForCausalLM` | 基于 Llama | `peiyi9979/math-shepherd-mistral-7b-prm` 等 | ✅︎ | ✅︎ |
| `Qwen2ForProcessRewardModel` | 基于 Qwen2 | `Qwen/Qwen2.5-Math-PRM-7B` 等 | ✅︎ | ✅︎ |

!!! important
    对于过程监督奖励模型，例如 `peiyi9979/math-shepherd-mistral-7b-prm`，应显式设置池化配置，
    例如：`--pooler-config '{"pooling_type": "STEP", "step_tag_id": 123, "returned_token_ids": [456, 789]}'`。

## 离线推理

### 池化参数

支持以下[池化参数][vllm.PoolingParams]。

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:classify-pooling-params"
```

### `LLM.encode`

[encode][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.encode] 方法适用于 vLLM 中的所有池化模型。

- 奖励模型

为 (序列) (结果) 奖励模型使用 `LLM.encode` 时设置 `pooling_task="classify"`：

```python
from vllm import LLM

llm = LLM(model="Skywork/Skywork-Reward-V2-Qwen3-0.6B", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="classify")

data = output.outputs.data
print(f"Data: {data!r}")
```

- Token 奖励模型

为 token (结果) 奖励模型使用 `LLM.encode` 时设置 `pooling_task="token_classify"`：

```python
from vllm import LLM

llm = LLM(model="internlm/internlm2-1_8b-reward", runner="pooling", trust_remote_code=True)
(output,) = llm.encode("Hello, my name is", pooling_task="token_classify")

data = output.outputs.data
print(f"Data: {data!r}")
```

- 过程奖励模型

为 token (结果) 奖励模型使用 `LLM.encode` 时设置 `pooling_task="token_classify"`：

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-Math-PRM-7B", runner="pooling")
(output,) = llm.encode("Hello, my name is<extra_0><extra_0><extra_0>", pooling_task="token_classify")

data = output.outputs.data
print(f"Data: {data!r}")
```

## 在线服务

请参考 [Pooling API](README.md#pooling-api)。与奖励模型类型对应的池化任务请参考[上表](#summary)。

## 更多示例

更多示例请参见：[examples/pooling/reward](../../../examples/pooling/reward)

## 已弃用的功能

### `LLM.reward`

`llm.reward` API 已弃用，将在 v0.23 中移除。请改用带有 `pooling_task="classify"` 或 `pooling_task="token_classify"` 的 `LLM.encode`。
