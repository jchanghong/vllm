# Token 分类用途

## 摘要

- 模型用途：token 分类
- 池化任务：`token_classify`
- 离线 API：
    - `LLM.encode(..., pooling_task="token_classify")`
- 在线 API：
    - Pooling API (`/pooling`)

(序列) 分类和 token 分类之间的关键区别在于它们的输出粒度：(序列) 分类为整个输入序列生成单个结果，而 token 分类为序列中的每个单独 token 生成结果。

许多分类模型同时支持 (序列) 分类和 token 分类。有关 (序列) 分类的更多详细信息，请参阅[此页面](classify.md)。

!!! note

    池化多任务支持自 v0.21 起已移除。当默认的池化任务 (classify) 不是您想要的时，您需要手动指定，可以通过离线的 `PoolerConfig(task="token_classify")` 或在线的 `--pooler-config.task token_classify`。

## 典型用例

### 命名实体识别 (NER)

实现示例请参见：

离线：[examples/pooling/token_classify/ner_offline.py](../../../examples/pooling/token_classify/ner_offline.py)

在线：[examples/pooling/token_classify/ner_online.py](../../../examples/pooling/token_classify/ner_online.py)

### 强制对齐

强制对齐以音频和参考文本作为输入，并生成词级时间戳。

离线：[examples/pooling/token_classify/forced_alignment_offline.py](../../../examples/pooling/token_classify/forced_alignment_offline.py)

### 稀疏检索（词汇匹配）

BAAI/bge-m3 模型利用 token 分类进行稀疏检索。更多信息请参见[此页面](specific_models.md#baaibge-m3)。

## 支持的模型

| 架构 | 模型 | 示例 HF 模型 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | --------------------------- | --------------------------------------- |
| `BertForTokenClassification` | 基于 bert | `boltuix/NeuroBERT-NER` (见附注) 等 | | |
| `ErnieForTokenClassification` | 类 BERT 中文 ERNIE | `gyr66/Ernie-3.0-base-chinese-finetuned-ner` | | |
| `ModernBertForTokenClassification` | 基于 ModernBERT | `disham993/electrical-ner-ModernBERT-base` | | |
| `Qwen3ForTokenClassification`<sup>C</sup> | 基于 Qwen3 | `bd2lcco/Qwen3-0.6B-finetuned` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup> 等 | 生成式模型 | N/A | \* | \* |

<sup>C</sup> 通过 `--convert classify` 自动转换为分类模型。([详细信息](./README.md#model-conversion))
\* 功能支持与原模型相同。

如果您的模型不在上述列表中，我们将尝试使用 [as_seq_cls_model][vllm.model_executor.models.adapters.as_seq_cls_model] 自动转换模型。默认情况下，类别概率从最后一个 token 对应的 softmax 化隐藏状态中提取。

### 多模态模型

!!! note
    有关多模态模型输入的更多信息，请参见[此页面](../supported_models.md#list-of-multimodal-language-models)。

| 架构 | 模型 | 输入 | 示例 HF 模型 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| --------------------------------------------- | ------------------- | ----------------- | ------------------------------------------ | ------------------------------ | ------------------------------------------ |
| `Qwen3ASRForcedAlignerForTokenClassification` | Qwen3-ForcedAligner | T + A<sup>+</sup> | `Qwen/Qwen3-ForcedAligner-0.6B` (见附注) | | ✅︎ |

!!! note
    强制对齐的使用需要 `--hf-overrides '{"architectures": ["Qwen3ASRForcedAlignerForTokenClassification"]}'`。
    请参考 [examples/pooling/token_classify/forced_alignment_offline.py](../../../examples/pooling/token_classify/forced_alignment_offline.py)。

### 奖励模型

使用 token 分类模型作为奖励模型。有关奖励模型的详细信息，请参见[奖励模型](reward.md)。

--8<-- "docs/models/pooling_models/reward.md:supported-token-reward-models"

## 离线推理

### 池化参数

支持以下[池化参数][vllm.PoolingParams]。

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:classify-pooling-params"
```

### `LLM.encode`

[encode][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.encode] 方法适用于 vLLM 中的所有池化模型。

为 token 分类模型使用 `LLM.encode` 时设置 `pooling_task="token_classify"`：

```python
from vllm import LLM

llm = LLM(model="boltuix/NeuroBERT-NER", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="token_classify")

data = output.outputs.data
print(f"Data: {data!r}")
```

## 在线服务

请参考 [Pooling API](README.md#pooling-api) 并使用 `"task":"token_classify"`。

## 更多示例

更多示例请参见：[examples/pooling/token_classify](../../../examples/pooling/token_classify)

## 支持的功能

Token 分类功能应与 (序列) 分类一致。更多信息请参见[此页面](classify.md#supported-features)。
