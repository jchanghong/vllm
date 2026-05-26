# Token 嵌入用途

## 摘要

- 模型用途：Token 分类模型
- 池化任务：`token_embed`
- 离线 API：
    - `LLM.encode(..., pooling_task="token_embed")`
- 在线 API：
    - Pooling API (`/pooling`)

(序列) 嵌入任务和 token 嵌入任务之间的区别在于，(序列) 嵌入为每个序列输出一个嵌入，而 token 嵌入为每个 token 输出一个嵌入。

许多嵌入模型同时支持 (序列) 嵌入和 token 嵌入。有关 (序列) 嵌入的更多详细信息，请参阅[此页面](embed.md)。

!!! note

    池化多任务支持自 v0.21 起已移除。当默认的池化任务 (embed) 不是您想要的时，您需要手动指定，可以通过离线的 `PoolerConfig(task="token_embed")` 或在线的 `--pooler-config.task token_embed`。

## 典型用例

### 多向量检索

实现示例请参见：

离线：[examples/pooling/token_embed/multi_vector_retrieval_offline.py](../../../examples/pooling/token_embed/multi_vector_retrieval_offline.py)

在线：[examples/pooling/token_embed/multi_vector_retrieval_online.py](../../../examples/pooling/token_embed/multi_vector_retrieval_online.py)

### 后期交互

可以通过评分 API 使用两个输入提示之间的后期交互来计算相似度分数。更多信息请参见[评分 API](scoring.md)。

### 提取最后隐藏状态

任何架构的模型都可以通过 `--convert embed` 转换为嵌入模型。然后可以使用 token 嵌入来提取这些模型的最后隐藏状态。

## 支持的模型

--8<-- [start:supported-token-embed-models]

### 纯文本模型

| 架构 | 模型 | 示例 HF 模型 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `ColBERTLfm2Model` | LFM2 | `LiquidAI/LFM2-ColBERT-350M` | | |
| `ColBERTModernBertModel` | ModernBERT | `lightonai/GTE-ModernColBERT-v1` | | |
| `ColBERTJinaRobertaModel` | Jina XLM-RoBERTa | `jinaai/jina-colbert-v2` | | |
| `HF_ColBERT` | BERT | `answerdotai/answerai-colbert-small-v1`, `colbert-ir/colbertv2.0` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup> 等 | 生成式模型 | N/A | \* | \* |

### 多模态模型

!!! note
    有关多模态模型输入的更多信息，请参见[此页面](../supported_models.md#list-of-multimodal-language-models)。

| 架构 | 模型 | 输入 | 示例 HF 模型 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----- | ----------------- | ------------------------------ | ------------------------------------------ |
| `ColModernVBertForRetrieval` | ColModernVBERT | T / I | `ModernVBERT/colmodernvbert-merged` | | |
| `ColPaliForRetrieval` | ColPali | T / I | `vidore/colpali-v1.3-hf` | | |
| `ColQwen3` | Qwen3-VL | T / I | `TomoroAI/tomoro-colqwen3-embed-4b`, `TomoroAI/tomoro-colqwen3-embed-8b` | | |
| `ColQwen3_5` | ColQwen3.5 | T + I + V | `athrael-soju/colqwen3.5-4.5B-v3` | | |
| `OpsColQwen3Model` | Qwen3-VL | T / I | `OpenSearch-AI/Ops-Colqwen3-4B`, `OpenSearch-AI/Ops-Colqwen3-8B` | | |
| `Qwen3VLNemotronEmbedModel` | Qwen3-VL | T / I | `nvidia/nemotron-colembed-vl-4b-v2`, `nvidia/nemotron-colembed-vl-8b-v2` | ✅︎ | ✅︎ |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup> 等 | 生成式模型 | \* | N/A | \* | \* |

<sup>C</sup> 通过 `--convert embed` 自动转换为嵌入模型。([详细信息](./README.md#model-conversion))  
\* 功能支持与原模型相同。

如果您的模型不在上述列表中，我们将尝试使用 [as_embedding_model][vllm.model_executor.models.adapters.as_embedding_model] 自动转换模型。

### 特殊模型

| 架构 | 模型 | 示例 HF 模型 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `JinaForRanking` | 基于 Qwen3 | `jinaai/jina-reranker-v3` | | |

jina-reranker-v3 是一个列表式文档重排序模型，具有新颖的 `last but not late interaction` 架构。更多信息请参见：[examples/pooling/token_embed/jina_reranker_v3_offline.py](../../../examples/pooling/token_embed/jina_reranker_v3_offline.py)

--8<-- [end:supported-token-embed-models]

## 离线推理

### 池化参数

支持以下[池化参数][vllm.PoolingParams]。

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:embed-pooling-params"
```

### `LLM.encode`

[encode][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.encode] 方法适用于 vLLM 中的所有池化模型。

为 token 嵌入模型使用 `LLM.encode` 时设置 `pooling_task="token_embed"`：

```python
from vllm import LLM

llm = LLM(model="answerdotai/answerai-colbert-small-v1", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="token_embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

### `LLM.score`

[score][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.score] 方法输出句子对之间的相似度分数。

所有支持 token 嵌入任务的模型也支持使用评分 API，通过计算两个输入提示的后期交互来计算相似度分数。

```python
from vllm import LLM

llm = LLM(model="answerdotai/answerai-colbert-small-v1", runner="pooling")
(output,) = llm.score(
    "What is the capital of France?",
    "The capital of Brazil is Brasilia.",
)

score = output.outputs.score
print(f"Score: {score}")
```

## 在线服务

请参考 [Pooling API](README.md#pooling-api) 并使用 `"task":"token_embed"`。

## 更多示例

更多示例请参见：[examples/pooling/token_embed](../../../examples/pooling/token_embed)

## 支持的功能

Token 嵌入功能应与 (序列) 嵌入一致。更多信息请参见[此页面](embed.md#supported-features)。
