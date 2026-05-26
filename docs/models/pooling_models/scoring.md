# 评分用途

评分模型旨在计算两个输入提示之间的相似度分数。它支持三种模型类型（又称 `score_type`）：`cross-encoder`、`late-interaction` 和 `bi-encoder`。

!!! note
    vLLM 只处理 RAG 流水线的模型推理部分（例如嵌入生成和重排序）。对于更高级的 RAG 编排，您应该利用 [LangChain](https://github.com/langchain-ai/langchain) 等集成框架。

## 摘要

- 模型用途：评分
- 池化任务：

| 评分类型 | 池化任务 | 评分函数 |
|--------------------|-----------------------|--------------------------|
| `cross-encoder` | `classify` (见附注) | 线性分类器 |
| `late-interaction` | `token_embed` | 后期交互 (MaxSim) |
| `bi-encoder` | `embed` | 余弦相似度 |

- 离线 API：
    - `LLM.score`
- 在线 API：
    - [评分 API](scoring.md#score-api) (`/score`)
    - [Cohere Rerank API](scoring.md#rerank-api) (`/rerank`, `/v1/rerank`, `/v2/rerank`)

!!! note
    只有当一个分类模型输出的 num_labels 等于 1 时，它才能用作评分模型并启用其评分 API。

### 评分类型

三种支持的评分函数如下图所示。

![评分类型](../../assets/models/pooling_models/score_types.svg)

## 支持的模型

### 交叉编码器模型

[交叉编码器](https://www.sbert.net/examples/applications/cross-encoder/README.html)（又称重排序器）模型是分类模型的一个子集，接受两个提示作为输入，并且输出 num_labels 等于 1。

--8<-- [start:supported-cross-encoder-models]

#### 纯文本模型

| 架构 | 模型 | 示例 HF 模型 | 评分模板 (见附注) | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | ------------------------- | --------------------------- | --------------------------------------- |
| `BertForSequenceClassification` | 基于 BERT | `cross-encoder/ms-marco-MiniLM-L-6-v2` 等 | N/A | | |
| `GemmaForSequenceClassification` | 基于 Gemma | `BAAI/bge-reranker-v2-gemma`(见附注) 等 | [bge-reranker-v2-gemma.jinja](../../../examples/pooling/score/template/bge-reranker-v2-gemma.jinja) | ✅︎ | ✅︎ |
| `GteNewForSequenceClassification` | mGTE-TRM (见附注) | `Alibaba-NLP/gte-multilingual-reranker-base` 等 | N/A | | |
| `LlamaBidirectionalForSequenceClassification`<sup>C</sup> | 基于 Llama 的双向注意力 | `nvidia/llama-nemotron-rerank-1b-v2` 等 | [nemotron-rerank.jinja](../../../examples/pooling/score/template/nemotron-rerank.jinja) | ✅︎ | ✅︎ |
| `ModernBertForSequenceClassification` | 基于 ModernBERT | `Alibaba-NLP/gte-reranker-modernbert-base` 等 | N/A | | |
| `Qwen2ForSequenceClassification`<sup>C</sup> | 基于 Qwen2 | `mixedbread-ai/mxbai-rerank-base-v2`(见附注) 等 | [mxbai_rerank_v2.jinja](../../../examples/pooling/score/template/mxbai_rerank_v2.jinja) | ✅︎ | ✅︎ |
| `Qwen3ForSequenceClassification`<sup>C</sup> | 基于 Qwen3 | `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`, `Qwen/Qwen3-Reranker-0.6B`(见附注) 等 | [qwen3_reranker.jinja](../../../examples/pooling/score/template/qwen3_reranker.jinja) | ✅︎ | ✅︎ |
| `RobertaForSequenceClassification` | 基于 RoBERTa | `cross-encoder/quora-roberta-base` 等 | N/A | | |
| `XLMRobertaForSequenceClassification` | 基于 XLM-RoBERTa | `BAAI/bge-reranker-v2-m3` 等 | N/A | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup> 等 | 生成式模型 | N/A | N/A | \* | \* |

<sup>C</sup> 通过 `--convert classify` 自动转换为分类模型。([详细信息](./README.md#model-conversion))  
\* 功能支持与原模型相同。

!!! note
    某些模型需要特定的提示格式才能正常工作。

    您可以在 [examples/pooling/score/template/](../../../examples/pooling/score/template) 中找到示例 HF 模型对应的评分模板。

    示例：[examples/pooling/score/using_template_offline.py](../../../examples/pooling/score/using_template_offline.py) [examples/pooling/score/using_template_online.py](../../../examples/pooling/score/using_template_online.py)

!!! note
    使用以下命令加载官方的原始 `BAAI/bge-reranker-v2-gemma`。

    ```bash
    vllm serve BAAI/bge-reranker-v2-gemma --hf_overrides '{"architectures": ["GemmaForSequenceClassification"],"classifier_from_token": ["Yes"],"method": "no_post_processing"}'
    ```

!!! note
    第二代 GTE 模型 (mGTE-TRM) 名为 `NewForSequenceClassification`。名称 `NewForSequenceClassification` 过于通用，您应该设置 `--hf-overrides '{"architectures": ["GteNewForSequenceClassification"]}'` 来指定使用 `GteNewForSequenceClassification` 架构。

!!! note
    使用以下命令加载官方的原始 `mxbai-rerank-v2`。

    ```bash
    vllm serve mixedbread-ai/mxbai-rerank-base-v2 --hf_overrides '{"architectures": ["Qwen2ForSequenceClassification"],"classifier_from_token": ["0", "1"], "method": "from_2_way_softmax"}'
    ```

!!! note
    使用以下命令加载官方的原始 `Qwen3 Reranker`。更多信息请参见：[examples/pooling/score/qwen3_reranker_offline.py](../../../examples/pooling/score/qwen3_reranker_offline.py) [examples/pooling/score/qwen3_reranker_online.py](../../../examples/pooling/score/qwen3_reranker_online.py)。

    ```bash
    vllm serve Qwen/Qwen3-Reranker-0.6B --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    ```

#### 多模态模型

!!! note
    有关多模态模型输入的更多信息，请参见[此页面](../supported_models.md#list-of-multimodal-language-models)。

| 架构 | 模型 | 输入 | 示例 HF 模型 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `JinaVLForSequenceClassification` | 基于 JinaVL | T + I<sup>E+</sup> | `jinaai/jina-reranker-m0` 等 | ✅︎ | ✅︎ |
| `LlamaNemotronVLForSequenceClassification` | Llama Nemotron Reranker + SigLIP | T + I<sup>E+</sup> | `nvidia/llama-nemotron-rerank-vl-1b-v2` | | |
| `Qwen3VLForSequenceClassification` | Qwen3-VL-Reranker | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-Reranker-2B`(见附注) 等 | ✅︎ | ✅︎ |

<sup>C</sup> 通过 `--convert classify` 自动转换为分类模型。([详细信息](README.md#model-conversion))  
\* 功能支持与原模型相同。

!!! note
    与 Qwen3-Reranker 类似，您需要使用以下 `--hf_overrides` 来加载官方的原始 `Qwen3-VL-Reranker`。`Qwen3-VL` 官方使用 `qwen_vl_utils` 进行图像预处理，而 vLLM 使用 Transformers 的 `video_processing_qwen3_vl`，这会导致与官方 Hugging Face 仓库示例略有不同的结果。

    ```bash
    vllm serve Qwen/Qwen3-VL-Reranker-2B --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    ```

--8<-- [end:supported-cross-encoder-models]

### 后期交互模型

所有支持 token 嵌入任务的模型也支持使用评分 API，通过计算两个输入提示的后期交互来计算相似度分数。有关 token 嵌入模型的更多信息，请参见[此页面](token_embed.md)。

--8<-- "docs/models/pooling_models/token_embed.md:supported-token-embed-models"

### 双编码器

所有支持嵌入任务的模型也支持使用评分 API，通过计算两个输入提示嵌入的余弦相似度来计算相似度分数。有关嵌入模型的更多信息，请参见[此页面](embed.md)。

--8<-- "docs/models/pooling_models/embed.md:supported-embed-models"

## 离线推理

### 池化参数

以下[池化参数][vllm.PoolingParams]仅由交叉编码器模型支持，不适用于后期交互和双编码器模型。

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:classify-pooling-params"
```

### `LLM.score`

[score][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.score] 方法输出句子对之间的相似度分数。

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-reranker-v2-m3", runner="pooling")
(output,) = llm.score(
    "What is the capital of France?",
    "The capital of Brazil is Brasilia.",
)

score = output.outputs.score
print(f"Score: {score}")
```

代码示例请参见：[examples/basic/offline_inference/score.py](../../../examples/basic/offline_inference/score.py)

## 在线服务

### 评分 API

我们的评分 API (`/score`) 类似于 `LLM.score`，计算两个输入提示之间的相似度分数。

#### 参数

支持以下评分 API 参数：

```python
--8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
--8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
--8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
--8<-- "vllm/entrypoints/pooling/scoring/protocol.py:scoring-common-params"
--8<-- "vllm/entrypoints/pooling/scoring/protocol.py:score-request-params"
```

#### 示例

##### 单次推理

您可以向 `queries` 和 `documents` 都传递字符串，形成单个句子对。

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "BAAI/bge-reranker-v2-m3",
  "encoding_format": "float",
  "queries": "What is the capital of France?",
  "documents": "The capital of France is Paris."
}'
```

??? console "响应"

    ```json
    {
      "id": "score-request-id",
      "object": "list",
      "created": 693447,
      "model": "BAAI/bge-reranker-v2-m3",
      "data": [
        {
          "index": 0,
          "object": "score",
          "score": 1
        }
      ],
      "usage": {}
    }
    ```

##### 批量推理

您可以向 `queries` 传递字符串，向 `documents` 传递列表，形成多个句子对，其中每个对由 `queries` 和 `documents` 中的一个字符串组成。总对数为 `len(documents)`。

??? console "请求"

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/score' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model": "BAAI/bge-reranker-v2-m3",
      "queries": "What is the capital of France?",
      "documents": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris."
      ]
    }'
    ```

??? console "响应"

    ```json
    {
      "id": "score-request-id",
      "object": "list",
      "created": 693570,
      "model": "BAAI/bge-reranker-v2-m3",
      "data": [
        {
          "index": 0,
          "object": "score",
          "score": 0.001094818115234375
        },
        {
          "index": 1,
          "object": "score",
          "score": 1
        }
      ],
      "usage": {}
    }
    ```

您可以向 `queries` 和 `documents` 都传递列表，形成多个句子对，其中每个对由 `queries` 中的一个字符串和 `documents` 中对应的字符串组成（类似于 `zip()`）。总对数为 `len(documents)`。

??? console "请求"

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/score' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model": "BAAI/bge-reranker-v2-m3",
      "encoding_format": "float",
      "queries": [
        "What is the capital of Brazil?",
        "What is the capital of France?"
      ],
      "documents": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris."
      ]
    }'
    ```

??? console "响应"

    ```json
    {
      "id": "score-request-id",
      "object": "list",
      "created": 693447,
      "model": "BAAI/bge-reranker-v2-m3",
      "data": [
        {
          "index": 0,
          "object": "score",
          "score": 1
        },
        {
          "index": 1,
          "object": "score",
          "score": 1
        }
      ],
      "usage": {}
    }
    ```

##### 多模态输入

您可以通过在请求中传递包含多模态输入（图像等）列表的 `content` 来向评分模型传递多模态输入。请参考下面的示例进行说明。

=== "JinaVL-Reranker"

    提供模型服务：

    ```bash
    vllm serve jinaai/jina-reranker-m0
    ```

    由于请求模式不由 OpenAI 客户端定义，我们使用较低级的 `requests` 库向服务器发送请求：

    ??? code

        ```python
        import requests
        
        response = requests.post(
            "http://localhost:8000/v1/score",
            json={
                "model": "jinaai/jina-reranker-m0",
                "queries": "slm markdown",
                "documents": [
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
                                },
                            }
                        ],
                    },
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png"
                                },
                            }
                        ]
                    },
                ],
            },
        )
        response.raise_for_status()
        response_json = response.json()
        print("Scoring output:", response_json["data"][0]["score"])
        print("Scoring output:", response_json["data"][1]["score"])
        ```
完整示例：

- [examples/pooling/score/vision_score_api_online.py](../../../examples/pooling/score/vision_score_api_online.py)
- [examples/pooling/score/vision_rerank_api_online.py](../../../examples/pooling/score/vision_rerank_api_online.py)

### Cohere Rerank API

`/rerank`、`/v1/rerank` 和 `/v2/rerank` API 与 [Jina AI 的 rerank API 接口](https://jina.ai/reranker/)和 [Cohere 的 rerank API 接口](https://docs.cohere.com/v2/reference/rerank)兼容，以确保与流行的开源工具兼容。

代码示例：[examples/pooling/score/rerank_api_online.py](../../../examples/pooling/score/rerank_api_online.py)

#### 参数

支持以下 rerank API 参数：

```python
--8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
--8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
--8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
--8<-- "vllm/entrypoints/pooling/scoring/protocol.py:scoring-common-params"
--8<-- "vllm/entrypoints/pooling/scoring/protocol.py:rerank-request-params"
```

#### 示例

请注意，`top_n` 请求参数是可选的，默认值为 `documents` 字段的长度。结果文档将按相关性排序，`index` 属性可用于确定原始顺序。

??? console "请求"

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/v1/rerank' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model": "BAAI/bge-reranker-base",
      "query": "What is the capital of France?",
      "documents": [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
        "Horses and cows are both animals"
      ]
    }'
    ```

??? console "响应"

    ```json
    {
      "id": "rerank-fae51b2b664d4ed38f5969b612edff77",
      "model": "BAAI/bge-reranker-base",
      "usage": {
        "total_tokens": 56
      },
      "results": [
        {
          "index": 1,
          "document": {
            "text": "The capital of France is Paris."
          },
          "relevance_score": 0.99853515625
        },
        {
          "index": 0,
          "document": {
            "text": "The capital of Brazil is Brasilia."
          },
          "relevance_score": 0.0005860328674316406
        }
      ]
    }
    ```

## 更多示例

更多示例请参见：[examples/pooling/score](../../../examples/pooling/score)

## 支持的功能

由于交叉编码器模型是分类模型的一个子集，接受两个提示作为输入并输出 num_labels 等于 1，交叉编码器的功能应与 (序列) 分类一致。更多信息请参见[此页面](classify.md#supported-features)。

### 评分模板

评分模板仅对 **交叉编码器** 模型支持。如果您使用 **嵌入** 模型进行评分，vLLM 不会应用评分模板。

某些评分模型需要特定的提示格式才能正常工作。您可以使用 `--chat-template` 参数指定自定义评分模板（请参见[聊天模板](../../serving/online_serving/README.md#chat-template)）。

与聊天模板类似，评分模板接收一个 `messages` 列表。对于评分，每条消息都有一个 `role` 属性——`"query"` 或 `"document"`。对于通常的点式交叉编码器，您可以期望恰好两条消息：一个查询和一个文档。要访问查询和文档内容，请使用 Jinja 的 `selectattr` 过滤器：

- **查询**：`{{ (messages | selectattr("role", "eq", "query") | first).content }}`
- **文档**：`{{ (messages | selectattr("role", "eq", "document") | first).content }}`

这种方法比基于索引的访问（`messages[0]`、`messages[1]`）更健壮，因为它根据消息的语义角色进行选择。如果将来向 `messages` 添加了其他消息类型，它也能避免对消息排序的假设。

模板文件示例：[examples/pooling/score/template/nemotron-rerank.jinja](../../../examples/pooling/score/template/nemotron-rerank.jinja)

### 启用/禁用激活

您可以通过 `use_activation` 启用或禁用激活，这仅适用于交叉编码器模型。
