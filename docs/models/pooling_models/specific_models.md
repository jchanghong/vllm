# 特定模型示例

## ColBERT 后期交互模型

[ColBERT](https://arxiv.org/abs/2004.12832)（基于 BERT 的上下文后期交互）是一种检索模型，使用逐 token 嵌入和 MaxSim 评分进行文档排序。与单向量嵌入模型不同，ColBERT 保留 token 级别的表示，并通过后期交互计算相关性分数，在比交叉编码器更高效的同时提供了更好的准确性。

vLLM 支持使用多种编码器骨干网络的 ColBERT 模型：

| 架构 | 骨干网络 | 示例 HF 模型 |
| - | - | - |
| `HF_ColBERT` | BERT | `answerdotai/answerai-colbert-small-v1`, `colbert-ir/colbertv2.0` |
| `ColBERTModernBertModel` | ModernBERT | `lightonai/GTE-ModernColBERT-v1` |
| `ColBERTJinaRobertaModel` | Jina XLM-RoBERTa | `jinaai/jina-colbert-v2` |
| `ColBERTLfm2Model` | LFM2 | `LiquidAI/LFM2-ColBERT-350M` |

**基于 BERT 的 ColBERT** 模型开箱即用：

```shell
vllm serve answerdotai/answerai-colbert-small-v1
```

对于**非 BERT 骨干网络**，使用 `--hf-overrides` 设置正确的架构：

```shell
# ModernBERT 骨干网络
vllm serve lightonai/GTE-ModernColBERT-v1 \
    --hf-overrides '{"architectures": ["ColBERTModernBertModel"]}'

# Jina XLM-RoBERTa 骨干网络
vllm serve jinaai/jina-colbert-v2 \
    --hf-overrides '{"architectures": ["ColBERTJinaRobertaModel"]}' \
    --trust-remote-code

# LFM2 骨干网络
vllm serve LiquidAI/LFM2-ColBERT-350M \
    --hf-overrides '{"architectures": ["ColBERTLfm2Model"]}'
```

然后您可以使用 rerank API：

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "answerdotai/answerai-colbert-small-v1",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "Deep learning uses neural networks."
    ]
}'
```

或 score API：

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "answerdotai/answerai-colbert-small-v1",
    "text_1": "What is machine learning?",
    "text_2": ["Machine learning is a subset of AI.", "The weather is sunny."]
}'
```

您还可以使用 `token_embed` 任务的 Pooling API 获取原始 token 嵌入：

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
    "model": "answerdotai/answerai-colbert-small-v1",
    "input": "What is machine learning?",
    "task": "token_embed"
}'
```

示例请参见：[examples/pooling/score/colbert_rerank_online.py](../../../examples/pooling/score/colbert_rerank_online.py)

## ColQwen3 多模态后期交互模型

ColQwen3 基于 [ColPali](https://arxiv.org/abs/2407.01449)，将 ColBERT 的后期交互方法扩展到**多模态**输入。ColBERT 仅处理纯文本 token 嵌入，而 ColPali/ColQwen3 可以将**文本和图像**（例如 PDF 页面、截图、图表）嵌入为逐 token L2 归一化向量，并通过 MaxSim 评分计算相关性。ColQwen3 具体使用 Qwen3-VL 作为其视觉语言骨干网络。

| 架构 | 骨干网络 | 示例 HF 模型 |
| - | - | - |
| `ColQwen3` | Qwen3-VL | `TomoroAI/tomoro-colqwen3-embed-4b`, `TomoroAI/tomoro-colqwen3-embed-8b` |
| `OpsColQwen3Model` | Qwen3-VL | `OpenSearch-AI/Ops-Colqwen3-4B`, `OpenSearch-AI/Ops-Colqwen3-8B` |
| `Qwen3VLNemotronEmbedModel` | Qwen3-VL | `nvidia/nemotron-colembed-vl-4b-v2`, `nvidia/nemotron-colembed-vl-8b-v2` |

启动服务器：

```shell
vllm serve TomoroAI/tomoro-colqwen3-embed-4b --max-model-len 4096
```

### 纯文本评分和重排序

使用 `/rerank` API：

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "Deep learning uses neural networks."
    ]
}'
```

或 `/score` API：

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "text_1": "What is the capital of France?",
    "text_2": ["The capital of France is Paris.", "Python is a programming language."]
}'
```

### 多模态评分和重排序（文本查询 × 图像文档）

`/score` 和 `/rerank` API 也直接接受多模态输入。
使用 `data_1`/`data_2`（用于 `/score`）或 `documents`（用于 `/rerank`）字段传递图像文档，
其中包含 `content` 列表，包含 `image_url` 和 `text` 部分——与 OpenAI 聊天补全 API 使用的格式相同：

对文本查询与图像文档进行评分：

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "data_1": "Retrieve the city of Beijing",
    "data_2": [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        }
    ]
}'
```

通过文本查询对图像文档进行重排序：

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "query": "Retrieve the city of Beijing",
    "documents": [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_1>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        },
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_2>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        }
    ],
    "top_n": 2
}'
```

### 原始 token 嵌入

您还可以使用 `/pooling` API 与 `token_embed` 任务获取原始 token 嵌入：

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "input": "What is machine learning?",
    "task": "token_embed"
}'
```

对于通过 Pooling API 的**图像输入**，请使用聊天风格的 `messages` 字段：

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        }
    ]
}'
```

### 示例

- 多向量检索：[examples/pooling/token_embed/colqwen3_token_embed_online.py](../../../examples/pooling/token_embed/colqwen3_token_embed_online.py)
- 重排序（文本 + 多模态）：[examples/pooling/score/colqwen3_rerank_online.py](../../../examples/pooling/score/colqwen3_rerank_online.py)

## ColQwen3.5 多模态后期交互模型

ColQwen3.5 基于 [ColPali](https://arxiv.org/abs/2407.01449)，将 ColBERT 的后期交互方法扩展到**多模态**输入。它使用 Qwen3.5 混合骨干网络（线性 + 完全注意力），并生成逐 token L2 归一化向量用于 MaxSim 评分。

| 架构 | 骨干网络 | 示例 HF 模型 |
| - | - | - |
| `ColQwen3_5` | Qwen3.5 | `athrael-soju/colqwen3.5-4.5B` |

启动服务器：

```shell
vllm serve athrael-soju/colqwen3.5-4.5B --max-model-len 4096
```

然后您可以使用 rerank 端点：

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "athrael-soju/colqwen3.5-4.5B",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "Deep learning uses neural networks."
    ]
}'
```

或 score 端点：

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "athrael-soju/colqwen3.5-4.5B",
    "text_1": "What is the capital of France?",
    "text_2": ["The capital of France is Paris.", "Python is a programming language."]
}'
```

示例请参见：[examples/pooling/score/colqwen3_5_rerank_online.py](../../../examples/pooling/score/colqwen3_5_rerank_online.py)

## Llama Nemotron 多模态

### 嵌入模型

Llama Nemotron VL 嵌入模型将双向 Llama 嵌入骨干网络（来自 `nvidia/llama-nemotron-embed-1b-v2`）与 SigLIP 视觉编码器相结合，从文本和/或图像生成单向量嵌入。

| 架构 | 骨干网络 | 示例 HF 模型 |
| - | - | - |
| `LlamaNemotronVLModel` | 双向 Llama + SigLIP | `nvidia/llama-nemotron-embed-vl-1b-v2` |

启动服务器：

```shell
vllm serve nvidia/llama-nemotron-embed-vl-1b-v2 \
    --trust-remote-code \
    --chat-template examples/pooling/embed/template/nemotron_embed_vl.jinja
```

!!! note
    此模型 tokenizer 附带的聊天模板不适用于嵌入 API。在基于 `messages`（聊天风格）的嵌入 API 提供服务时，请使用上面提供的覆盖模板。

    覆盖模板使用消息 `role` 来自动添加适当的前缀：对于查询，将 `role` 设置为 `"query"`（添加 `query: ` 前缀）；对于段落，将 `role` 设置为 `"document"`（添加 `passage: ` 前缀）。任何其他角色将省略前缀。

嵌入文本查询：

```shell
curl -s http://localhost:8000/v1/embeddings -H "Content-Type: application/json" -d '{
    "model": "nvidia/llama-nemotron-embed-vl-1b-v2",
    "messages": [
        {
            "role": "query",
            "content": [
                {"type": "text", "text": "What is machine learning?"}
            ]
        }
    ]
}'
```

通过聊天风格 `messages` 字段嵌入图像：

```shell
curl -s http://localhost:8000/v1/embeddings -H "Content-Type: application/json" -d '{
    "model": "nvidia/llama-nemotron-embed-vl-1b-v2",
    "messages": [
        {
            "role": "document",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        }
    ]
}'
```

### 重排序模型

Llama Nemotron VL 重排序模型将相同的双向 Llama + SigLIP 骨干网络与序列分类头相结合，用于交叉编码器评分和重排序。

| 架构 | 骨干网络 | 示例 HF 模型 |
| - | - | - |
| `LlamaNemotronVLForSequenceClassification` | 双向 Llama + SigLIP | `nvidia/llama-nemotron-rerank-vl-1b-v2` |

启动服务器：

```shell
vllm serve nvidia/llama-nemotron-rerank-vl-1b-v2 \
    --runner pooling \
    --trust-remote-code \
    --chat-template examples/pooling/score/template/nemotron-vl-rerank.jinja
```

!!! note
    此检查点 tokenizer 附带的聊天模板不适用于评分/重排序 API。在提供服务时，请使用提供的覆盖模板：`examples/pooling/score/template/nemotron-vl-rerank.jinja`。

对文本查询与图像文档进行评分：

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
    "data_1": "Find diagrams about autonomous robots",
    "data_2": [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
                {"type": "text", "text": "Robotics workflow diagram."}
            ]
        }
    ]
}'
```

通过文本查询对图像文档进行重排序：

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
    "query": "Find diagrams about autonomous robots",
    "documents": [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_1>"}},
                {"type": "text", "text": "Robotics workflow diagram."}
            ]
        },
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_2>"}},
                {"type": "text", "text": "General skyline photo."}
            ]
        }
    ],
    "top_n": 2
}'
```

## BAAI/bge-m3

`BAAI/bge-m3` 模型附带用于稀疏和 colbert 嵌入的额外权重，但不幸的是在其 `config.json` 中架构声明为 `XLMRobertaModel`，这使得 `vLLM` 将其作为普通的 ROBERTA 模型加载，而不会加载这些额外权重。要加载完整的模型权重，请像这样覆盖其架构：

```shell
vllm serve BAAI/bge-m3 --hf-overrides '{"architectures": ["BgeM3EmbeddingModel"]}'
```

然后您可以像这样获取稀疏嵌入：

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
     "model": "BAAI/bge-m3",
     "task": "token_classify",
     "input": ["What is BGE M3?", "Definition of BM25"]
}'
```

由于输出模式的限制，输出由每个输入的每个 token 的 token 分数列表组成。这意味着您还需要调用 `/tokenize` 才能将 token 与分数配对。请参考 `tests/models/language/pooling/test_bge_m3.py` 中的测试来了解如何操作。

您可以像这样获取 colbert 嵌入：

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
     "model": "BAAI/bge-m3",
     "task": "token_embed",
     "input": ["What is BGE M3?", "Definition of BM25"]
}'
```
