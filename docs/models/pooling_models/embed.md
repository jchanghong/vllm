# 嵌入用途

嵌入模型是一类机器学习模型，旨在将非结构化数据（如文本、图像或音频）转换为称为嵌入的结构化数值表示。

## 摘要

- 模型用途：(序列) 嵌入
- 池化任务：`embed`
- 离线 API：
    - `LLM.embed(...)`
    - `LLM.encode(..., pooling_task="embed")`
    - `LLM.score(...)`
- 在线 API：
    - [Cohere Embed API](embed.md#cohere-embed-api) (`/v2/embed`)
    - [兼容 OpenAI 的 Embeddings API](embed.md#openai-compatible-embeddings-api) (`/v1/embeddings`)
    - Pooling API (`/pooling`)

(序列) 嵌入和 token 嵌入之间的主要区别在于它们的输出粒度：(序列) 嵌入为整个输入序列生成单个嵌入向量，而 token 嵌入为序列中的每个单独 token 生成一个嵌入。

许多嵌入模型同时支持 (序列) 嵌入和 token 嵌入。有关 token 嵌入的更多详细信息，请参阅[此页面](token_embed.md)。

## 典型用例

### 嵌入

嵌入模型最基本的用例是嵌入输入，例如用于 RAG。

### 成对相似度

您可以使用[评分 API](scoring.md) 计算成对相似度分数来构建相似度矩阵。

## 支持的模型

--8<-- [start:supported-embed-models]

### 纯文本模型

| 架构 | 模型 | 示例 HF 模型 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `BertModel` | 基于 BERT | `BAAI/bge-base-en-v1.5`, `Snowflake/snowflake-arctic-embed-xs` 等 | | |
| `BertSpladeSparseEmbeddingModel` | SPLADE | `naver/splade-v3` | | |
| `ErnieModel` | 类 BERT 中文 ERNIE | `shibing624/text2vec-base-chinese-sentence` | | |
| `Gemma2Model`<sup>C</sup> | 基于 Gemma 2 | `BAAI/bge-multilingual-gemma2` 等 | ✅︎ | ✅︎ |
| `Gemma3TextModel`<sup>C</sup> | 基于 Gemma 3 | `google/embeddinggemma-300m` 等 | ✅︎ | ✅︎ |
| `GritLM` | GritLM | `parasail-ai/GritLM-7B-vllm` | ✅︎ | ✅︎ |
| `GteModel` | Arctic-Embed-2.0-M | `Snowflake/snowflake-arctic-embed-m-v2.0` | | |
| `GteNewModel` | mGTE-TRM (见附注) | `Alibaba-NLP/gte-multilingual-base` 等 | | |
| `JinaEmbeddingsV5Model`<sup>C</sup> | 基于 Qwen3 的特定任务 LoRA 适配器 | `jinaai/jina-embeddings-v5-text-small` (见附注) | ✅︎ | ✅︎ |
| `LlamaBidirectionalModel`<sup>C</sup> | 基于 Llama 的双向注意力 | `nvidia/llama-nemotron-embed-1b-v2` 等 | ✅︎ | ✅︎ |
| `LlamaModel`<sup>C</sup>, `LlamaForCausalLM`<sup>C</sup>, `MistralModel`<sup>C</sup> 等 | 基于 Llama | `intfloat/e5-mistral-7b-instruct` 等 | ✅︎ | ✅︎ |
| `ModernBertModel` | 基于 ModernBERT | `Alibaba-NLP/gte-modernbert-base` 等 | | |
| `NomicBertModel` | Nomic BERT | `nomic-ai/nomic-embed-text-v1`, `nomic-ai/nomic-embed-text-v2-moe`, `Snowflake/snowflake-arctic-embed-m-long` 等 | | |
| `Qwen2Model`<sup>C</sup>, `Qwen2ForCausalLM`<sup>C</sup> | 基于 Qwen2 | `ssmits/Qwen2-7B-Instruct-embed-base` (见附注), `Alibaba-NLP/gte-Qwen2-7B-instruct` (见附注) 等 | ✅︎ | ✅︎ |
| `Qwen3Model`<sup>C</sup>, `Qwen3ForCausalLM`<sup>C</sup> | 基于 Qwen3 | `Qwen/Qwen3-Embedding-0.6B` 等 | ✅︎ | ✅︎ |
| `RobertaModel`, `RobertaForMaskedLM` | 基于 RoBERTa | `sentence-transformers/all-roberta-large-v1` 等 | | |
| `VoyageQwen3BidirectionalEmbedModel`<sup>C</sup> | 基于 Voyage Qwen3 的双向注意力 | `voyageai/voyage-4-nano` 等 | ✅︎ | ✅︎ |
| `XLMRobertaModel` | 基于 XLMRobertaModel | `BAAI/bge-m3` (见附注), `intfloat/multilingual-e5-base`, `jinaai/jina-embeddings-v3` (见附注) 等 | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup> 等 | 生成式模型 | N/A | \* | \* |

!!! note
    第二代 GTE 模型 (mGTE-TRM) 名为 `NewModel`。名称 `NewModel` 过于通用，您应该设置 `--hf-overrides '{"architectures": ["GteNewModel"]}'` 来指定使用 `GteNewModel` 架构。

!!! note
    `ssmits/Qwen2-7B-Instruct-embed-base` 的 Sentence Transformers 配置定义不当。
    您需要通过传递 `--pooler-config '{"pooling_type": "MEAN"}'` 来手动设置均值池化。

!!! note
    对于 `Alibaba-NLP/gte-Qwen2-*`，您需要启用 `--trust-remote-code` 才能加载正确的 tokenizer。
    请参见 [HF Transformers 上的相关问题](https://github.com/huggingface/transformers/issues/34882)。

!!! note
    `BAAI/bge-m3` 模型附带用于稀疏和 colbert 嵌入的额外权重，请参见[此页面](specific_models.md#baaibge-m3)了解更多信息。

!!! note
    `jinaai/jina-embeddings-v3` 通过 LoRA 支持多个任务，而 vllm 暂时仅支持通过合并 LoRA 权重来进行文本匹配任务。

!!! note
    `jinaai/jina-embeddings-v5-text-small` 附带四个特定任务的 LoRA 适配器
    (`retrieval`, `text-matching`, `classification`, `clustering`)。vLLM 在加载时
    将所选适配器合并到基础权重中。使用
    `--hf-overrides '{"jina_task": "<task>"}'` 选择任务；默认值为 `retrieval`。

### 多模态模型

!!! note
    有关多模态模型输入的更多信息，请参见[此页面](../supported_models.md#list-of-multimodal-language-models)。

| 架构 | 模型 | 输入 | 示例 HF 模型 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `CLIPModel` | CLIP | T / I | `openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14` 等 | | |
| `LlamaNemotronVLModel` | Llama Nemotron Embedding + SigLIP | T + I | `nvidia/llama-nemotron-embed-vl-1b-v2` | | |
| `LlavaNextForConditionalGeneration`<sup>C</sup> | 基于 LLaVA-NeXT | T / I | `royokong/e5-v` | | ✅︎ |
| `Phi3VForCausalLM`<sup>C</sup> | 基于 Phi-3-Vision | T + I | `TIGER-Lab/VLM2Vec-Full` | | ✅︎ |
| `Qwen3VLForConditionalGeneration`<sup>C</sup> (见附注) | Qwen3-VL | T + I + V | `Qwen/Qwen3-VL-Embedding-2B` 等 | ✅︎ | ✅︎ |
| `SiglipModel` | SigLIP, SigLIP2 | T / I | `google/siglip-base-patch16-224`, `google/siglip2-base-patch16-224` | | |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup> 等 | 生成式模型 | \* | N/A | \* | \* |

<sup>C</sup> 通过 `--convert embed` 自动转换为嵌入模型。([详细信息](./README.md#model-conversion))  
\* 功能支持与原模型相同。

如果您的模型不在上述列表中，我们将尝试使用 [as_embedding_model][vllm.model_executor.models.adapters.as_embedding_model] 自动转换模型。默认情况下，整个提示的嵌入从最后一个 token 对应的归一化隐藏状态中提取。

!!! note
    `Qwen3-VL-Embedding` 官方使用 `qwen_vl_utils` 进行图像预处理，而 vLLM 使用 Transformers 的 `video_processing_qwen3_vl`，这会导致与官方 Hugging Face 仓库示例略有不同的结果。使用 `qwen_vl_utils` 进行离线推理的示例代码可以在 [vision_embedding_offline.py](../../../examples/pooling/embed/vision_embedding_offline.py) 示例中找到。

!!! note
    虽然 vLLM 支持通过 `--convert embed` 自动将任何架构的模型转换为嵌入模型，但为了获得最佳结果，您应该使用专门训练为嵌入模型的池化模型。

--8<-- [end:supported-embed-models]

## 离线推理

### 池化参数

支持以下[池化参数][vllm.PoolingParams]。

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:embed-pooling-params"
```

### `LLM.embed`

[embed][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.embed] 方法为每个提示输出一个嵌入向量。

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.embed("Hello, my name is")

embeds = output.outputs.embedding
print(f"Embeddings: {embeds!r} (size={len(embeds)})")
```

代码示例请参见：[examples/basic/offline_inference/embed.py](../../../examples/basic/offline_inference/embed.py)

### `LLM.encode`

[encode][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.encode] 方法适用于 vLLM 中的所有池化模型。

为嵌入模型使用 `LLM.encode` 时设置 `pooling_task="embed"`：

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

### `LLM.score`

[score][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.score] 方法输出句子对之间的相似度分数。

所有支持嵌入任务的模型也支持使用评分 API，通过计算两个输入提示嵌入的余弦相似度来计算相似度分数。

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.score(
    "What is the capital of France?",
    "The capital of Brazil is Brasilia.",
)

score = output.outputs.score
print(f"Score: {score}")
```

## 在线服务

### 兼容 OpenAI 的 Embeddings API

我们的 Embeddings API 与 [OpenAI 的 Embeddings API](https://platform.openai.com/docs/api-reference/embeddings) 兼容；
您可以使用[官方的 OpenAI Python 客户端](https://github.com/openai/openai-python)与其交互。

代码示例：[examples/pooling/embed/openai_embedding_client.py](../../../examples/pooling/embed/openai_embedding_client.py)

#### Completion 参数

支持以下分类 API 参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:completion-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:encoding-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:embed-params"
    ```

支持以下额外参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:completion-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:encoding-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:embed-extra-params"
    ```

#### Chat 参数

对于类似聊天的输入（即如果传递了 `messages`），则支持以下参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:chat-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:encoding-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:embed-params"
    ```

而是支持以下额外参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:chat-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:encoding-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:embed-extra-params"
    ```

#### 示例

如果模型有[聊天模板](../../serving/online_serving/README.md#chat-template)，您可以将 `inputs` 替换为 `messages` 列表（与 [Chat API](../../serving/online_serving/openai_compatible_server.md#chat-api) 相同的模式），
它将被视为模型的单个提示。以下是调用 API 同时保留 OpenAI 类型注解的便捷函数：

??? code

    ```python
    from openai import OpenAI
    from openai._types import NOT_GIVEN, NotGiven
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.create_embedding_response import CreateEmbeddingResponse

    def create_chat_embeddings(
        client: OpenAI,
        *,
        messages: list[ChatCompletionMessageParam],
        model: str,
        encoding_format: Union[Literal["base64", "float"], NotGiven] = NOT_GIVEN,
    ) -> CreateEmbeddingResponse:
        return client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={"messages": messages, "model": model, "encoding_format": encoding_format},
        )
    ```

##### 多模态输入

您可以通过为服务器定义自定义聊天模板并在请求中传递 `messages` 列表来向嵌入模型传递多模态输入。请参考下面的示例进行说明。

=== "VLM2Vec"

    提供模型服务：

    ```bash
    vllm serve TIGER-Lab/VLM2Vec-Full --runner pooling \
      --trust-remote-code \
      --max-model-len 4096 \
      --chat-template examples/pooling/embed/template/vlm2vec_phi3v.jinja
    ```

    !!! important
        由于 VLM2Vec 与 Phi-3.5-Vision 具有相同的模型架构，我们必须显式传递 `--runner pooling`
        才能以嵌入模式（而不是文本生成模式）运行此模型。

        此模型的自定义聊天模板与原始模板完全不同，
        可以在此处找到：[examples/pooling/embed/template/vlm2vec_phi3v.jinja](../../../examples/pooling/embed/template/vlm2vec_phi3v.jinja)

    由于请求模式不由 OpenAI 客户端定义，我们使用较低级的 `requests` 库向服务器发送请求：

    ??? code

        ```python
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
        )
        image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

        response = create_chat_embeddings(
            client,
            model="TIGER-Lab/VLM2Vec-Full",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "Represent the given image."},
                    ],
                }
            ],
            encoding_format="float",
        )

        print("Image embedding output:", response.data[0].embedding)
        ```

=== "DSE-Qwen2-MRL"

    提供模型服务：

    ```bash
    vllm serve MrLight/dse-qwen2-2b-mrl-v1 --runner pooling \
      --trust-remote-code \
      --max-model-len 8192 \
      --chat-template examples/pooling/embed/template/dse_qwen2_vl.jinja
    ```

    !!! important
        与 VLM2Vec 类似，我们必须显式传递 `--runner pooling`。

        此外，`MrLight/dse-qwen2-2b-mrl-v1` 需要 EOS token 来生成嵌入，这由
        自定义聊天模板处理：[examples/pooling/embed/template/dse_qwen2_vl.jinja](../../../examples/pooling/embed/template/dse_qwen2_vl.jinja)

    !!! important
        `MrLight/dse-qwen2-2b-mrl-v1` 需要为文本查询嵌入提供最小图像尺寸的占位图像。详情请参见下面的完整代码示例。

完整示例：[examples/pooling/embed/vision_embedding_online.py](../../../examples/pooling/embed/vision_embedding_online.py)

### Cohere Embed API

我们的 API 也与 [Cohere 的 Embed v2 API](https://docs.cohere.com/reference/embed) 兼容，它添加了对某些现代嵌入功能的支持，例如截断、输出维度、嵌入类型和输入类型。此端点适用于任何嵌入模型（包括多模态模型）。

#### Cohere Embed API 请求参数

| 参数 | 类型 | 必需 | 描述 |
| --------- | ---- | -------- | ----------- |
| `model` | string | 是 | 模型名称 |
| `input_type` | string | 否 | 提示前缀键（取决于模型，见下文） |
| `texts` | list[string] | 否 | 文本输入（使用 `texts`、`images` 或 `inputs` 之一） |
| `images` | list[string] | 否 | Base64 data URI 图像 |
| `inputs` | list[object] | 否 | 混合文本和图像内容对象 |
| `embedding_types` | list[string] | 否 | 输出类型（默认：`["float"]`） |
| `output_dimension` | int | 否 | 将嵌入截断到此维度（Matryoshka） |
| `truncate` | string | 否 | `END`、`START` 或 `NONE`（默认：`END`） |

#### 文本嵌入

```bash
curl -X POST "http://localhost:8000/v2/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Snowflake/snowflake-arctic-embed-m-v1.5",
    "input_type": "query",
    "texts": ["Hello world", "How are you?"],
    "embedding_types": ["float"]
  }'
```

??? console "响应"

    ```json
    {
      "id": "embd-...",
      "embeddings": {
        "float": [
          [0.012, -0.034, ...],
          [0.056, 0.078, ...]
        ]
      },
      "texts": ["Hello world", "How are you?"],
      "meta": {
        "api_version": {"version": "2"},
        "billed_units": {"input_tokens": 12}
      }
    }
    ```

#### 混合文本和图像输入

对于多模态模型，您可以通过传递 base64 data URI 来嵌入图像。`inputs` 字段接受一个包含混合文本和图像内容的对象列表：

```bash
curl -X POST "http://localhost:8000/v2/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/siglip-so400m-patch14-384",
    "inputs": [
      {
        "content": [
          {"type": "text", "text": "A photo of a cat"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}
        ]
      }
    ],
    "embedding_types": ["float"]
  }'
```

#### 嵌入类型

`embedding_types` 参数控制输出格式。可以在单次调用中请求多种类型：

| 类型 | 描述 |
| ---- | ----------- |
| `float` | 原始 float32 嵌入（默认） |
| `binary` | 位打包有符号二进制 |
| `ubinary` | 位打包无符号二进制 |
| `base64` | 小端 float32 编码为 base64 |

```bash
curl -X POST "http://localhost:8000/v2/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Snowflake/snowflake-arctic-embed-m-v1.5",
    "input_type": "query",
    "texts": ["What is machine learning?"],
    "embedding_types": ["float", "binary"]
  }'
```

??? console "响应"

    ```json
    {
      "id": "embd-...",
      "embeddings": {
        "float": [[0.012, -0.034, ...]],
        "binary": [[42, -117, ...]]
      },
      "texts": ["What is machine learning?"],
      "meta": {
        "api_version": {"version": "2"},
        "billed_units": {"input_tokens": 8}
      }
    }
    ```

#### 截断

`truncate` 参数控制如何处理超过模型最大序列长度的输入：

| 值 | 行为 |
| ----- | --------- |
| `END`（默认） | 保留前面的 token，丢弃末尾 |
| `START` | 保留后面的 token，丢弃开头 |
| `NONE` | 如果输入过长则返回错误 |

#### 输入类型和提示前缀

`input_type` 字段选择一个提示前缀，该前缀将添加到每个文本输入之前。可用的取值取决于模型：

- **在 `config.json` 中有 `task_instructions` 的模型**：`task_instructions` 字典中的键是有效的 `input_type` 值，对应的值会添加到每个文本之前。
- **有 `config_sentence_transformers.json` 提示的模型**：`prompts` 字典中的键是有效的 `input_type` 值。例如，`Snowflake/snowflake-arctic-embed-xs` 定义了 `"query"`，因此设置 `input_type: "query"` 会添加 `"Represent this sentence for searching relevant passages: "` 前缀。
- **其他模型**：不支持 `input_type`，如果传递则会引发验证错误。

## 更多示例

更多示例请参见：[examples/pooling/embed](../../../examples/pooling/embed)

## 支持的功能

### 启用/禁用归一化

您可以通过 `use_activation` 启用或禁用归一化。

### Matryoshka 嵌入

[Matryoshka 嵌入](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#matryoshka-embeddings)或 [Matryoshka Representation Learning (MRL)](https://arxiv.org/abs/2205.13147) 是一种用于训练嵌入模型的技术。它允许用户在性能和成本之间进行权衡。

!!! warning
    并非所有嵌入模型都使用 Matryoshka Representation Learning 进行训练。为避免误用 `dimensions` 参数，vLLM 会返回错误信息，以阻止对不支持 Matryoshka 嵌入的模型更改输出维度的请求。

    例如，在使用 `BAAI/bge-m3` 模型时设置 `dimensions` 参数将导致以下错误。

    ```json
    {"object":"error","message":"Model \"BAAI/bge-m3\" does not support matryoshka representation, changing output dimensions will lead to poor results.","type":"BadRequestError","param":null,"code":400}
    ```

#### 手动启用 Matryoshka 嵌入

目前没有用于指定 Matryoshka 嵌入支持的官方接口。在 vLLM 中，如果 `config.json` 中的 `is_matryoshka` 为 `True`，您可以将输出维度更改为任意值。使用 `matryoshka_dimensions` 来控制允许的输出维度。

对于支持 Matryoshka 嵌入但未被 vLLM 识别的模型，请使用 `hf_overrides={"is_matryoshka": True}` 或 `hf_overrides={"matryoshka_dimensions": [<allowed output dimensions>]}`（离线），或 `--hf-overrides '{"is_matryoshka": true}'` 或 `--hf-overrides '{"matryoshka_dimensions": [<allowed output dimensions>]}'`（在线）手动覆盖配置。

以下是启用 Matryoshka 嵌入的模型服务示例。

```bash
vllm serve Snowflake/snowflake-arctic-embed-m-v1.5 --hf-overrides '{"matryoshka_dimensions":[256]}'
```

#### 离线推理

您可以通过 [PoolingParams][vllm.PoolingParams] 中的 dimensions 参数更改支持 Matryoshka 嵌入的嵌入模型的输出维度。

```python
from vllm import LLM, PoolingParams

llm = LLM(
    model="jinaai/jina-embeddings-v3",
    runner="pooling",
    trust_remote_code=True,
)
outputs = llm.embed(
    ["Follow the white rabbit."],
    pooling_params=PoolingParams(dimensions=32),
)
print(outputs[0].outputs)
```

代码示例请参见：[examples/pooling/embed/embed_matryoshka_fy_offline.py](../../../examples/pooling/embed/embed_matryoshka_fy_offline.py)

#### 在线推理

使用以下命令启动 vLLM 服务器。

```bash
vllm serve jinaai/jina-embeddings-v3 --trust-remote-code
```

您可以使用 dimensions 参数更改支持 Matryoshka 嵌入的嵌入模型的输出维度。

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Follow the white rabbit.",
    "model": "jinaai/jina-embeddings-v3",
    "encoding_format": "float",
    "dimensions": 32
  }'
```

预期输出：

```json
{"id":"embd-5c21fc9a5c9d4384a1b021daccaf9f64","object":"list","created":1745476417,"model":"jinaai/jina-embeddings-v3","data":[{"index":0,"object":"embedding","embedding":[-0.3828125,-0.1357421875,0.03759765625,0.125,0.21875,0.09521484375,-0.003662109375,0.1591796875,-0.130859375,-0.0869140625,-0.1982421875,0.1689453125,-0.220703125,0.1728515625,-0.2275390625,-0.0712890625,-0.162109375,-0.283203125,-0.055419921875,-0.0693359375,0.031982421875,-0.04052734375,-0.2734375,0.1826171875,-0.091796875,0.220703125,0.37890625,-0.0888671875,-0.12890625,-0.021484375,-0.0091552734375,0.23046875]}],"usage":{"prompt_tokens":8,"total_tokens":8,"completion_tokens":0,"prompt_tokens_details":null}}
```

OpenAI 客户端示例请参见：[examples/pooling/embed/openai_embedding_matryoshka_fy_client.py](../../../examples/pooling/embed/openai_embedding_matryoshka_fy_client.py)

## 已移除的功能

### 从 PoolingParams 中移除 `normalize`

我们已从 PoolingParams 中移除 `normalize`，请改用 `use_activation`。
