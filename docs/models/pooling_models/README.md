# 池化模型

!!! note
    我们目前支持池化模型主要是为了方便。这并不保证能比直接使用 Hugging Face Transformers 或 Sentence Transformers 带来任何性能提升。

    我们计划在 vLLM 中优化池化模型。如果您有任何建议，请在 <https://github.com/vllm-project/vllm/issues/21796> 上留言！

## 什么是池化模型？

自然语言处理 (NLP) 主要可以分为以下两类任务：

- 自然语言理解 (NLU)
- 自然语言生成 (NLG)

vLLM 支持的生成式模型涵盖了多种任务类型，例如我们熟悉的大语言模型 (LLM)、处理图像、视频和音频等多模态输入的多模态模型 (VLM)、语音转文本转录模型，以及支持流式输入的实时模型。它们的共同特点是能够生成文本。更进一步，vLLM-Omni 支持多模态内容的生成，包括图像、视频和音频。

随着生成式模型能力的不断提升，这些模型的边界也在不断扩展。然而，某些应用场景仍然需要专门的小语言模型来高效完成特定任务。这些模型通常具有以下特点：

- 它们不需要内容生成。
- 它们只需要执行非常有限的功能，不需要很强的泛化能力、创造力或高智能。
- 它们要求极低的延迟，并且可能在成本受限的硬件上运行。
- 纯文本模型通常少于 10 亿参数，而多模态模型通常少于 100 亿参数。

尽管这些模型规模相对较小，但它们仍然基于 Transformer 架构，与当今最先进的大语言模型相似甚至完全相同。许多最近发布的池化模型也是从大语言模型微调而来，使它们能够受益于大模型的持续改进。这种架构相似性使它们能够重用 vLLM 的大部分基础设施。如果兼容，我们也很乐意帮助它们利用 vLLM 的最新特性。

### 速查表

如下图所示，我们总结了池化模型关键要素之间的关系作为要点。

![速查表](../../assets/models/pooling_models/cheat_sheet.svg)

### 序列级任务和 Token 级任务

序列级任务和 token 级任务之间的关键区别在于它们的输出粒度：序列级任务为整个输入序列生成单个结果，而 token 级任务为序列中的每个单独 token 生成结果。

许多池化模型同时支持（序列）任务和 token 任务。当默认的池化任务（例如序列级任务）不是您想要的时，您需要手动指定（例如 token 级任务），可以通过离线的 `PoolerConfig(task=<task>)` 或在线的 `--pooler-config.task <task>`。

当然，我们也有"插件"任务，允许用户自定义输入和输出处理器。更多信息请参阅 [IO 处理器插件](../../design/io_processor_plugins.md)。

### 池化任务

| 池化任务 | 粒度 | 输出 |
|-----------------------|---------------|-------------------------------------------------|
| `classify` (见附注) | 序列级 | 每个序列的类别概率向量 |
| `embed` | 序列级 | 每个序列的向量表示 |
| `token_classify` | Token 级 | 每个 token 的类别概率向量 |
| `token_embed` | Token 级 | 每个 token 的向量表示 |

!!! note
    在分类任务中，有一个专门的子类别：交叉编码器（又称重排序器）模型。这些模型是分类模型的一个子集，接受两个提示作为输入，并且输出 num_labels 等于 1。

### 池化类型

![池化类型](../../assets/models/pooling_models/pooling_types.svg)

| 池化任务 | 粒度 | 描述 |
|----------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CLS` 池化 | 序列级 | 对于 BERT 类（双向自注意力）模型，默认使用 CLS 池化。这意味着取第一个 token（[CLS] token）对应的 last_hidden_states 作为输出。 |
| `LAST` 池化 | 序列级 | 对于 GPT 类（因果自注意力）模型，默认使用 LAST 池化。这意味着取最后一个 token 对应的 last_hidden_states 作为输出。 |
| `MEAN` 池化 | 序列级 | 许多研究表明，对所有输入 token 的 last_hidden_states 取平均在某些下游任务上表现更好。因此，越来越多的模型开始使用 MEAN 池化。 |
| `ALL` 池化 | Token 级 | 输出所有输入 token 的 last_hidden_states。 |
| `STEP` 池化 | Token 级 | 过滤并输出由 returned_token_ids 返回的 token ID 对应的 last_hidden_states。 |

### 评分类型

![评分类型](../../assets/models/pooling_models/score_types.svg)

评分模型旨在计算两个输入提示之间的相似度分数。它支持三种模型类型（又称 `score_type`）：`cross-encoder`、`late-interaction` 和 `bi-encoder`。

| 池化任务 | 粒度 | 输出 | 评分类型 | 评分函数 |
|-----------------------|---------------|----------------------------------------------|--------------------|--------------------------|
| `classify` (见附注) | 序列级 | 每个序列的重排序分数 | `cross-encoder` | 线性分类器 |
| `embed` | 序列级 | 每个序列的向量表示 | `bi-encoder` | 余弦相似度 |
| `token_classify` | Token 级 | 每个 token 的类别概率向量 | N/A | N/A |
| `token_embed` | Token 级 | 每个 token 的向量表示 | `late-interaction` | 后期交互 (MaxSim) |

!!! note
    只有当一个分类模型输出的 num_labels 等于 1 时，它才能用作评分模型并启用其评分 API。

### 池化用途

| 池化用途 | 描述 |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| 分类用途 | 预测哪个预定义的类别、类或标签最适合给定的输入。 |
| 嵌入用途 | 将非结构化数据（文本、图像、音频等）转换为结构化的数值向量（嵌入）。 |
| Token 分类用途 | Token 级别的分类 |
| Token 嵌入用途 | Token 级别的嵌入 |
| 奖励用途 | 评估语言模型生成输出的质量，作为人类偏好的代理。 |
| 评分用途 | 计算两个输入之间的相似度分数。支持三种模型类型（又称 `score_type`）：`cross-encoder`、`late-interaction` 和 `bi-encoder`。 |
| 插件用途 | 允许用户自定义输入和输出处理器。更多信息请参阅 [IO 处理器插件](../../design/io_processor_plugins.md)。 |

我们还有一些特殊的模型，它们支持多个池化任务，或者具有特定的使用场景，或者支持特殊的输入和输出。

更多详细信息，请参阅下面的链接。

- [分类用途](classify.md)
- [嵌入用途](embed.md)
- [Token 分类用途](token_classify.md)
- [Token 嵌入用途](token_embed.md)
- [奖励用途](reward.md)
- [评分用途](scoring.md)
- [特定模型示例](specific_models.md)

## 离线推理

vLLM 中的每个池化模型根据 [Pooler.get_supported_tasks][vllm.model_executor.layers.pooler.Pooler.get_supported_tasks] 支持一个或多个这些任务，从而启用相应的 API。

### 对应于池化用途的离线 API

| 池化用途 | 专用 API | 用于 `LLM.encode` API 的池化任务 | 评分类型 | 评分函数 |
|-----------------------------|---------------------|-----------------------------------|----------------------------|--------------------------|
| 分类用途 | `LLM.classify(...)` | `classify` | `cross-encoder` (见附注) | 线性分类器 |
| 嵌入用途 | `LLM.embed(...)` | `embed` | `bi-encoder` | 余弦相似度 |
| Token 分类用途 | N/A | `token_classify` | N/A | N/A |
| Token 嵌入用途 | N/A | `token_embed` | `late-interaction` | 后期交互 (MaxSim) |
| 奖励用途 | N/A | `classify` & `token_classify` | N/A | N/A |
| 评分用途 | `LLM.score(...)` | N/A | N/A | N/A |
| 插件用途 | N/A | `plugin` | N/A | N/A |

!!! note
    只有当一个分类模型输出的 num_labels 等于 1 时，它才能用作评分模型并启用其评分 API。

### `LLM.classify`

[classify][vllm.LLM.classify] 方法为每个提示输出一个概率向量。
它主要用于[分类模型](classify.md)。
有关 `LLM.embed` 的更多信息，请参见[此页面](classify.md#offline-inference)。

### `LLM.embed`

[embed][vllm.LLM.embed] 方法为每个提示输出一个嵌入向量。
它主要用于[嵌入模型](embed.md)。
有关 `LLM.embed` 的更多信息，请参见[此页面](embed.md#offline-inference)。

### `LLM.score`

[score][vllm.LLM.score] 方法输出句子对之间的相似度分数。
它主要用于[评分模型](scoring.md)。

### `LLM.encode`

[encode][vllm.LLM.encode] 方法适用于 vLLM 中的所有池化模型。

在使用 `LLM.encode` 时，请使用更具体的方法之一或直接设置任务，请参考[上表](#offline-apis-corresponding-to-pooling-usages)。

### 示例

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

## 在线服务

我们的在线服务器提供了与离线 API 对应的端点：

- 对应 `LLM.embed`：
    - [Cohere Embed API](embed.md#cohere-embed-api) (`/v2/embed`)
    - [兼容 OpenAI 的 Embeddings API](embed.md#openai-compatible-embeddings-api) (`/v1/embeddings`)
- 对应 `LLM.classify`：
    - [分类 API](classify.md#online-serving) (`/classify`)
- 对应 `LLM.score`：
    - [评分 API](scoring.md#score-api) (`/score`)
    - [Cohere Rerank API](scoring.md#rerank-api) (`/rerank`, `/v1/rerank`, `/v2/rerank`)
- Pooling API (`/pooling`) 类似于 `LLM.encode`，适用于所有类型的池化模型。

以下介绍 Pooling API。对于其他 API，请参考上面的链接。

### Pooling API

我们的 Pooling API (`/pooling`) 类似于 `LLM.encode`，适用于所有类型的池化模型。

输入格式与 [Embeddings API](embed.md#openai-compatible-embeddings-api) 相同，但输出数据可以包含任意嵌套列表，而不仅仅是 1 维浮点数列表。

在使用 Pooling API 时，请使用更具体的 API 之一或直接设置任务，请参考[上表](#offline-apis-corresponding-to-pooling-usages)。

代码示例：

- [在线示例](../../../examples/pooling/reward/token_reward_online.py)
- [离线示例](../../../examples/pooling/reward/token_reward_offline.py)

### 示例

```python
# 使用 `vllm serve` 启动支持的嵌入模型服务器，例如：
# vllm serve intfloat/e5-small
import requests

host = "localhost"
port = "8000"
model_name = "intfloat/e5-small"

api_url = f"http://{host}:{port}/pooling"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
prompt = {"model": model_name, "input": prompts, "task": "embed"}

response = requests.post(api_url, json=prompt)

for output in response.json()["data"]:
    data = output["data"]
    print(f"Data: {data!r} (size={len(data)})")
```

## 配置

在 vLLM 中，池化模型实现了 [VllmModelForPooling][vllm.model_executor.models.VllmModelForPooling] 接口。
这些模型在返回结果之前使用 [Pooler][vllm.model_executor.layers.pooler.Pooler] 来提取输入的最终隐藏状态。

### 模型运行器

通过 `--runner pooling` 选项在池化模式下运行模型。

!!! tip
    绝大多数情况下无需设置此选项，因为 vLLM 可以通过 `--runner auto` 自动检测适当的模型运行器。

### 模型转换

vLLM 可以通过 `--convert <type>` 选项使模型适应各种池化任务。

如果已设置 `--runner pooling`（手动或自动），但模型没有实现 [VllmModelForPooling][vllm.model_executor.models.VllmModelForPooling] 接口，
vLLM 将尝试根据下表中显示的架构名称自动转换模型。

| 架构 | `--convert` | 支持的池化任务 |
|-------------------------------------------------|-------------|------------------------------|
| `*ForTextEncoding`, `*EmbeddingModel`, `*Model` | `embed` | `token_embed`, `embed` |
| `*ForRewardModeling`, `*RewardModel` | `embed` | `token_embed`, `embed` |
| `*For*Classification`, `*ClassificationModel` | `classify` | `token_classify`, `classify` |

!!! tip
    您可以显式设置 `--convert <type>` 来指定如何转换模型。

### Pooler 配置

#### 预定义模型

如果模型定义的 [Pooler][vllm.model_executor.layers.pooler.Pooler] 接受 `pooler_config`，
您可以通过 `--pooler-config` 选项覆盖其某些属性。

#### 转换后的模型

如果模型已通过 `--convert` 转换（见上文），
分配给每个任务的池化器默认具有以下属性：

| 任务 | 池化类型 | 归一化 | Softmax |
| ---------- | ------------ | ------------- | ------- |
| `embed` | `LAST` | ✅︎ | ❌ |
| `classify` | `LAST` | ❌ | ✅︎ |

加载 [Sentence Transformers](https://huggingface.co/sentence-transformers) 模型时，
其 Sentence Transformers 配置文件 (`modules.json`) 优先级高于模型的默认设置。

您可以通过 `--pooler-config` 选项进一步自定义，
该选项的优先级高于模型和 Sentence Transformers 的默认设置。

## 已移除的功能

### Encode 任务

我们已将 `encode` 任务拆分为两个更具体的 token 级任务：`token_embed` 和 `token_classify`：

- `token_embed` 与 `embed` 相同，使用归一化作为激活函数。
- `token_classify` 与 `classify` 相同，默认使用 softmax 作为激活函数。

池化模型现在支持 token 级任务。

- 提取隐藏状态首选使用 `token_embed` 任务。
- 命名实体识别 (NER) 和奖励模型首选使用 `token_classify` 任务。

### Score 任务

`score` 任务已在 v0.21 中移除，请改用 `classify`。只有当分类模型输出的 num_labels 等于 1 时，它才能用作评分模型并启用其评分 API。

### 池化多任务支持

池化多任务支持已在 v0.21 中移除。当默认的池化任务不是您想要的时，您需要手动指定，可以通过离线的 `PoolerConfig(task=<task>)` 或在线的 `--pooler-config.task <task>`。
