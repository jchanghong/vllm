# 在线服务

vLLM 提供了一个与多种接口兼容的 HTTP 服务器！

## OpenAI 兼容服务器

我们目前支持以下 OpenAI API：

- [补全 API](./openai_compatible_server.md#completions-api)（`/v1/completions`）
    - 仅适用于[文本生成模型](../../models/generative_models.md)。
    - *注意：不支持 `suffix` 参数。*
- [响应 API](./openai_compatible_server.md#responses-api)（`/v1/responses`）
    - 仅适用于[文本生成模型](../../models/generative_models.md)。
- [聊天补全 API](./openai_compatible_server.md#chat-api)（`/v1/chat/completions`）
    - 仅适用于带有[聊天模板](./openai_compatible_server.md#chat-template)的[文本生成模型](../../models/generative_models.md)。
    - *注意：`user` 参数被忽略。*
    - *注意：* 将 `parallel_tool_calls` 参数设置为 `false` 可确保 vLLM 每个请求仅返回零个或一个工具调用。设置为 `true`（默认值）允许每个请求返回多个工具调用。如果设置为 `true`，并不能保证一定会返回多个工具调用，因为该行为取决于模型，并非所有模型都设计为支持并行工具调用。
- [嵌入 API](../../models/pooling_models/embed.md#openai-compatible-embeddings-api)（`/v1/embeddings`）
    - 仅适用于[嵌入模型](../../models/pooling_models/embed.md)。
- [转录 API](./speech_to_text.md#transcriptions-api)（`/v1/audio/transcriptions`）
    - 仅适用于[自动语音识别（ASR）模型](../../models/supported_models.md#transcription)。
- [翻译 API](./speech_to_text.md#translations-api)（`/v1/audio/translations`）
    - 仅适用于[自动语音识别（ASR）模型](../../models/supported_models.md#transcription)。

## Anthropic API

- Anthropic messages API（`/v1/messages`）

## Cohere API

- [Cohere Embed API](../../models/pooling_models/embed.md#cohere-embed-api)（`/v2/embed`）
    - 与 [Cohere 的 Embed API](https://docs.cohere.com/reference/embed) 兼容
    - 适用于任何[嵌入模型](../../models/pooling_models/embed.md#supported-models)，包括多模态模型。
- [Cohere Rerank API](../../models/pooling_models/scoring.md#rerank-api)（`/rerank`、`/v1/rerank`、`/v2/rerank`）
    - 实现了 [Jina AI 的 v1 重排序 API](https://jina.ai/reranker/)
    - 与 [Cohere 的 v1 和 v2 重排序 API](https://docs.cohere.com/v2/reference/rerank) 兼容

## SageMaker API

- `/invocations`——兼容 SageMaker 的端点（路由到与 `/v1` 端点相同的推理函数）

## 池化 API

有关池化模型的更多详细信息，请参考[此页面](../../models/pooling_models/README.md)。

- [分类用法](../../models/pooling_models/classify.md)
    - [分类 API](../../models/pooling_models/classify.md#online-serving)（`/classify`）
    - 仅适用于[分类模型](../../models/pooling_models/classify.md)。
- [嵌入用法](../../models/pooling_models/embed.md)
    - [Cohere Embed API](../../models/pooling_models/embed.md#cohere-embed-api)（`/v2/embed`）
    - [兼容 OpenAI 的嵌入 API](../../models/pooling_models/embed.md#openai-compatible-embeddings-api)（`/v1/embeddings`）
    - 仅适用于[嵌入模型](../../models/pooling_models/embed.md)。
- [评分用法](../../models/pooling_models/scoring.md)
    - [评分 API](../../models/pooling_models/scoring.md#score-api)（`/score`）
    - [Cohere Rerank API](../../models/pooling_models/scoring.md#rerank-api)（`/rerank`、`/v1/rerank`、`/v2/rerank`）
    - 适用于[评分模型](../../models/pooling_models/scoring.md)（交叉编码器、双编码器、后期交互）。
- [池化 API](../../models/pooling_models/README.md#pooling-api)（`/pooling`）
    - 适用于所有[池化模型](../../models/pooling_models/README.md)。

## 语音转文本 API

有关语音转文本的更多详细信息，请参考[此页面](speech_to_text.md)。

- [转录 API](./speech_to_text.md#transcriptions-api)（`/v1/audio/transcriptions`）
    - 仅适用于[自动语音识别（ASR）模型](../../models/supported_models.md#transcription)。
- [翻译 API](./speech_to_text.md#translations-api)（`/v1/audio/translations`）
    - 仅适用于[自动语音识别（ASR）模型](../../models/supported_models.md#transcription)。
- [实时 API](./speech_to_text.md#realtime-api)（`/v1/realtime`）
    - 仅适用于[自动语音识别（ASR）模型](../../models/supported_models.md#realtime-transcription)。

## 分离式 API

### 渲染器 API

有关渲染器 API 的更多详细信息，请参考[此页面](renderer.md)。

- [补全渲染 API](renderer.md)（`/v1/completions/render`）
    - 渲染补全请求
- [聊天补全渲染 API](renderer.md)（`/v1/chat/completions/render`）
    - 渲染聊天补全

## 自定义 API

- [分类 API](../../models/pooling_models/classify.md#classification-api)（`/classify`）
    - 仅适用于[分类模型](../../models/pooling_models/classify.md)。
- [评分 API](../../models/pooling_models/scoring.md#score-api)（`/score`、`/v1/score`）
    - 适用于[评分模型](../../models/pooling_models/scoring.md)（交叉编码器、双编码器、后期交互）。
- [池化 API](../../models/pooling_models/README.md#pooling-api)（`/pooling`）
    - 适用于所有[池化模型](../../models/pooling_models/README.md)。
- [生成式评分 API](generative_scoring.md#generative-scoring-api)（`/generative_scoring`）
    - 适用于 [CausalLM 模型](../../models/generative_models.md)（任务 `"generate"`）。
    - 计算指定 `label_token_ids` 的下一个令牌概率。

## 实用 API

- `/tokenize`——分词
- `/detokenize`——逆分词
- `/health`——健康检查
- `/ping`——SageMaker 健康检查
- `/version`——版本信息
- `/load`——服务器负载指标

## 休眠模式 API

有关休眠模式的更多详细信息，请参考[此页面](../../features/sleep_mode.md)。

- `/sleep`——使引擎休眠（会导致服务拒绝）
- `/wake_up`——将引擎从休眠中唤醒
- `/is_sleeping`——检查引擎是否处于休眠状态
- `/collective_rpc`——在引擎上执行任意 RPC 方法（极其危险）

## 聊天模板

为了使语言模型支持聊天协议，vLLM 要求模型在其分词器配置中包含聊天模板。聊天模板是一个 Jinja2 模板，指定角色、消息和其他聊天特定令牌在输入中如何编码。

`NousResearch/Meta-Llama-3-8B-Instruct` 的示例聊天模板可以在[此处](https://llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/#prompt-template-for-meta-llama-3)找到。

有些模型即使经过指令/聊天微调，也不提供聊天模板。对于这些模型，你可以在 `--chat-template` 参数中手动指定聊天模板的文件路径或字符串形式的模板。没有聊天模板，服务器将无法处理聊天，所有聊天请求都将出错。

```bash
vllm serve <model> --chat-template ./path-to-chat-template.jinja
```

vLLM 社区为流行模型提供了一组聊天模板。你可以在 [examples](../../../examples) 目录下找到它们。

随着多模态聊天 API 的加入，OpenAI 规范现在接受一种新格式的聊天消息，该格式同时指定了 `type` 和 `text` 字段。示例如下：

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify this sentiment: vLLM is wonderful!"},
            ],
        },
    ],
)
```

大多数 LLM 的聊天模板期望 `content` 字段是字符串，但一些较新的模型（如 `meta-llama/Llama-Guard-3-1B`）期望内容按照请求中的 OpenAI 模式格式化。vLLM 提供尽力而为的支持来自动检测这一点，并记录为类似 *"Detected the chat template content format to be..."* 的字符串，并在内部转换传入的请求以匹配检测到的格式，可以是以下之一：

- `"string"`：字符串。
    - 示例：`"Hello world"`
- `"openai"`：字典列表，类似于 OpenAI 模式。
    - 示例：`[{"type": "text", "text": "Hello world!"}]`

如果结果不是你期望的，你可以设置 `--chat-template-content-format` CLI 参数来覆盖要使用的格式。

## 离线 API 文档

默认情况下，FastAPI `/docs` 端点需要互联网连接。要在隔离环境中启用离线访问，请使用 `--enable-offline-docs` 标志：

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --enable-offline-docs
```

## Ray Serve LLM

Ray Serve LLM 实现了 vLLM 引擎的可扩展、生产级服务。它与 vLLM 紧密集成，并通过自动扩展、负载均衡和背压等功能进行扩展。

关键能力：

- 公开 OpenAI 兼容的 HTTP API 以及 Python 风格的 API。
- 从单个 GPU 扩展到多节点集群，无需更改代码。
- 通过 Ray 仪表板和指标提供可观测性和自动扩展策略。

以下示例展示了如何使用 Ray Serve LLM 部署像 DeepSeek R1 这样的大模型：[examples/ray_serving/ray_serve_deepseek.py](../../../examples/ray_serving/ray_serve_deepseek.py)。

通过官方 [Ray Serve LLM 文档](https://docs.ray.io/en/latest/serve/llm/index.html)了解更多关于 Ray Serve LLM 的信息。
