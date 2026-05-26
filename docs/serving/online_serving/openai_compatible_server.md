# OpenAI 兼容服务器

vLLM 提供了一个实现 OpenAI [补全 API](https://platform.openai.com/docs/api-reference/completions)、[聊天 API](https://platform.openai.com/docs/api-reference/chat) 等的 HTTP 服务器！此功能使你可以服务模型并使用 HTTP 客户端与之交互。

## 支持的 API

我们目前支持以下 OpenAI API：

- [补全 API](#completions-api)（`/v1/completions`）
    - 仅适用于[文本生成模型](../../models/generative_models.md)。
    - *注意：不支持 `suffix` 参数。*
- [响应 API](#responses-api)（`/v1/responses`）
    - 仅适用于[文本生成模型](../../models/generative_models.md)。
- [聊天补全 API](#chat-api)（`/v1/chat/completions`）
    - 仅适用于带有[聊天模板](../online_serving/README.md#chat-template)的[文本生成模型](../../models/generative_models.md)。
    - *注意：`user` 参数被忽略。*
    - *注意：* 将 `parallel_tool_calls` 参数设置为 `false` 可确保 vLLM 每个请求仅返回零个或一个工具调用。设置为 `true`（默认值）允许每个请求返回多个工具调用。如果设置为 `true`，并不能保证一定会返回多个工具调用，因为该行为取决于模型，并非所有模型都设计为支持并行工具调用。
- [嵌入 API](../../models/pooling_models/embed.md#openai-compatible-embeddings-api)（`/v1/embeddings`）
    - 仅适用于[嵌入模型](../../models/pooling_models/embed.md)。
- [转录 API](./speech_to_text.md#transcriptions-api)（`/v1/audio/transcriptions`）
    - 仅适用于[自动语音识别（ASR）模型](../../models/supported_models.md#transcription)。
- [翻译 API](./speech_to_text.md#translations-api)（`/v1/audio/translations`）
    - 仅适用于[自动语音识别（ASR）模型](../../models/supported_models.md#transcription)。

## 补全 API

在终端中，你可以[安装](../../getting_started/installation/README.md) vLLM，然后使用 [`vllm serve`](../../configuration/serve_args.md) 命令启动服务器。（你也可以使用我们的 [Docker](../../deployment/docker.md) 镜像。）

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key token-abc123
```

要调用服务器，在你喜欢的文本编辑器中创建一个使用 HTTP 客户端的脚本。包含你想要发送给模型的任何消息。然后运行该脚本。以下是使用[官方 OpenAI Python 客户端](https://github.com/openai/openai-python)的示例脚本。

??? code

    ```python
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    completion = client.chat.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "user", "content": "Hello!"},
        ],
    )

    print(completion.choices[0].message)
    ```

!!! tip
    vLLM 支持一些 OpenAI 不支持的参数，例如 `top_k`。
    你可以在请求的 `extra_body` 参数中将这些参数传递给 vLLM，即 `extra_body={"top_k": 50}`。

!!! important
    默认情况下，如果 Hugging Face 模型仓库中存在 `generation_config.json`，服务器会应用它。这意味着某些采样参数的默认值可能会被模型创建者推荐的参数覆盖。

    要禁用此行为，请在启动服务器时传递 `--generation-config vllm`。

## 额外参数

vLLM 支持一组不属于 OpenAI API 的参数。
要使用它们，你可以将它们作为 OpenAI 客户端中的额外参数传递。
或者，如果你直接使用 HTTP 调用，可以直接将它们合并到 JSON 负载中。

```python
completion = client.chat.completions.create(
    model="NousResearch/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"},
    ],
    extra_body={
        "structured_outputs": {"choice": ["positive", "negative"]},
    },
)
```

## 额外 HTTP 头

目前仅支持 `X-Request-Id` HTTP 请求头。可以通过 `--enable-request-id-headers` 启用。

??? code

    ```python
    completion = client.chat.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"},
        ],
        extra_headers={
            "x-request-id": "sentiment-classification-00001",
        },
    )
    print(completion._request_id)

    completion = client.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        prompt="A robot may not injure a human being",
        extra_headers={
            "x-request-id": "completion-test",
        },
    )
    print(completion._request_id)
    ```

## API 参考

### 补全 API

我们的补全 API 与 [OpenAI 的补全 API](https://platform.openai.com/docs/api-reference/completions) 兼容；
你可以使用[官方 OpenAI Python 客户端](https://github.com/openai/openai-python)与之交互。

代码示例：[examples/basic/online_serving/openai_completion_client.py](../../../examples/basic/online_serving/openai_completion_client.py)

#### 额外参数

支持以下[采样参数](../../api/README.md#inference-parameters)。

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/completion/protocol.py:completion-sampling-params"
    ```

支持以下额外参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/completion/protocol.py:completion-extra-params"
    ```

### 聊天 API

我们的聊天 API 与 [OpenAI 的聊天补全 API](https://platform.openai.com/docs/api-reference/chat) 兼容；
你可以使用[官方 OpenAI Python 客户端](https://github.com/openai/openai-python)与之交互。

我们支持 [Vision](https://platform.openai.com/docs/guides/vision) 和
[Audio](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in) 相关参数；
更多信息请参见我们的[多模态输入](../../features/multimodal_inputs.md)指南。

- *注意：不支持 `image_url.detail` 参数。*

代码示例：[examples/basic/online_serving/openai_chat_completion_client.py](../../../examples/basic/online_serving/openai_chat_completion_client.py)

#### 额外参数

支持以下[采样参数](../../api/README.md#inference-parameters)。

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/chat_completion/protocol.py:chat-completion-sampling-params"
    ```

支持以下额外参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/chat_completion/protocol.py:chat-completion-extra-params"
    ```

### 响应 API

我们的响应 API 与 [OpenAI 的响应 API](https://platform.openai.com/docs/api-reference/responses) 兼容；
你可以使用[官方 OpenAI Python 客户端](https://github.com/openai/openai-python)与之交互。

代码示例：[examples/tool_calling/openai_responses_client_with_tools.py](../../../examples/tool_calling/openai_responses_client_with_tools.py)

#### 额外参数

请求对象中支持以下额外参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/responses/protocol.py:responses-extra-params"
    ```

响应对象中支持以下额外参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/openai/responses/protocol.py:responses-response-extra-params"
    ```
