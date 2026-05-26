# Haystack

[Haystack](https://github.com/deepset-ai/haystack) 是一个端到端的 LLM 框架，允许您构建由 LLM、Transformer 模型、向量搜索等技术驱动的应用程序。无论您想要执行检索增强生成（RAG）、文档搜索、问答还是答案生成，Haystack 都可以将最先进的嵌入模型和 LLM 编排成流水线，以构建端到端的 NLP 应用程序并解决您的使用场景。

它允许您使用 vLLM 作为后端部署大语言模型（LLM）服务器，该服务器暴露与 OpenAI 兼容的端点。

## 前提条件

设置 vLLM 和 Haystack 环境：

```bash
pip install vllm haystack-ai
```

## 部署

1. 使用支持的对话补全模型启动 vLLM 服务器，例如：

    ```bash
    vllm serve mistralai/Mistral-7B-Instruct-v0.1
    ```

1. 在 Haystack 中使用 `OpenAIGenerator` 和 `OpenAIChatGenerator` 组件查询 vLLM 服务器。

??? code

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret

    generator = OpenAIChatGenerator(
        # for compatibility with the OpenAI API, a placeholder api_key is needed
        api_key=Secret.from_token("VLLM-PLACEHOLDER-API-KEY"),
        model="mistralai/Mistral-7B-Instruct-v0.1",
        api_base_url="http://{your-vLLM-host-ip}:{your-vLLM-host-port}/v1",
        generation_kwargs={"max_tokens": 512},
    )

    response = generator.run(
      messages=[ChatMessage.from_user("Hi. Can you help me plan my next trip to Italy?")]
    )

    print("-"*30)
    print(response)
    print("-"*30)
    ```

```console
------------------------------
{'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=' Of course! Where in Italy would you like to go and what type of trip are you looking to plan?')], _name=None, _meta={'model': 'mistralai/Mistral-7B-Instruct-v0.1', 'index': 0, 'finish_reason': 'stop', 'usage': {'completion_tokens': 23, 'prompt_tokens': 21, 'total_tokens': 44, 'completion_tokens_details': None, 'prompt_tokens_details': None}})]}
------------------------------
```

有关详细信息，请参阅教程 [在 Haystack 中使用 vLLM](https://github.com/deepset-ai/haystack-integrations/blob/main/integrations/vllm.md)。
