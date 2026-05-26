# AutoGen

[AutoGen](https://github.com/microsoft/autogen) 是一个用于创建多智能体 AI 应用程序的框架，这些应用程序可以自主运行或与人类协作。

## 前提条件

设置 vLLM 和 [AutoGen](https://microsoft.github.io/autogen/0.2/docs/installation/) 环境：

```bash
pip install vllm

# 安装 AgentChat 和来自扩展的 OpenAI 客户端
# AutoGen 需要 Python 3.10 或更高版本。
pip install -U "autogen-agentchat" "autogen-ext[openai]"
```

## 部署

1. 启动 vLLM 服务器，使用受支持的聊天补全模型，例如：

    ```bash
    vllm serve mistralai/Mistral-7B-Instruct-v0.2
    ```

1. 使用 AutoGen 调用它：

??? code

    ```python
    import asyncio
    from autogen_core.models import UserMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_core.models import ModelFamily


    async def main() -> None:
        # 创建模型客户端
        model_client = OpenAIChatCompletionClient(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            base_url="http://{your-vllm-host-ip}:{your-vllm-host-port}/v1",
            api_key="EMPTY",
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "family": ModelFamily.MISTRAL,
                "structured_output": True,
            },
        )

        messages = [UserMessage(content="Write a very short story about a dragon.", source="user")]

        # 创建流。
        stream = model_client.create_stream(messages=messages)

        # 迭代流并打印响应。
        print("Streamed responses:")
        async for response in stream:
            if isinstance(response, str):
                # 部分响应是一个字符串。
                print(response, flush=True, end="")
            else:
                # 最后一个响应是包含完整消息的 CreateResult 对象。
                print("\n\n------------\n")
                print("The complete response:", flush=True)
                print(response.content, flush=True)

        # 完成后关闭客户端。
        await model_client.close()


    asyncio.run(main())
    ```

详细信息请参阅教程：

- [在 AutoGen 中使用 vLLM](https://microsoft.github.io/autogen/0.2/docs/topics/non-openai-models/local-vllm/)

- [OpenAI 兼容 API 示例](https://microsoft.github.io/autogen/stable/reference/python/autogen_ext.models.openai.html#autogen_ext.models.openai.OpenAIChatCompletionClient)
