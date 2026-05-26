# AnythingLLM

[AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) 是一个全栈应用程序，允许您将任何文档、资源或内容片段转换为任何 LLM 在聊天中可用作参考的上下文。

它允许您使用 vLLM 作为后端部署大语言模型（LLM）服务器，该服务器暴露与 OpenAI 兼容的端点。

## 前提条件

设置 vLLM 环境：

```bash
pip install vllm
```

## 部署

1. 使用支持的对话补全模型启动 vLLM 服务器，例如：

    ```bash
    vllm serve Qwen/Qwen1.5-32B-Chat-AWQ --max-model-len 4096
    ```

1. 下载并安装 [AnythingLLM 桌面版](https://anythingllm.com/desktop)。

1. 配置 AI 提供者：

    - 在底部，点击 🔧 扳手图标 -> **打开设置** -> **AI 提供者** -> **LLM**。
    - 输入以下值：
        - LLM 提供者：通用 OpenAI（Generic OpenAI）
        - 基础 URL：`http://{vllm 服务器主机}:{vllm 服务器端口}/v1`
        - 聊天模型名称：`Qwen/Qwen1.5-32B-Chat-AWQ`

    ![设置 AI 提供者](../../assets/deployment/anything-llm-provider.png)

1. 创建工作空间：

    1. 在底部，点击 ↺ 返回图标，返回工作空间。
    1. 创建一个工作空间（例如 `vllm`）并开始聊天。

    ![创建工作空间](../../assets/deployment/anything-llm-chat-without-doc.png)

1. 添加文档。

    1. 点击 📎 附件图标。
    1. 上传文档。
    1. 选择并将文档移动到您的工作空间中。
    1. 保存并嵌入。

    ![添加文档](../../assets/deployment/anything-llm-upload-doc.png)

1. 使用您的文档作为上下文进行聊天。

    ![使用上下文聊天](../../assets/deployment/anything-llm-chat-with-doc.png)
