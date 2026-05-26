# Chatbox

[Chatbox](https://github.com/chatboxai/chatbox) 是一款 LLM 桌面客户端，适用于 Windows、Mac 和 Linux。

它允许您使用 vLLM 作为后端部署大语言模型（LLM）服务器，该服务器暴露与 OpenAI 兼容的端点。

## 前提条件

设置 vLLM 环境：

```bash
pip install vllm
```

## 部署

1. 使用支持的对话补全模型启动 vLLM 服务器，例如：

    ```bash
    vllm serve qwen/Qwen1.5-0.5B-Chat
    ```

1. 下载并安装 [Chatbox 桌面版](https://chatboxai.app/en#download)。

1. 在设置的左下角，添加自定义提供者（Add Custom Provider）
    - API 模式：`OpenAI API Compatible`
    - 名称：vllm
    - API 主机：`http://{vllm server host}:{vllm server port}/v1`
    - API 路径：`/chat/completions`
    - 模型：`qwen/Qwen1.5-0.5B-Chat`

    ![Chatbox 设置界面](../../assets/deployment/chatbox-settings.png)

1. 前往 `Just chat`，开始聊天：

    ![聊天机器人聊天界面](../../assets/deployment/chatbox-chat.png)
