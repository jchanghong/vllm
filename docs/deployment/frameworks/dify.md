# Dify

[Dify](https://github.com/langgenius/dify) 是一个开源 LLM 应用开发平台。其直观的界面结合了智能体 AI 工作流、RAG 流水线、智能体能力、模型管理、可观测性功能等，使您能够快速从原型过渡到生产。

它支持 vLLM 作为模型提供者，以高效地服务大语言模型。

本指南将引导您使用 vLLM 后端部署 Dify。

## 前提条件

设置 vLLM 环境：

```bash
pip install vllm
```

并安装 [Docker](https://docs.docker.com/engine/install/) 和 [Docker Compose](https://docs.docker.com/compose/install/)。

## 部署

1. 启动 vLLM 服务器，使用受支持的聊天补全模型，例如：

    ```bash
    vllm serve Qwen/Qwen1.5-7B-Chat
    ```

1. 使用 docker compose 启动 Dify 服务器（[详情](https://github.com/langgenius/dify?tab=readme-ov-file#quick-start)）：

    ```bash
    git clone https://github.com/langgenius/dify.git
    cd dify
    cd docker
    cp .env.example .env
    docker compose up -d
    ```

1. 打开浏览器访问 `http://localhost/install`，配置基本登录信息并登录。

1. 在右上角用户菜单（个人资料图标下）中，进入设置，然后点击 `Model Provider`，找到 `vLLM` 提供商并安装它。

1. 填写模型提供商详细信息如下：

    - **Model Type**：`LLM`
    - **Model Name**：`Qwen/Qwen1.5-7B-Chat`
    - **API Endpoint URL**：`http://{vllm_server_host}:{vllm_server_port}/v1`
    - **Model Name for API Endpoint**：`Qwen/Qwen1.5-7B-Chat`
    - **Completion Mode**：`Completion`

    ![Dify settings screen](../../assets/deployment/dify-settings.png)

1. 要创建一个测试聊天机器人，进入 `Studio → Chatbot → Create from Blank`，然后选择 Chatbot 作为类型：

    ![Dify create chatbot screen](../../assets/deployment/dify-create-chatbot.png)

1. 点击您刚刚创建的聊天机器人，打开聊天界面并开始与模型交互：

    ![Dify chat screen](../../assets/deployment/dify-chat.png)
