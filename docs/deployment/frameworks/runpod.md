# RunPod

vLLM 可以部署在 [RunPod](https://www.runpod.io/) 上，RunPod 是一个云 GPU 平台，提供按需和无服务器 GPU 实例用于 AI 推理工作负载。

## 前提条件

- 一个具有 GPU Pod 访问权限的 RunPod 账户
- 一个运行 CUDA 兼容模板的 GPU Pod（例如 `runpod/pytorch`）

## 启动服务器

通过 SSH 登录到您的 RunPod Pod 并启动 vLLM 的 OpenAI 兼容服务器：

```bash
vllm serve <model-name> \
    --host 0.0.0.0 \
    --port 8000
```

!!! note

    使用 `--host 0.0.0.0` 绑定到所有接口，以便服务器可从容器外部访问。

## 暴露端口 8000

RunPod 通过其代理暴露 HTTP 服务。要使端口 8000 可访问：

1. 在 RunPod 仪表板中，导航到您的 Pod 设置。
2. 将 `8000` 添加到暴露的 HTTP 端口列表中。
3. Pod 重启后，RunPod 会提供一个格式如下的公共 URL：

    ```text
    https://<pod-id>-8000.proxy.runpod.net
    ```

## 排查 502 Bad Gateway 错误

来自 RunPod 代理的 `502 Bad Gateway` 错误通常表示服务器尚未开始监听。常见原因：

- **模型仍在加载中** — 大型模型需要时间下载并加载到 GPU 内存中。检查 Pod 日志以了解进度。
- **主机绑定错误** — 确保您传递了 `--host 0.0.0.0`。绑定到 `127.0.0.1`（默认值）会使服务器无法从代理访问。
- **端口不匹配** — 验证 `--port` 值与 RunPod 仪表板中暴露的端口相匹配。
- **GPU 内存不足** — 模型可能对于分配的 GPU 来说过大。检查日志中是否有 CUDA OOM 错误，并考虑使用更大的实例或为多 GPU Pod 添加 `--tensor-parallel-size`。

## 验证部署

服务器运行后，使用 curl 请求进行测试：

!!! console "Command"

    ```bash
    curl https://<pod-id>-8000.proxy.runpod.net/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "<model-name>",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "max_tokens": 50
        }'
    ```

!!! console "Response"

    ```json
    {
        "id": "chat-abc123",
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'm doing well, thank you for asking! How can I help you today?"
                },
                "index": 0,
                "finish_reason": "stop"
            }
        ]
    }
    ```

您也可以检查服务器健康状态端点：

```bash
curl https://<pod-id>-8000.proxy.runpod.net/health
```
