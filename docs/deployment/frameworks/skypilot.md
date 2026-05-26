# SkyPilot

<p align="center">
  <img src="https://imgur.com/yxtzPEu.png" alt="vLLM"/>
</p>

vLLM 可以通过 [SkyPilot](https://github.com/skypilot-org/skypilot) **在云和 Kubernetes 上运行并扩展至多个服务副本**，SkyPilot 是一个用于在任何云上运行 LLM 的开源框架。更多关于各种开放模型（如 Llama-3、Mixtral 等）的示例，请参阅 [SkyPilot AI 画廊](https://skypilot.readthedocs.io/en/latest/gallery/index.html)。

## 前提条件

- 前往 [HuggingFace 模型页面](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 并申请访问模型 `meta-llama/Meta-Llama-3-8B-Instruct`。
- 确保您已安装 SkyPilot（[文档](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html)）。
- 确保 `sky check` 显示已启用云或 Kubernetes。

```bash
pip install skypilot-nightly
sky check
```

## 在单实例上运行

请参阅用于服务的 vLLM SkyPilot YAML 文件 [serving.yaml](https://github.com/skypilot-org/skypilot/blob/master/llm/vllm/serve.yaml)。

??? code "Yaml"

    ```yaml
    resources:
      accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # 8B 模型可使用更便宜的加速器。
      use_spot: True
      disk_size: 512  # 确保模型检查点能够容纳。
      disk_tier: best
      ports: 8081  # 暴露于互联网流量。

    envs:
      PYTHONUNBUFFERED: 1
      MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
      HF_TOKEN: <your-huggingface-token>  # 替换为您自己的 huggingface token，或使用 --env 传递。

    setup: |
      conda create -n vllm python=3.10 -y
      conda activate vllm

      pip install vllm==0.4.0.post1
      # 安装 Gradio 用于 Web UI。
      pip install gradio openai
      pip install flash-attn==2.5.7

    run: |
      conda activate vllm
      echo 'Starting vllm api server...'
      vllm serve $MODEL_NAME \
        --port 8081 \
        --trust-remote-code \
        --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
        2>&1 | tee api_server.log &

      echo 'Waiting for vllm api server to start...'
      while ! `cat api_server.log | grep -q 'Uvicorn running on'`; do sleep 1; done

      echo 'Starting gradio server...'
      git clone https://github.com/vllm-project/vllm.git || true
      python vllm/examples/applications/chatbot/gradio_openai_chatbot_webserver.py \
        -m $MODEL_NAME \
        --port 8811 \
        --model-url http://localhost:8081/v1 \
        --stop-token-ids 128009,128001
    ```

在列出的任何候选 GPU（L4、A10g 等）上启动 Llama-3 8B 模型的服务：

```bash
HF_TOKEN="your-huggingface-token" sky launch serving.yaml --env HF_TOKEN
```

查看命令的输出。将会有一个可共享的 gradio 链接（如下面的最后一行）。在浏览器中打开它，使用 LLaMA 模型进行文本补全。

```console
(task, pid=7431) Running on public URL: https://<gradio-hash>.gradio.live
```

**可选**：提供 70B 模型而非默认的 8B 模型，并使用更多 GPU：

```bash
HF_TOKEN="your-huggingface-token" \
  sky launch serving.yaml \
  --gpus A100:8 \
  --env HF_TOKEN \
  --env MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct
```

## 扩展至多个副本

SkyPilot 可以通过内置的自动缩放、负载均衡和容错功能，将服务扩展到多个服务副本。只需在 YAML 文件中添加 services 部分即可实现。

??? code "Yaml"

    ```yaml
    service:
      replicas: 2
      # An actual request for readiness probe.
      readiness_probe:
        path: /v1/chat/completions
        post_data:
        model: $MODEL_NAME
        messages:
          - role: user
            content: Hello! What is your name?
      max_completion_tokens: 1
    ```

??? code "Yaml"

    ```yaml
    service:
      replicas: 2
      # An actual request for readiness probe.
      readiness_probe:
        path: /v1/chat/completions
        post_data:
          model: $MODEL_NAME
          messages:
            - role: user
              content: Hello! What is your name?
          max_completion_tokens: 1

    resources:
      accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # 8B 模型可使用更便宜的加速器。
      use_spot: True
      disk_size: 512  # 确保模型检查点能够容纳。
      disk_tier: best
      ports: 8081  # 暴露于互联网流量。

    envs:
      PYTHONUNBUFFERED: 1
      MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
      HF_TOKEN: <your-huggingface-token>  # 替换为您自己的 huggingface token，或使用 --env 传递。

    setup: |
      conda create -n vllm python=3.10 -y
      conda activate vllm

      pip install vllm==0.4.0.post1
      # 安装 Gradio 用于 Web UI。
      pip install gradio openai
      pip install flash-attn==2.5.7

    run: |
      conda activate vllm
      echo 'Starting vllm api server...'
      vllm serve $MODEL_NAME \
        --port 8081 \
        --trust-remote-code \
        --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
        2>&1 | tee api_server.log
    ```

在多个副本上启动 Llama-3 8B 模型的服务：

```bash
HF_TOKEN="your-huggingface-token" \
  sky serve up -n vllm serving.yaml \
  --env HF_TOKEN
```

等待服务就绪：

```bash
watch -n10 sky serve status vllm
```

示例输出：

```console
Services
NAME  VERSION  UPTIME  STATUS  REPLICAS  ENDPOINT
vllm  1        35s     READY   2/2       xx.yy.zz.100:30001

Service Replicas
SERVICE_NAME  ID  VERSION  IP            LAUNCHED     RESOURCES                STATUS  REGION
vllm          1   1        xx.yy.zz.121  18 mins ago  1x GCP([Spot]{'L4': 1})  READY   us-east4
vllm          2   1        xx.yy.zz.245  18 mins ago  1x GCP([Spot]{'L4': 1})  READY   us-east4
```

服务就绪后，您可以找到服务的单一端点并通过该端点访问服务：

??? console "Commands"

    ```bash
    ENDPOINT=$(sky serve status --endpoint 8081 vllm)
    curl -L http://$ENDPOINT/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "Who are you?"
        }
        ],
        "stop_token_ids": [128009,  128001]
      }'
    ```

要启用自动缩放，您可以将 `service` 中的 `replicas` 替换为以下配置：

```yaml
service:
  replica_policy:
    min_replicas: 2
    max_replicas: 4
    target_qps_per_replica: 2
```

当每个副本的 QPS 超过 2 时，此配置将自动扩展服务。

??? code "Yaml"

    ```yaml
    service:
      replica_policy:
        min_replicas: 2
        max_replicas: 4
        target_qps_per_replica: 2
      # 用于就绪检查的实际请求。
      readiness_probe:
        path: /v1/chat/completions
        post_data:
          model: $MODEL_NAME
          messages:
            - role: user
              content: Hello! What is your name?
          max_completion_tokens: 1

    resources:
      accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # 8B 模型可使用更便宜的加速器。
      use_spot: True
      disk_size: 512  # 确保模型检查点能够容纳。
      disk_tier: best
      ports: 8081  # 暴露于互联网流量。

    envs:
      PYTHONUNBUFFERED: 1
      MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
      HF_TOKEN: <your-huggingface-token>  # 替换为您自己的 huggingface token，或使用 --env 传递。

    setup: |
      conda create -n vllm python=3.10 -y
      conda activate vllm

      pip install vllm==0.4.0.post1
      # 安装 Gradio 用于 Web UI。
      pip install gradio openai
      pip install flash-attn==2.5.7

    run: |
      conda activate vllm
      echo 'Starting vllm api server...'
      vllm serve $MODEL_NAME \
        --port 8081 \
        --trust-remote-code \
        --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
        2>&1 | tee api_server.log
    ```

使用新配置更新服务：

```bash
HF_TOKEN="your-huggingface-token" sky serve update vllm serving.yaml --env HF_TOKEN
```

停止服务：

```bash
sky serve down vllm
```

### **可选**：将 GUI 连接到端点

也可以使用单独的 GUI 前端访问 Llama-3 服务，这样发送到 GUI 的用户请求将在副本之间进行负载均衡。

??? code "Yaml"

    ```yaml
    envs:
      MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
      ENDPOINT: x.x.x.x:3031 # 运行 vllm 的 API 服务器地址。

    resources:
      cpus: 2

    setup: |
      conda create -n vllm python=3.10 -y
      conda activate vllm

      # 安装 Gradio 用于 Web UI。
      pip install gradio openai

    run: |
      conda activate vllm
      export PATH=$PATH:/sbin

      echo 'Starting gradio server...'
      git clone https://github.com/vllm-project/vllm.git || true
      python vllm/examples/applications/api_client/gradio_openai_chatbot_webserver.py \
        -m $MODEL_NAME \
        --port 8811 \
        --model-url http://$ENDPOINT/v1 \
        --stop-token-ids 128009,128001 | tee ~/gradio.log
    ```

1. 启动聊天 Web UI：

    ```bash
    sky launch \
      -c gui ./gui.yaml \
      --env ENDPOINT=$(sky serve status --endpoint vllm)
    ```

2. 然后，我们可以通过返回的 gradio 链接访问 GUI：

    ```console
    | INFO | stdout | Running on public URL: https://6141e84201ce0bb4ed.gradio.live
    ```
