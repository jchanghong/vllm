# Prometheus 和 Grafana

这是一个简单示例，向您展示如何将 vLLM 指标日志连接到 Prometheus/Grafana 栈。在本示例中，我们通过 Docker 启动 Prometheus 和 Grafana。您可以通过 [Prometheus](https://prometheus.io/) 和 [Grafana](https://grafana.com/) 网站了解其他方法。

安装：

- [`docker`](https://docs.docker.com/engine/install/)
- [`docker compose`](https://docs.docker.com/compose/install/linux/#install-using-the-repository)

## 启动

Prometheus 指标日志记录在 OpenAI 兼容服务器中默认启用。通过入口点启动：

```bash
vllm serve mistralai/Mistral-7B-v0.1 \
    --max-model-len 2048
```

使用 `docker compose` 启动 Prometheus 和 Grafana 服务器：

```bash
docker compose up
```

向服务器提交一些示例请求：

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
    --model mistralai/Mistral-7B-v0.1 \
    --tokenizer mistralai/Mistral-7B-v0.1 \
    --endpoint /v1/completions \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 3.0
```

访问 [`http://localhost:8000/metrics`](http://localhost:8000/metrics) 将显示 vLLM 暴露的原始 Prometheus 指标。

## Grafana 仪表板

访问 [`http://localhost:3000`](http://localhost:3000)。使用默认用户名（`admin`）和密码（`admin`）登录。

### 添加 Prometheus 数据源

访问 [`http://localhost:3000/connections/datasources/new`](http://localhost:3000/connections/datasources/new) 并选择 Prometheus。

在 Prometheus 配置页面上，我们需要在 `Connection` 中添加 `Prometheus Server URL`。对于此设置，Grafana 和 Prometheus 运行在单独的容器中，但 Docker 为每个容器创建了 DNS 名称。您可以直接使用 `http://prometheus:9090`。

点击 `Save & Test`。您应该会看到一个绿色的勾选，显示 "Successfully queried the Prometheus API."。

### 导入仪表板

访问 [`http://localhost:3000/dashboard/import`](http://localhost:3000/dashboard/import)，上传 `grafana.json`，并选择 `prometheus` 数据源。您应该会看到如下所示的界面：

![Grafana Dashboard Image](https://i.imgur.com/R2vH9VW.png)
