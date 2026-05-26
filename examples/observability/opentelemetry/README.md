# 设置 OpenTelemetry 概念验证

> **注意：** 核心 OpenTelemetry 包（`opentelemetry-sdk`、`opentelemetry-api`、`opentelemetry-exporter-otlp`、`opentelemetry-semantic-conventions-ai`）已随 vLLM 一起打包。无需手动安装。

1. 在 Docker 容器中启动 Jaeger：

    ```bash
    # 来自：https://www.jaegertracing.io/docs/1.57/getting-started/
    docker run --rm --name jaeger \
        -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
        -p 6831:6831/udp \
        -p 6832:6832/udp \
        -p 5778:5778 \
        -p 16686:16686 \
        -p 4317:4317 \
        -p 4318:4318 \
        -p 14250:14250 \
        -p 14268:14268 \
        -p 14269:14269 \
        -p 9411:9411 \
        jaegertracing/all-in-one:1.57
    ```

1. 在新的 shell 中，导出 Jaeger IP：

    ```bash
    export JAEGER_IP=$(docker inspect   --format '{{ .NetworkSettings.IPAddress }}' jaeger)
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://$JAEGER_IP:4317
    ```

    然后为 OpenTelemetry 设置 vLLM 的服务名称，启用与 Jaeger 的不安全连接，并运行 vLLM：

    ```bash
    export OTEL_SERVICE_NAME="vllm-server"
    export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
    vllm serve facebook/opt-125m --otlp-traces-endpoint="$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    ```

1. 在新的 shell 中，从模拟客户端发送带有追踪上下文的请求

    ```bash
    export JAEGER_IP=$(docker inspect --format '{{ .NetworkSettings.IPAddress }}' jaeger)
    export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=grpc://$JAEGER_IP:4317
    export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true
    export OTEL_SERVICE_NAME="client-service"
    python dummy_client.py
    ```

1. 打开 Jaeger WebUI：<http://localhost:16686/>

    在搜索面板中，选择 `vllm-server` 服务并点击 `Find Traces`。您应该会看到一个追踪列表，每个请求对应一个追踪。
    ![Traces](https://i.imgur.com/GYHhFjo.png)

1. 点击一个追踪将显示其 spans 及其标签。在此演示中，每个追踪有 2 个 spans。一个来自包含提示文本的模拟客户端，另一个来自包含请求元数据的 vLLM。
![Spans details](https://i.imgur.com/OPf6CBL.png)

## 导出器协议

OpenTelemetry 支持 `grpc` 或 `http/protobuf` 作为导出器中追踪数据的传输协议。
默认情况下，使用 `grpc`。要将 `http/protobuf` 设置为协议，请按如下方式配置 `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` 环境变量：

```bash
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://$JAEGER_IP:4318/v1/traces
vllm serve facebook/opt-125m --otlp-traces-endpoint="$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
```

## FastAPI 的插桩

OpenTelemetry 允许对 FastAPI 进行自动插桩。

1. 安装插桩库

    ```bash
    pip install opentelemetry-instrumentation-fastapi
    ```

1. 使用 `opentelemetry-instrument` 运行 vLLM

    ```bash
    opentelemetry-instrument vllm serve facebook/opt-125m
    ```

1. 向 vLLM 发送请求并在 Jaeger 中查找其追踪。它应包含来自 FastAPI 的 spans。

![FastAPI Spans](https://i.imgur.com/hywvoOJ.png)
