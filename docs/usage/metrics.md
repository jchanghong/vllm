# 生产指标

vLLM 公开了许多可用于监控系统健康状态的指标。这些指标通过 vLLM 兼容 OpenAI 的 API 服务器上的 `/metrics` 端点暴露。

您可以使用 Python 或 [Docker](../deployment/docker.md) 启动服务器：

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
```

然后查询端点以获取服务器的最新指标：

??? console "输出"

    ```console
    $ curl http://0.0.0.0:8000/metrics

    # HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
    # TYPE vllm:iteration_tokens_total histogram
    vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
    vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    ...
    ```

公开了以下指标：

## 通用指标

--8<-- "docs/generated/metrics/general.inc.md"

## 推测解码指标

--8<-- "docs/generated/metrics/spec_decode.inc.md"

## NIXL KV 连接器指标

--8<-- "docs/generated/metrics/nixl_connector.inc.md"

## 模型算力利用率（MFU）性能指标

这些指标可通过 `--enable-mfu-metrics` 获得：

--8<-- "docs/generated/metrics/perf.inc.md"

## 弃用策略

注意：当指标在版本 `X.Y` 中被弃用时，它们在版本 `X.Y+1` 中被隐藏，但可以使用 `--show-hidden-metrics-for-version=X.Y` 这个逃生口重新启用，然后在版本 `X.Y+2` 中被移除。
