# 指标

vLLM 为 V1 引擎暴露了丰富的指标集，以支持可观测性和容量规划。

## 目标

- 提供引擎级和请求级指标的全面覆盖，以辅助生产监控。
- 优先支持 Prometheus 集成，这是我们预期在生产环境中使用的方式。
- 提供日志支持（即向信息日志打印指标），用于临时测试、调试、开发和探索性用例。

## 背景

vLLM 中的指标可以分类如下：

1. **服务器级指标**：追踪 LLM 引擎状态和性能的全局指标。这些通常在 Prometheus 中作为 Gauge 或 Counter 暴露。
2. **请求级指标**：追踪单个请求特征（如大小和时序）的指标。这些通常在 Prometheus 中作为 Histogram 暴露，并且通常是监控 vLLM 的 SRE 所追踪的 SLO。

其思维模型是服务器级指标有助于解释请求级指标的值。

### 指标概览

### v1 指标

在 v1 中，通过 Prometheus 兼容的 `/metrics` 端点使用 `vllm:` 前缀暴露了大量指标，例如：

- `vllm:num_requests_running`（Gauge）- 当前正在运行的请求数。
- `vllm:kv_cache_usage_perc`（Gauge）- 已使用的 KV 缓存块比例（0-1）。
- `vllm:prefix_cache_queries`（Counter）- 前缀缓存查询次数。
- `vllm:prefix_cache_hits`（Counter）- 前缀缓存命中次数。
- `vllm:prompt_tokens_total`（Counter）- 已处理的 prompt token 总数。
- `vllm:generation_tokens_total`（Counter）- 已生成的 token 总数。
- `vllm:request_success_total`（Counter）- 已完成的请求数（按完成原因分类）。
- `vllm:request_prompt_tokens`（Histogram）- 输入 prompt token 数量的直方图。
- `vllm:request_generation_tokens`（Histogram）- 生成 token 数量的直方图。
- `vllm:time_to_first_token_seconds`（Histogram）- 首 token 延迟（TTFT）。
- `vllm:inter_token_latency_seconds`（Histogram）- token 间延迟。
- `vllm:e2e_request_latency_seconds`（Histogram）- 端到端请求延迟。
- `vllm:request_prefill_time_seconds`（Histogram）- 请求预填充时间。
- `vllm:request_decode_time_seconds`（Histogram）- 请求解码时间。

这些记录在[推理与服务 -> 生产指标](../usage/metrics.md)下。

### Grafana 仪表盘

vLLM 还提供了[一个参考示例](../../examples/observability/prometheus_grafana/README.md)，说明如何使用 Prometheus 收集和存储这些指标，以及如何使用 Grafana 仪表盘进行可视化。

Grafana 仪表盘中暴露的指标子集显示了哪些指标特别重要：

- `vllm:e2e_request_latency_seconds_bucket` - 端到端请求延迟（秒）。
- `vllm:prompt_tokens` - Prompt token。
- `vllm:generation_tokens` - 生成 token。
- `vllm:inter_token_latency_seconds` - token 间延迟（每输出 token 时间，TPOT，秒）。
- `vllm:time_to_first_token_seconds` - 首 token 延迟（TTFT，秒）。
- `vllm:num_requests_running`（还有 `_swapped` 和 `_waiting`）- RUNNING、WAITING 和 SWAPPED 状态中的请求数。
- `vllm:kv_cache_usage_perc` - vLLM 使用的缓存块百分比。
- `vllm:request_prompt_tokens` - 请求 prompt 长度。
- `vllm:request_generation_tokens` - 请求生成长度。
- `vllm:request_success` - 按完成原因统计的已完成请求数：要么生成了 EOS token，要么达到了最大序列长度。
- `vllm:request_queue_time_seconds` - 排队时间。
- `vllm:request_prefill_time_seconds` - 请求预填充时间。
- `vllm:request_decode_time_seconds` - 请求解码时间。
- `vllm:request_max_num_generation_tokens` - 序列组中的最大生成 token 数。

参见[添加此仪表盘的 PR](https://github.com/vllm-project/vllm/pull/2316) 了解所做选择的有趣且有用的背景信息。

### Prometheus 客户端库

最初使用 [aioprometheus 库](https://github.com/vllm-project/vllm/pull/1890)添加了 Prometheus 支持，但很快切换到了 [prometheus_client](https://github.com/vllm-project/vllm/pull/2730)。相关理由在两个链接的 PR 中均有讨论。

在这些迁移过程中，我们短暂丢失了一个用于追踪 HTTP 指标的 `MetricsMiddleware`，但后来使用 [prometheus_fastapi_instrumentator](https://github.com/vllm-project/vllm/pull/15657) 恢复了：

```bash
$ curl http://0.0.0.0:8000/metrics 2>/dev/null  | grep -P '^http_(?!.*(_bucket|_created|_sum)).*'
http_requests_total{handler="/v1/completions",method="POST",status="2xx"} 201.0
http_request_size_bytes_count{handler="/v1/completions"} 201.0
http_response_size_bytes_count{handler="/v1/completions"} 201.0
http_request_duration_highr_seconds_count 201.0
http_request_duration_seconds_count{handler="/v1/completions",method="POST"} 201.0
```

### 多进程模式

历史上，指标在引擎核心进程中收集，并使用多进程模式使其在 API 服务器进程中可用。参见 <https://github.com/vllm-project/vllm/pull/7279>。

最近，指标在 API 服务器进程中收集，多进程模式仅在 `--api-server-count > 1` 时使用。参见 <https://github.com/vllm-project/vllm/pull/17546> 和[API 服务器扩展](../serving/data_parallel_deployment.md#internal-load-balancing)的详细信息。

### 内置的 Python/进程指标

以下指标默认由 `prometheus_client` 支持，但在使用多进程模式时不暴露：

- `python_gc_objects_collected_total`
- `python_gc_objects_uncollectable_total`
- `python_gc_collections_total`
- `python_info`
- `process_virtual_memory_bytes`
- `process_resident_memory_bytes`
- `process_start_time_seconds`
- `process_cpu_seconds_total`
- `process_open_fds`
- `process_max_fds`

因此，当 `--api-server-count > 1` 时，这些指标不可用。它们的相关性存疑，因为它们不会聚合构成 vLLM 实例的所有进程的这些统计数据。

## 指标设计

["更好的可观测性"](https://github.com/vllm-project/vllm/issues/3616) 功能是大部分指标设计的规划场所。例如，参见[详细路线图](https://github.com/vllm-project/vllm/issues/3616#issuecomment-2030858781)的制定。

### 旧版 PR

为了帮助理解指标设计的背景，以下是一些添加了原始（现为旧版）指标的相关 PR：

- <https://github.com/vllm-project/vllm/pull/1890>
- <https://github.com/vllm-project/vllm/pull/2316>
- <https://github.com/vllm-project/vllm/pull/2730>
- <https://github.com/vllm-project/vllm/pull/4464>
- <https://github.com/vllm-project/vllm/pull/7279>

### 指标实现 PR

以下是相关背景的指标实现 PR，参见 <https://github.com/vllm-project/vllm/issues/10582>：

- <https://github.com/vllm-project/vllm/pull/11962>
- <https://github.com/vllm-project/vllm/pull/11973>
- <https://github.com/vllm-project/vllm/pull/10907>
- <https://github.com/vllm-project/vllm/pull/12416>
- <https://github.com/vllm-project/vllm/pull/12478>
- <https://github.com/vllm-project/vllm/pull/12516>
- <https://github.com/vllm-project/vllm/pull/12530>
- <https://github.com/vllm-project/vllm/pull/12561>
- <https://github.com/vllm-project/vllm/pull/12579>
- <https://github.com/vllm-project/vllm/pull/12592>
- <https://github.com/vllm-project/vllm/pull/12644>

### 指标收集

在 v1 中，我们希望将计算和开销移出引擎核心进程，以最小化每次前向传播之间的时间。

V1 EngineCore 设计的总体思路是：

- EngineCore 是内部循环。性能在此处最为关键
- AsyncLLM 是外部循环。这与 GPU 执行重叠（理想情况下），因此任何"开销"应尽可能放在此处。因此，AsyncLLM.output_handler_loop 是指标记账的理想位置（如果可能）。

我们将通过在前端 API 服务器中收集指标来实现这一点，并将这些指标基于我们从引擎核心进程返回到前端的 `EngineCoreOutputs` 中可以获取的信息。

### 间隔计算

我们的许多指标是请求处理过程中各种事件之间的时间间隔。最佳实践是使用基于"单调时钟"（`time.monotonic()`）而不是"挂钟时间"（`time.time()`）的时间戳来计算间隔，因为前者不受系统时钟变化（例如 NTP）的影响。

同样重要的是要注意，不同进程的单调时钟不同——每个进程有自己的参考点。因此，比较来自不同进程的单调时间戳是没有意义的。

因此，为了计算一个间隔，我们必须比较来自同一进程的两个单调时间戳。

### 调度器统计信息

引擎核心进程将收集一些来自调度器的关键统计信息——例如，上次调度传递后被调度或等待的请求数——并将这些统计信息包含在 `EngineCoreOutputs` 中。

### 引擎核心事件

引擎核心还将记录某些每个请求事件的时间戳，以便前端可以计算这些事件之间的间隔。

事件包括：

- `QUEUED` - 当请求被引擎核心接收并添加到调度器队列时。
- `SCHEDULED` - 当请求首次被调度执行时。
- `PREEMPTED` - 请求被放回等待队列，以便为其他请求的完成腾出空间。它将在未来被重新调度并重新开始其预填充阶段。
- `NEW_TOKENS` - 当 `EngineCoreOutput` 中包含的输出被生成时。由于这在给定迭代中对所有请求都是通用的，我们在 `EngineCoreOutputs` 上使用单个时间戳来记录此事件。

计算出的间隔包括：

- 队列间隔 - 在 `QUEUED` 和最近的 `SCHEDULED` 之间。
- 预填充间隔 - 在最近的 `SCHEDULED` 和随后的第一个 `NEW_TOKENS` 之间。
- 解码间隔 - 在第一个（最近的 `SCHEDULED` 之后）和最后一个 `NEW_TOKENS` 之间。
- 推理间隔 - 在最近的 `SCHEDULED` 和最后一个 `NEW_TOKENS` 之间。
- token 间间隔 - 在连续的 `NEW_TOKENS` 之间。

换句话说：

![间隔计算 - 常见情况](../assets/design/metrics/intervals-1.png)

我们探索了让前端使用前端可见的事件时间来计算这些间隔的可能性。然而，前端无法看到 `QUEUED` 和 `SCHEDULED` 事件的时序，并且由于我们需要基于同一进程的单调时间戳来计算间隔……我们需要引擎核心为所有这些事件记录时间戳。

#### 间隔计算 vs 抢占

当解码期间发生抢占时，由于任何已生成的 token 都会被重用，我们认为抢占影响了 token 间、解码和推理间隔。

![间隔计算 - 解码被抢占](../assets/design/metrics/intervals-2.png)

当预填充期间发生抢占时（假设此事件可能发生），我们认为抢占影响了首 token 延迟和预填充间隔。

![间隔计算 - 预填充被抢占](../assets/design/metrics/intervals-3.png)

### 前端统计信息收集

当前端处理单个 `EngineCoreOutputs`——即单个引擎核心迭代的输出——时，它收集与该迭代相关的各种统计信息：

- 此迭代中生成的新 token 总数。
- 在此迭代中完成的预填充所处理的 prompt token 总数。
- 在此迭代中被调度的任何请求的队列间隔。
- 在此迭代中完成预填充的任何请求的预填充间隔。
- 此迭代中包含的所有请求的 token 间间隔（每输出 token 时间，TPOT）。
- 在此迭代中完成预填充的任何请求的首 token 延迟（TTFT）。但是，我们相对于请求首次被前端接收的时间（`arrival_time`）计算此间隔，以计入输入处理时间。目前 `arrival_time` 在分词开始时开始。

对于在给定迭代中完成的任何请求，我们还记录：

- 推理和解码间隔——相对于调度和首 token 事件，如上所述。
- 端到端延迟——从前端 `arrival_time` 到前端收到最终 token 之间的间隔。

### KV 缓存驻留指标

我们还发出一组直方图，描述采样 KV 缓存块驻留的时间以及它们被重用的频率。采样（`--kv-cache-metrics-sample`）保持开销极小；当一个块被选中时，我们记录：

- `lifetime` – 分配 ⟶ 驱逐
- `idle before eviction` – 最后访问 ⟶ 驱逐
- `reuse gaps` – 块被重用时的访问间隔

这些直接映射到 Prometheus 指标：

- `vllm:kv_block_lifetime_seconds` – 每个采样块存在的时长。
- `vllm:kv_block_idle_before_evict_seconds` – 最终访问后的空闲时间。
- `vllm:kv_block_reuse_gap_seconds` – 连续访问之间的时间。

引擎核心仅通过 `SchedulerStats` 发送原始驱逐事件；前端消耗这些事件，将其转换为 Prometheus 观测值，并在启用日志记录时通过 `LLM.get_metrics()` 暴露相同的数据。在一个图表上查看生存时间和空闲时间，可以很容易地发现卡住的缓存或将 prompt 锁定在长解码上的工作负载。

### 指标发布 - 日志记录

`LoggingStatLogger` 指标发布器每 5 秒输出一条 `INFO` 级别日志消息，其中包含一些关键指标：

- 当前正在运行/等待的请求数
- 当前 GPU 缓存使用率
- 过去 5 秒内每秒处理的 prompt token 数
- 过去 5 秒内每秒生成的新 token 数
- 最近 1k 个 KV 缓存块查询的前缀缓存命中率

### 指标发布 - Prometheus

`PrometheusStatLogger` 指标发布器通过 `/metrics` HTTP 端点以 Prometheus 兼容格式暴露指标。然后可以配置 Prometheus 实例轮询此端点（例如每秒一次）并将值记录在其时间序列数据库中。Prometheus 通常通过 Grafana 使用，使这些指标可以随时间绘制成图表。

Prometheus 支持以下指标类型：

- Counter：一个随时间增加的值，永不减少，通常在 vLLM 实例重启时重置为零。例如，实例生命周期内生成的 token 数。
- Gauge：一个可以上下波动的值，例如当前计划执行的请求数。
- Histogram：指标样本的计数，记录在存储桶中。例如，TTFT 小于 1ms、5ms、10ms、20ms 等的请求数。

Prometheus 指标还可以带标签，允许根据匹配的标签组合指标。在 vLLM 中，我们为每个指标添加一个 `model_name` 标签，包括该实例服务的模型名称。

示例输出：

```bash
$ curl http://0.0.0.0:8000/metrics
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.0
...
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 27453.0
...
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
vllm:request_success_total{finished_reason="length",model_name="meta-llama/Llama-3.1-8B-Instruct"} 131.0
vllm:request_success_total{finished_reason="abort",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
...
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.001",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 13.0
vllm:time_to_first_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 97.0
vllm:time_to_first_token_seconds_bucket{le="0.06",model_name="meta-llama/Llama-3.1-8B-Instruct"} 123.0
vllm:time_to_first_token_seconds_bucket{le="0.08",model_name="meta-llama/Llama-3.1-8B-Instruct"} 138.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 140.0
vllm:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 140.0
```

!!! note
    选择对广大用户最有用的直方图存储桶并非易事，需要随着时间推移不断完善。

### 缓存配置信息

`prometheus_client` 支持 [Info 指标](https://prometheus.github.io/client_python/instrumenting/info/)，相当于一个值永久设置为 1 的 Gauge，但通过标签暴露有趣的键/值对信息。这用于不会更改的实例信息——因此只需在启动时观察一次——并允许在 Prometheus 中跨实例进行比较。

我们使用此概念来实现 `vllm:cache_config_info` 指标：

```text
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",calculate_kv_scales="False",cpu_offload_gb="0",enable_prefix_caching="False",gpu_memory_utilization="0.9",...} 1.0
```

然而，`prometheus_client` [从未在多进程模式下支持 Info 指标](https://github.com/prometheus/client_python/pull/300)——原因[不明](gh-pr:7279#discussion_r1710417152)。我们改用设置为 1 且使用 `multiprocess_mode="mostrecent"` 的 `Gauge` 指标。

### LoRA 指标

`vllm:lora_requests_info` `Gauge` 有些类似，只是其值是当前的挂钟时间，并且每次迭代都会更新。

使用的标签名称包括：

- `running_lora_adapters`：每个适配器使用该适配器的正在运行的请求数的计数，格式为逗号分隔的字符串。
- `waiting_lora_adapters`：类似，但计数等待调度的请求。
- `max_lora` - 静态的"单批次中最大 LoRA 数量"配置。

将多个适配器的运行/等待计数编码在逗号分隔的字符串中似乎相当不妥——我们可以使用标签来区分每个适配器的计数。这应该重新审视。

注意，使用了 `multiprocess_mode="livemostrecent"`——使用最新的指标，但仅来自当前正在运行的进程。

此功能在 <https://github.com/vllm-project/vllm/pull/9477> 中添加，并且[至少有一个已知用户](https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/54)。如果我们重新审视此设计并弃用旧指标，应与下游用户协调，以便他们可以在移除前迁移。

### 前缀缓存指标

<https://github.com/vllm-project/vllm/issues/10582> 中关于添加前缀缓存指标的讨论产生了一些有趣的观点，可能与我们处理未来指标的方式相关。

每次查询前缀缓存时，我们记录查询的 token 数和缓存中存在的查询 token 数（即命中数）。

然而，人们感兴趣的指标是命中率——即每次查询的命中数。

在日志记录的情况下，我们期望用户最好通过计算固定数量的最近查询（目前间隔固定为最近 1k 次查询）上的命中率来获得最佳服务。

在 Prometheus 的情况下，我们应该利用 Prometheus 的时间序列特性，允许用户在他们选择的间隔上计算命中率。例如，一个计算过去 5 分钟命中间隔的 PromQL 查询：

```text
rate(cache_query_hit[5m]) / rate(cache_query_total[5m])
```

为实现此目的，我们应该在 Prometheus 中将查询和命中记录为计数器，而不是将命中率记录为 Gauge。

## 已弃用的指标

### 如何弃用

弃用指标不应掉以轻心。用户可能不会注意到一个指标已被弃用，并且当它突然（从他们的角度）被移除时可能会感到不便，即使有等效指标可供使用。

例如，参见 `vllm:avg_prompt_throughput_toks_per_s` 如何被[弃用](https://github.com/vllm-project/vllm/pull/2764)（在代码中添加注释）、[移除](https://github.com/vllm-project/vllm/pull/12383)、然后被[用户注意到](https://github.com/vllm-project/vllm/issues/13218)。

一般来说：

1. 我们应该谨慎弃用指标，特别是因为很难预测用户影响。
2. 我们应该在包含在 `/metrics` 输出中的帮助字符串中放置显眼的弃用通知。
3. 我们应该在面向用户的文档和发布说明中列出已弃用的指标。
4. 我们应该考虑将已弃用的指标隐藏在 CLI 参数后面，以便在删除它们之前给管理员[提供一个逃生舱口](https://kubernetes.io/docs/concepts/cluster-administration/system-metrics/#show-hidden-metrics)，持续一段时间。

关于项目范围的弃用策略，请参见[弃用策略](../contributing/deprecation_policy.md)。

### 未实现 - `vllm:tokens_total`

由 <https://github.com/vllm-project/vllm/pull/4464> 添加，但显然从未实现。可以直接移除。

### 重复 - 队列时间

`vllm:time_in_queue_requests` Histogram 指标由 <https://github.com/vllm-project/vllm/pull/9659> 添加，其计算方式为：

```python
    self.metrics.first_scheduled_time = now
    self.metrics.time_in_queue = now - self.metrics.arrival_time
```

两周后，<https://github.com/vllm-project/vllm/pull/4464> 添加了 `vllm:request_queue_time_seconds`，导致：

```python
if seq_group.is_finished():
    if (seq_group.metrics.first_scheduled_time is not None and
            seq_group.metrics.first_token_time is not None):
        time_queue_requests.append(
            seq_group.metrics.first_scheduled_time -
            seq_group.metrics.arrival_time)
    ...
    if seq_group.metrics.time_in_queue is not None:
        time_in_queue_requests.append(
            seq_group.metrics.time_in_queue)
```

这似乎是重复的，应该移除其中一个。后者被 Grafana 仪表盘使用，因此我们应该弃用或移除前者。

### 前缀缓存命中率

参见上文——我们现在暴露的是"查询"和"命中"计数器，而不是"命中率"Gauge。

### KV 缓存卸载

两个旧版指标与 v1 中不再相关的"交换"抢占模式有关：

- `vllm:num_requests_swapped`
- `vllm:cpu_cache_usage_perc`

在此模式下，当请求被抢占时（例如为 KV 缓存中完成其他请求腾出空间），KV 缓存块被交换到 CPU 内存。`--swap-space` 标志已被移除，因为此功能在 V1 中不再使用。

历史上，[vLLM 长期以来支持束搜索](https://github.com/vllm-project/vllm/issues/6226)。SequenceGroup 封装了共享相同 prompt kv 块的 N 个序列的概念。这使得 KV 缓存块可以在请求之间共享，并通过写时复制实现分支。CPU 交换本意是用于此类束搜索场景。

后来，引入了前缀缓存的概念，允许 KV 缓存块隐式共享。这被证明是比 CPU 交换更好的选择，因为块可以按需缓慢驱逐，并且被驱逐的 prompt 部分可以重新计算。

SequenceGroup 在 V1 中被移除，尽管"并行采样"（`n>1`）需要一个替代品。[束搜索已移出核心](https://github.com/vllm-project/vllm/issues/8306)。对于一个非常不常见的功能，有大量复杂的代码。

在 V1 中，前缀缓存更好（零开销）且默认开启，抢占和重新计算策略应该工作得更好。

## 未来工作

### 并行采样

一些旧版指标仅在"并行采样"的背景下相关。这在请求中使用 `n` 参数从同一个 prompt 请求多个补全时发生。

作为在 <https://github.com/vllm-project/vllm/pull/10980> 中添加并行采样支持的一部分，我们也应添加这些指标。

- `vllm:request_params_n`（Histogram）

  观察每个已完成请求的 'n' 参数的值。

- `vllm:request_max_num_generation_tokens`（Histogram）

  观察每个已完成的序列组中所有序列的最大输出长度。在没有并行采样的情况下，这等同于 `vllm:request_generation_tokens`。

### 推测解码

一些旧版指标特定于"推测解码"。这是指我们使用更快的近似方法或模型生成候选 token，然后用较大模型验证这些 token。

- `vllm:spec_decode_draft_acceptance_rate`（Gauge）
- `vllm:spec_decode_efficiency`（Gauge）
- `vllm:spec_decode_num_accepted_tokens`（Counter）
- `vllm:spec_decode_num_draft_tokens`（Counter）
- `vllm:spec_decode_num_emitted_tokens`（Counter）

有一个正在审查的 PR（<https://github.com/vllm-project/vllm/pull/12193>），用于向 v1 添加"prompt lookup（ngram）"推测解码。其他技术将随之而来。我们应该在此背景下重新审视这些指标。

!!! note
    我们可能应该像处理前缀缓存命中率那样，将接受率作为单独的接受计数器和草案计数器来暴露。效率可能也需要类似的处理。

### 自动扩缩容和负载均衡

我们指标的一个常见用例是支持 vLLM 实例的自动扩缩容。

来自 [Kubernetes 服务工作组](https://github.com/kubernetes/community/tree/master/wg-serving)的相关讨论，请参见：

- [在 Kubernetes 中标准化大模型服务器指标](https://docs.google.com/document/d/1SpSp1E6moa4HSrJnS4x3NpLuj88sMXr2tbofKlzTZpk)
- [基准测试 LLM 工作负载以在 Kubernetes 中进行性能评估和自动扩缩容](https://docs.google.com/document/d/1k4Q4X14hW4vftElIuYGDu5KDe2LtV1XammoG-Xi3bbQ)
- [Inference Perf](https://github.com/kubernetes-sigs/wg-serving/tree/main/proposals/013-inference-perf)
- <https://github.com/vllm-project/vllm/issues/5041> 和 <https://github.com/vllm-project/vllm/pull/12726>。

这是一个非平凡的话题。请考虑 Rob 的评论：

> 我认为这个指标应侧重于尝试估计导致平均请求长度超过每秒查询数的最大并发数……因为这实际上是使服务器"饱和"的因素。

一个明确的目标是，我们应该暴露检测此饱和点所需的指标，以便管理员可以基于这些指标实现自动扩缩容规则。然而，为了做到这一点，我们需要清楚管理员（和自动监控系统）应如何判断实例是否接近饱和：

> 如何确定模型服务器计算资源的饱和点（即我们无法通过更高请求率获得更多吞吐量，但开始产生额外延迟的拐点），以便我们能够有效自动扩缩容？

### 指标命名

我们对指标命名的方法可能值得重新审视：

1. 在指标名称中使用冒号似乎与["冒号保留用于用户定义的记录规则"](https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels)相悖。
2. 我们的大多数指标遵循以单位结尾的约定，但并非全部。
3. 我们的一些指标名称以 `_total` 结尾：

    如果指标名称带有 `_total` 后缀，它将被移除。当暴露计数器的时间序列时，将添加 `_total` 后缀。这是为了 OpenMetrics 和 Prometheus 文本格式之间的兼容性，因为 OpenMetrics 要求 `_total` 后缀。

### 添加更多指标

新指标的想法层出不穷：

- 来自其他项目的示例，如 [TGI](https://github.com/IBM/text-generation-inference?tab=readme-ov-file#metrics)
- 来自特定用例的提议，如上述 Kubernetes 自动扩缩容主题
- 可能来自标准化工作的提议，如 [OpenTelemetry 语义约定 for Gen AI](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai)

在添加新指标时，我们应该谨慎。虽然指标通常相对容易添加：

1. 它们可能很难移除——参见上面的弃用部分。
2. 它们可能对启用时的性能产生显著影响。而且，指标通常只有在能在默认情况下和生产环境中启用时才非常有用。
3. 它们对项目的开发和维护有影响。随着时间推移添加的每个指标都使这项工作更加耗时，也许并非所有指标都值得这种持续的维护投入。

## 追踪 - OpenTelemetry

指标提供了系统性能和健康状况随时间的聚合视图。而追踪则是跟踪各个请求在不同服务和组件间的移动。两者都属于更一般的"可观测性"范畴。

vLLM 支持 OpenTelemetry 追踪：

- 由 <https://github.com/vllm-project/vllm/pull/4687> 添加，由 <https://github.com/vllm-project/vllm/pull/20372> 恢复
- 使用 `--oltp-traces-endpoint` 和 `--collect-detailed-traces` 配置
- [OpenTelemetry 博客文章](https://opentelemetry.io/blog/2024/llm-observability/)
- [面向用户的文档](../../examples/observability/opentelemetry/README.md)
- [博客文章](https://medium.com/@ronen.schaffer/follow-the-trail-supercharging-vllm-with-opentelemetry-distributed-tracing-aa655229b46f)
- [IBM 产品文档](https://www.ibm.com/docs/en/instana-observability/current?topic=mgaa-monitoring-large-language-models-llms-vllm-public-preview)

OpenTelemetry 有一个 [Gen AI 工作组](https://github.com/open-telemetry/community/blob/main/projects/gen-ai.md)。

由于指标本身就是一个足够大的主题，我们认为追踪的主题与指标相当独立。

### OpenTelemetry 模型前向 vs 执行时间

当前实现暴露了以下两个指标：

- `vllm:model_forward_time_milliseconds`（Histogram）- 当此请求在批次中时，模型前向传播所花费的时间。
- `vllm:model_execute_time_milliseconds`（Histogram）- 模型执行函数所花费的时间。这将包括模型前向、块/跨工作器同步、CPU-GPU 同步时间和采样时间。

这些指标仅在启用 OpenTelemetry 追踪并且使用 `--collect-detailed-traces=all/model/worker` 时启用。此选项的文档说明：

> 为指定模块收集详细追踪。这可能涉及使用可能昂贵和/或阻塞的操作，因此可能对性能产生影响。

这些指标由 <https://github.com/vllm-project/vllm/pull/7089> 添加，并在 OpenTelemetry 追踪中显示为：

```text
-> gen_ai.latency.time_in_scheduler: Double(0.017550230026245117)
-> gen_ai.latency.time_in_model_forward: Double(3.151565277099609)
-> gen_ai.latency.time_in_model_execute: Double(3.6468167304992676)
```

我们已经有了 `inference_time` 和 `decode_time` 指标，因此问题在于是否有足够常见的使用场景，值得为更高分辨率的时间信息承担开销。

由于我们将单独处理 OpenTelemetry 支持的问题，我们将把这些特定指标归入该主题下。
