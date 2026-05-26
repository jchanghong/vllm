# vLLM 基准测试套件

## 介绍

此目录包含一个基准测试套件，供**开发人员**在本地运行，以判断其 PR 是提升还是降低了 vLLM 的性能。
vLLM 还在 [perf.vllm.ai](https://perf.vllm.ai/) 上维护了持续性能基准测试，托管在 PyTorch CI HUD 下。

## 性能基准测试快速概览

**测试覆盖范围**：在 B200、A100、H100、Intel® Xeon® 处理器、Intel® Gaudi® 3 加速器和 Arm® Neoverse™ 上，使用不同模型进行延迟、吞吐量和固定 QPS 服务测试。

**测试持续时间**：约 1 小时。

**对基准测试开发者的建议**：请尽量将基准测试的持续时间控制在约 1 小时左右，以免运行时间过长。

## 触发基准测试

基准测试需要手动触发：

```bash
bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

运行时环境变量：

- `ON_CPU`：在 Intel® Xeon® 和 Arm® Neoverse™ 处理器上设置为 '1'。默认值为 0。
- `SERVING_JSON`：用于服务测试的 JSON 文件。默认值为空字符串（使用默认文件）。
- `LATENCY_JSON`：用于延迟测试的 JSON 文件。默认值为空字符串（使用默认文件）。
- `THROUGHPUT_JSON`：用于吞吐量测试的 JSON 文件。默认值为空字符串（使用默认文件）。
- `REMOTE_HOST`：要测试的远程 vLLM 服务的 IP 地址。默认值为空字符串。
- `REMOTE_PORT`：要测试的远程 vLLM 服务的端口。默认值为空字符串。

## 性能基准测试详情

详细描述请参见 [performance-benchmarks-descriptions.md](performance-benchmarks-descriptions.md)，并使用 `tests/latency-tests.json`、`tests/throughput-tests.json`、`tests/serving-tests.json` 配置测试用例。
> 注意：对于 Intel® Xeon® 处理器，请使用 `tests/latency-tests-cpu.json`、`tests/throughput-tests-cpu.json`、`tests/serving-tests-cpu.json`。
> 对于 Intel® Gaudi® 3 加速器，请使用 `tests/latency-tests-hpu.json`、`tests/throughput-tests-hpu.json`、`tests/serving-tests-hpu.json`。
> 对于 Arm® Neoverse™，请使用 `tests/latency-tests-arm64-cpu.json`、`tests/throughput-tests-arm64-cpu.json`、`tests/serving-tests-arm64-cpu.json`。

### 延迟测试

以下是 `latency-tests.json` 中的一个测试示例：

```json
[
    {
        "test_name": "latency_llama8B_tp1",
        "parameters": {
            "model": "meta-llama/Meta-Llama-3-8B",
            "tensor_parallel_size": 1,
            "load_format": "dummy",
            "num_iters_warmup": 5,
            "num_iters": 15
        }
    },
]
```

在此示例中：

- `test_name` 属性是测试的唯一标识符。在 `latency-tests.json` 中，必须以 `latency_` 开头。
- `parameters` 属性控制用于 `vllm bench latency` 的命令行参数。请注意，指定命令行参数时应使用下划线 `_` 而非连字符 `-`，`run-performance-benchmarks.sh` 会将下划线转换为连字符后再传递给 `vllm bench latency`。例如，`vllm bench latency` 对应的命令行参数将是 `--model meta-llama/Meta-Llama-3-8B --tensor-parallel-size 1 --load-format dummy --num-iters-warmup 5 --num-iters 15`

需要注意的是，性能数值对参数值非常敏感。请确保参数设置正确。

警告：基准测试脚本会自行保存 JSON 结果，因此请不要在 JSON 文件中配置 `--output-json` 参数。

### 吞吐量测试

测试在 `throughput-tests.json` 中指定。语法与 `latency-tests.json` 类似，不同之处在于参数将传递给 `vllm bench throughput`。

此测试的数值也较为敏感——该数值的微小变化可能会大幅改变性能结果。

### 服务测试

我们使用 `vllm bench serve` 以请求率 = inf 来测试吞吐量，以覆盖在线服务开销。相应的参数在 `serving-tests.json` 中，以下是一个示例：

```json
[
    {
        "test_name": "serving_llama8B_tp1_sharegpt",
        "qps_list": [1, 4, 16, "inf"],
        "server_parameters": {
            "model": "meta-llama/Meta-Llama-3-8B",
            "tensor_parallel_size": 1,
            "disable_log_stats": "",
            "load_format": "dummy"
        },
        "client_parameters": {
            "model": "meta-llama/Meta-Llama-3-8B",
            "backend": "vllm",
            "dataset_name": "sharegpt",
            "dataset_path": "./ShareGPT_V3_unfiltered_cleaned_split.json",
            "num_prompts": 200
        }
    },
]
```

在此示例中：

- `test_name` 属性也是测试的唯一标识符。必须以 `serving_` 开头。
- `server-parameters` 包括 vLLM 服务器的命令行参数。
- `client-parameters` 包括 `vllm bench serve` 的命令行参数。
- `qps_list` 控制测试的 QPS 列表。它将用于配置 `vllm bench serve` 中的 `--request-rate` 参数

此测试的数值相比延迟和吞吐量基准测试稳定性稍差（由于 `benchmark_serving.py` 内部的随机 sharegpt 数据集采样），但该数值的大幅变化（例如 5% 的变化）仍然会显著改变输出结果。

警告：基准测试脚本会自行保存 JSON 结果，因此请不要在 `serving-tests.json` 中配置 `--save-results` 或其他与保存结果相关的参数。

#### 默认参数字段

我们可以在键为 `defaults` 的 JSON 字段中指定默认参数。在该字段中定义的参数会全局应用于所有服务测试，并可在测试用例字段中被覆盖。以下是一个示例：

<details>
<summary> 默认参数字段示例 </summary>

```json
{
  "defaults": {
    "qps_list": [
      "inf"
    ],
    "server_environment_variables": {
      "VLLM_ALLOW_LONG_MAX_MODEL_LEN": 1
    },
    "server_parameters": {
      "tensor_parallel_size": 1,
      "dtype": "bfloat16",
      "block_size": 128,
      "disable_log_stats": "",
      "load_format": "dummy"
    },
    "client_parameters": {
      "backend": "vllm",
      "dataset_name": "random",
      "random-input-len": 128,
      "random-output-len": 128,
      "num_prompts": 200,
      "ignore-eos": ""
    }
  },
  "tests": [
    {
      "test_name": "serving_llama3B_tp2_random_128_128",
      "server_parameters": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "tensor_parallel_size": 2,
      },
      "client_parameters": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
      }
    },
    {
      "test_name": "serving_qwen3_tp4_random_128_128",
      "server_parameters": {
        "model": "Qwen/Qwen3-14B",
        "tensor_parallel_size": 4,
      },
      "client_parameters": {
        "model": "Qwen/Qwen3-14B",
      }
    },
  ]
}
```

</details>

### 可视化结果

`convert-results-json-to-markdown.py` 通过使用真实基准测试结果格式化 [descriptions.md](performance-benchmarks-descriptions.md)，帮助您将基准测试结果放入 Markdown 表格中。
您可以在 `buildkite/performance-benchmark` 任务页面中找到以表格形式呈现的结果。
如果您没有看到表格，请等待基准测试运行完成。
表格的 JSON 版本（以及基准测试的 JSON 版本）也将附加到 Markdown 文件中。
原始基准测试结果（以 JSON 文件格式）位于基准测试的 `Artifacts` 选项卡中。

#### 性能结果对比

按照[性能结果对比](https://docs.vllm.ai/en/latest/benchmarking/dashboard/#performance-results-comparison)中的说明分析性能结果和规模调整指南。
