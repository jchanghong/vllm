# 性能仪表盘

性能仪表盘用于确认新变更是否在各种工作负载下提升/降低了性能。
它通过在每次提交时（同时带有 `perf-benchmarks` 和 `ready` 标签）触发基准测试运行来更新，并且在 PR 合并到 vLLM 时也会触发。

结果会自动发布到公开的 [vLLM 性能仪表盘](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm)。

## 手动触发基准测试

使用 [vllm-ci-test-repo 镜像](https://gallery.ecr.aws/q9t5s3a7/vllm-ci-test-repo) 配合 vLLM 基准测试套件。
对于 x86 CPU 环境，请使用带有 "-cpu" 后缀的镜像。对于 AArch64 CPU 环境，请使用带有 "-arm64-cpu" 后缀的镜像。

以下是 CPU 的 docker 运行命令示例。对于 GPU，不需要设置 `ON_CPU` 环境变量。

```bash
export VLLM_COMMIT=7f42dc20bb2800d09faa72b26f25d54e26f1b694 # 使用主分支的完整 commit hash
export HF_TOKEN=<有效的 Hugging Face token>
if [[ "$(uname -m)" == aarch64 || "$(uname -m)" == arm64 ]]; then
  IMG_SUFFIX="arm64-cpu"
else
  IMG_SUFFIX="cpu"
fi
docker run -it --entrypoint /bin/bash -v /data/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF_TOKEN -e ON_CPU=1 --shm-size=16g --name vllm-cpu-ci public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:${VLLM_COMMIT}-${IMG_SUFFIX}
```

然后，在 docker 实例内运行以下命令。

```bash
bash .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
```

运行时，基准测试脚本会在 **benchmark/results** 文件夹下生成结果，以及 benchmark_results.md 和 benchmark_results.json。

### 运行时环境变量

- `ON_CPU`：在 Intel® Xeon® 和 Arm® Neoverse™ 处理器上设置为 '1'。默认值为 0。
- `SERVING_JSON`：用于服务测试的 JSON 文件。默认值为空字符串（使用默认文件）。
- `LATENCY_JSON`：用于延迟测试的 JSON 文件。默认值为空字符串（使用默认文件）。
- `THROUGHPUT_JSON`：用于吞吐量测试的 JSON 文件。默认值为空字符串（使用默认文件）。
- `REMOTE_HOST`：要基准测试的远程 vLLM 服务的 IP。默认值为空字符串。
- `REMOTE_PORT`：要基准测试的远程 vLLM 服务的端口。默认值为空字符串。
- `PROMPTS_PER_CONCURRENCY`：用于计算服务测试的 `num_prompts` 的乘数（`num_prompts = max_concurrency × 值`）。覆盖 JSON 中的 `num_prompts`。默认值为 NULL。
- `ENABLE_ADAPTIVE_CONCURRENCY`：设置为 '1' 以启用基于 SLA 的自适应并发搜索（在静态服务 max_concurrency 扫描之后）。默认值为 0。
- `SLA_TTFT_MS`：自适应并发搜索的默认 TTFT SLA 阈值（毫秒）。默认值为 3000。
- `SLA_TPOT_MS`：自适应并发搜索的默认 TPOT SLA 阈值（毫秒）。默认值为 100。
- `ADAPTIVE_MAX_PROBES`：最大自适应搜索额外探测次数。默认值为 8。
- `ADAPTIVE_MAX_CONCURRENCY`：自适应搜索期间允许的最大并发数。默认值为 1024。

### 可视化

`convert-results-json-to-markdown.py` 帮助你将基准测试结果放入带有真实基准测试结果的 markdown 表格中。
你可以在 `buildkite/performance-benchmark` 任务页面中找到以表格形式呈现的结果。
如果看不到表格，请等待基准测试完成运行。
表格的 JSON 版本（以及基准测试的 JSON 版本）也会附加到 markdown 文件中。
原始基准测试结果（JSON 文件格式）位于基准测试的 `Artifacts` 标签页中。

#### 性能结果比较

`compare-json-results.py` 用于比较使用 `convert-results-json-to-markdown.py` 转换的基准测试结果 JSON 文件。
运行时，基准测试脚本会在 `benchmark/results` 文件夹下生成结果，以及 `benchmark_results.md` 和 `benchmark_results.json`。
`compare-json-results.py` 比较两个 `benchmark_results.json` 文件，并提供性能比率，例如输出吞吐量、中位数 TTFT 和中位数 TPOT。
如果只传入一个 benchmark_results.json，则 `compare-json-results.py` 会改为比较 benchmark_results.json 中不同的 TP 和 PP 配置。

以下是使用该脚本比较 result_a 和 result_b（相同模型、数据集名称、输入/输出长度下的最大并发和 qps）的示例：
`python3 compare-json-results.py -f results_a/benchmark_results.json -f results_b/benchmark_results.json`

***输出吞吐量 (tok/s) — 模型 : [ meta-llama/Llama-3.1-8B-Instruct ] , 数据集名称 : [ random ] , 输入长度 : [ 2048.0 ] , 输出长度 : [ 2048.0 ]***

| | # 最大并发数 | qps | results_a/benchmark_results.json | results_b/benchmark_results.json | 性能比率 |
| | -------------------- | --- | -------------------------------- | -------------------------------- | ---------- |
| 0 | 12 | inf | 24.98 | 186.03 |  7.45 |
| 1 | 16 | inf |  25.49 | 246.92 | 9.69 |
| 2 | 24 | inf | 27.74 | 293.34 |  10.57 |
| 3 | 32 | inf | 28.61 |306.69 | 10.72 |

***compare-json-results.py – 命令行参数***

`compare-json-results.py` 提供可配置的参数，用于比较一个或多个 `benchmark_results.json` 文件，并生成汇总表格和图表。
在大多数情况下，用户只需指定 `--file` 来解析所需的基准测试结果。

| 参数                  | 类型               | 默认值                  | 描述                                                                                             |
| ---------------------- | ------------------ | ----------------------- | ------------------------------------------------------------------------------------------------- |
| `--file`               | `str` (可追加)      | *无*                    | 输入的 JSON 结果文件。可多次指定以比较多个基准测试输出。                                          |
| `--debug`              | `bool`             | `False`                 | 启用调试模式。设置后，打印所有可用信息以帮助故障排除和验证。                                      |
| `--plot` / `--no-plot` | `bool`             | `True`                  | 控制是否生成性能图表。使用 `--no-plot` 禁用图表生成。                                             |
| `--xaxis`              | `str`              | `# of max concurrency.` | 比较图表中用作 X 轴的列名（例如，并发数或批大小）。                                               |
| `--latency`            | `str`              | `p99`                   | TTFT/TPOT 使用的延迟聚合方法。支持的值：`median` 或 `p99`。                                      |
| `--ttft-max-ms`        | `float`            | `3000.0`                | TTFT 图表的上限参考值（毫秒），通常用于可视化 SLA 阈值。                                          |
| `--tpot-max-ms`        | `float`            | `100.0`                 | TPOT 图表的上限参考值（毫秒），通常用于可视化 SLA 阈值。                                          |

***有效最大并发数摘要***

根据配置的 TTFT 和 TPOT SLA 阈值，compare-json-results.py 会计算每个基准测试结果的有效最大并发数。
“最大并发数（两者）”列表示同时满足 TTFT 和 TPOT 约束的最高并发级别。
该值通常用于容量规划和规模指南。

| # | 配置          | 最大并发数 (TTFT ≤ 10000 ms) | 最大并发数 (TPOT ≤ 100 ms) | 最大并发数 (两者) | 两者下的输出吞吐量 (tok/s) | 两者下的 TTFT (ms) | 两者下的 TPOT (ms) |
| - | -------------- | ------------------------------------------- | ----------------------------------------- | -------------------------------- | -------------------------- | ---------------- | ---------------- |
| 0 | results-a      | 128.00                                      | 12.00                                     | 12.00                            | 127.76                     | 3000.82          | 93.24            |
| 1 | results-b      | 128.00                                      | 32.00                                     | 32.00                            | 371.42                     | 2261.53          | 81.74            |

关于性能基准测试及其参数的更多信息，请参阅 [Benchmark README](https://github.com/intel-ai-tce/vllm/blob/more_cpu_models/.buildkite/nightly-benchmarks/README.md) 和 [性能基准测试描述](../../.buildkite/performance-benchmarks/performance-benchmarks-descriptions.md)。

## 持续基准测试

持续基准测试为 vLLM 提供跨不同模型和 GPU 设备的自动化性能监控。这有助于追踪 vLLM 随时间的性能特征，并识别任何性能回退或改进。

### 工作原理

持续基准测试通过 PyTorch 基础设施仓库中的 [GitHub workflow CI](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-benchmark.yml) 触发，每 4 小时自动运行一次。该工作流执行三种类型的性能测试：

- **服务测试**：衡量请求处理和 API 性能
- **吞吐量测试**：评估 token 生成速率
- **延迟测试**：评估响应时间特征

### 基准测试配置

基准测试目前运行在 [vllm-benchmarks 目录](https://github.com/pytorch/pytorch-integration-testing/tree/main/vllm-benchmarks/benchmarks) 中配置的一组预定义模型上。要添加新的模型进行基准测试：

1. 导航到基准测试配置中相应的 GPU 目录
2. 将你的模型规格添加到相应的配置文件中
3. 新模型将包含在下一次计划好的基准测试运行中
