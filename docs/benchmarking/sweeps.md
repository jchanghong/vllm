# 参数扫描

`vllm bench sweep` 是一套命令套件，旨在跨多种配置运行基准测试，并通过可视化结果进行比较。

## 在线基准测试

### 基本用法

`vllm bench sweep serve` 启动 `vllm serve`，并为每种服务配置迭代运行 `vllm bench serve`。

!!! tip
    如果你只需要对单一服务配置运行基准测试，请考虑使用 [GuideLLM](https://github.com/vllm-project/guidellm)，这是一个成熟的性能基准测试框架，提供实时进度更新和自动报告生成。它在数据集加载、请求格式和工作负载模式方面也比 `vllm bench serve` 更灵活。

按照以下步骤运行脚本：

1. 构建 `vllm serve` 的基础命令，并将其传递给 `--serve-cmd` 选项。
2. 构建 `vllm bench serve` 的基础命令，并将其传递给 `--bench-cmd` 选项。
3. （可选）如果你想改变 `vllm serve` 的设置，创建一个新的 JSON 文件，填入你想要测试的参数组合。将文件路径传递给 `--serve-params`。

    - 示例：调整 `--max-num-seqs` 和 `--max-num-batched-tokens`：

    ```json
    [
        {
            "max_num_seqs": 32,
            "max_num_batched_tokens": 1024
        },
        {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 1024
        },
        {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 2048
        },
        {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 2048
        },
        {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 4096
        },
        {
            "max_num_seqs": 256,
            "max_num_batched_tokens": 4096
        }
    ]
    ```

4. （可选）如果你想改变 `vllm bench serve` 的设置，创建一个新的 JSON 文件，填入你想要测试的参数组合。将文件路径传递给 `--bench-params`。

    - 示例：为随机数据集使用不同的输入/输出长度：

    ```json
    [
        {
            "_benchmark_name": "scenario_A",
            "random_input_len": 128,
            "random_output_len": 32
        },
        {
            "_benchmark_name": "scenario_B",
            "random_input_len": 256,
            "random_output_len": 64
        },
        {
            "_benchmark_name": "scenario_C",
            "random_input_len": 512,
            "random_output_len": 128
        }
    ]
    ```

5. 设置 `--output-dir` 并可选地设置 `--experiment-name` 来控制结果的保存位置。

示例命令：

```bash
vllm bench sweep serve \
    --serve-cmd 'vllm serve meta-llama/Llama-2-7b-chat-hf' \
    --bench-cmd 'vllm bench serve --model meta-llama/Llama-2-7b-chat-hf --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json' \
    --serve-params benchmarks/serve_hparams.json \
    --bench-params benchmarks/bench_hparams.json \
    --output-dir benchmarks/results \
    --experiment-name demo
```

默认情况下，每种参数组合会被基准测试 3 次，以使结果更可靠。你可以通过设置 `--num-runs` 来调整运行次数。

!!! important
    如果同时传递了 `--serve-params` 和 `--bench-params`，脚本将遍历它们之间的笛卡尔积。
    你可以使用 `--dry-run` 预览将要运行的命令。

    对于每个 `--serve-params`，我们只启动一次服务器，并在多个 `--bench-params` 期间保持其运行。
    每次基准测试运行之间，我们会调用所有 `/reset_*_cache` 端点以清空状态，为下一次运行做准备。
    如果你使用了自定义的 `--serve-cmd`，可以通过设置 `--after-bench-cmd` 覆盖重置状态所用的命令。

!!! note
    对于涉及许多变量的参数组合，你应该设置 `_benchmark_name` 以提供人类可读的名称。
    如果文件名因超出文件系统允许的最大路径长度，则此项变为必需。

!!! tip
    你可以使用 `--resume` 选项在发生意外错误时继续参数扫描，例如连接到 HF Hub 超时。

### 工作负载浏览器

`vllm bench sweep serve_workload` 是 `vllm bench sweep serve` 的一个变体，它探索不同的工作负载级别，以找到延迟和吞吐量之间的权衡。结果也可以[可视化](#可视化)以确定可行的 SLA。

工作负载可以用请求率或并发数来表示（使用 `--workload-var` 选择）。

示例命令：

```bash
vllm bench sweep serve_workload \
    --serve-cmd 'vllm serve meta-llama/Llama-2-7b-chat-hf' \
    --bench-cmd 'vllm bench serve --model meta-llama/Llama-2-7b-chat-hf --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 100' \
    --workload-var max_concurrency \
    --serve-params benchmarks/serve_hparams.json \
    --bench-params benchmarks/bench_hparams.json \
    --num-runs 1 \
    --output-dir benchmarks/results \
    --experiment-name demo
```

探索不同工作负载级别的算法可总结如下：

1. 逐条发送请求运行基准测试（串行推理，最低工作负载）。这会产生最低的延迟和吞吐量。
2. 一次性发送所有请求运行基准测试（批量推理，最高工作负载）。这会产生最高的延迟和吞吐量。
3. 估算与步骤 2 对应的 `workload_var` 值。
4. 使用剩余迭代次数在 `workload_var` 的中间值上均匀运行基准测试。

你可以通过设置 `--workload-iters` 来覆盖算法中的迭代次数。

!!! tip
    这是我们实现中与 [GuideLLM 的 `--profile sweep`](https://github.com/vllm-project/guidellm/blob/v0.5.3/src/guidellm/benchmark/profiles.py#L575) 等效的功能。

    通常，`--workload-var max_concurrency` 产生更可靠的结果，因为它直接控制施加在 vLLM 引擎上的工作负载。
    尽管如此，我们默认使用 `--workload-var request_rate` 以保持与 GuideLLM 类似的行为。

## 启动基准测试

`vllm bench sweep startup` 跨参数组合运行 `vllm bench startup`，以比较不同引擎设置下的冷/热启动时间。

按照以下步骤运行脚本：

1. （可选）构建 `vllm bench startup` 的基础命令，并将其传递给 `--startup-cmd`（默认值：`vllm bench startup`）。
2. （可选）复用 `vllm bench sweep serve` 的 `--serve-params` JSON 来变更引擎设置。仅应用 `vllm bench startup` 支持的参数。
3. （可选）创建 `--startup-params` JSON 来变更启动特定选项，如迭代次数。
4. 确定结果的保存位置，并将其传递给 `--output-dir`。

示例 `--serve-params`：

```json
[
    {
        "_benchmark_name": "tp1",
        "model": "Qwen/Qwen3-0.6B",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9
    },
    {
        "_benchmark_name": "tp2",
        "model": "Qwen/Qwen3-0.6B",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.9
    }
]
```

示例 `--startup-params`：

```json
[
    {
        "_benchmark_name": "qwen3-0.6",
        "num_iters_cold": 2,
        "num_iters_warmup": 1,
        "num_iters_warm": 2
    }
]
```

示例命令：

```bash
vllm bench sweep startup \
    --startup-cmd 'vllm bench startup --model Qwen/Qwen3-0.6B' \
    --serve-params benchmarks/serve_hparams.json \
    --startup-params benchmarks/startup_hparams.json \
    --output-dir benchmarks/results \
    --experiment-name demo
```

!!! important
    默认情况下，`--serve-params` 或 `--startup-params` 中不支持的参数会被忽略并给出警告。
    使用 `--strict-params` 可在遇到未知键时立即失败。

## 可视化

### 基本用法

`vllm bench sweep plot` 可用于绘制参数扫描结果的性能曲线。

通过 `--var-x` 和 `--var-y` 控制要绘制的变量，并可选择性地对值应用 `--filter-by` 和 `--bin-by`。图表按照 `--fig-by`、`--row-by`、`--col-by` 和 `--curve-by` 进行组织。

可视化[工作负载浏览器](#工作负载浏览器)结果的示例命令：

```bash
EXPERIMENT_DIR=${1:-"benchmarks/results/demo"}

# 延迟随工作负载增加而增加
vllm bench sweep plot $EXPERIMENT_DIR \
    --var-x max_concurrency \
    --var-y median_ttft_ms \
    --col-by _benchmark_name \
    --curve-by max_num_seqs,max_num_batched_tokens \
    --fig-name latency_curve

# 吞吐量随工作负载增加而趋于饱和
vllm bench sweep plot $EXPERIMENT_DIR \
    --var-x max_concurrency \
    --var-y total_token_throughput \
    --col-by _benchmark_name \
    --curve-by max_num_seqs,max_num_batched_tokens \
    --fig-name throughput_curve

# 延迟和吞吐量之间的权衡
vllm bench sweep plot $EXPERIMENT_DIR \
    --var-x total_token_throughput \
    --var-y median_ttft_ms \
    --col-by _benchmark_name \
    --curve-by max_num_seqs,max_num_batched_tokens \
    --fig-name latency_throughput
```

!!! tip
    你可以使用 `--dry-run` 预览将要绘制的图表。

### 帕累托图

`vllm bench sweep plot_pareto` 帮助选择在每用户和每 GPU 吞吐量之间取得平衡的配置。

更高的并发数或批大小可以提高 GPU 效率（每 GPU），但会增加每用户延迟；较低的并发数可以提高每用户速率，但会使 GPU 利用不足。帕累托前沿显示了你的运行中最佳的可实现配对。

- x 轴：tokens/s/用户 = `output_throughput` ÷ concurrency（`--user-count-var`，默认 `max_concurrency`，回退 `max_concurrent_requests`）。
- y 轴：tokens/s/GPU = `output_throughput` ÷ GPU 数量（如果设置了 `--gpu-count-var`；否则 `gpu_count` 为 TP×PP*DP）。
- 输出：在 `OUTPUT_DIR/pareto/PARETO.png` 的单个图表。
- 显示每个数据点中使用的配置 `--label-by`（默认值：`max_concurrency,gpu_count`）。

示例：

```bash
EXPERIMENT_DIR=${1:-"benchmarks/results/demo"}

vllm bench sweep plot_pareto $EXPERIMENT_DIR \
  --label-by max_concurrency,tensor_parallel_size,pipeline_parallel_size
```

!!! tip
    你可以使用 `--dry-run` 预览将要绘制的图表。
