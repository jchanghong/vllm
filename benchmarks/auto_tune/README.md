# 自动化 vLLM 服务器参数调优

此脚本自动寻找最优的服务器参数组合（`max-num-seqs` 和 `max-num-batched-tokens`），以最大化 vLLM 服务器的吞吐量。它还支持额外的约束条件，如端到端延迟和前缀缓存命中率。

## 目录

- [前提条件](#前提条件)
- [配置](#配置)
- [如何运行](#如何运行)
- [示例用例](#示例用例)
- [输出](#输出)
- [工作原理](#工作原理)

## 前提条件

在运行脚本之前，请确保完成以下步骤：

1. **克隆 vLLM 并设置分支**：克隆 vLLM 仓库并切换到您的目标分支。

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
# git checkout <your-branch>
```

1. **安装环境**：安装或更新正确的运行环境。对于 TPU 使用，激活您的 `conda` 环境并安装相应的 `torch` 和 `torch_xla` 版本。

2. **模型配置**：如果您使用自定义模型，请确保其配置文件已正确放置并可访问。

## 配置

在运行脚本之前，必须在脚本顶部设置以下变量。

   注意：您也可以在运行脚本时通过环境变量覆盖以下默认值。

```bash
MODEL=meta-llama/Llama-3.3-70B-Instruct SYSTEM=TPU TP=8 DOWNLOAD_DIR='' INPUT_LEN=128 OUTPUT_LEN=2048 MAX_MODEL_LEN=2300 MIN_CACHE_HIT_PCT=0 MAX_LATENCY_ALLOWED_MS=100000000000 NUM_SEQS_LIST="128 256" NUM_BATCHED_TOKENS_LIST="1024 2048 4096" VLLM_LOGGING_LEVEL=DEBUG bash auto_tune.sh
```

| 变量 | 描述 | 示例值 |
| --- | --- | --- |
| `BASE` | **必需。** vLLM 仓库目录父目录的绝对路径。 | `"$HOME"` |
| `MODEL` | **必需。** vllm 服务的 Hugging Face 模型标识符。 | `"meta-llama/Llama-3.1-8B-Instruct"` |
| `SYSTEM` | **必需。** 您运行的硬件。选项：`TPU` 或 `GPU`。（对于其他系统，可能不支持保存 profile） | `"TPU"` |
| `TP` | **必需。** 张量并行度。 | `1` |
| `DOWNLOAD_DIR` | **必需。** 下载和加载模型权重的目录。 | `""`（默认下载路径） |
| `INPUT_LEN` | **必需。** 请求输入长度。 | `4000` |
| `OUTPUT_LEN` | **必需。** 请求输出长度。 | `16` |
| `MAX_MODEL_LEN` | **必需。** 最大模型长度。 | `4096` |
| `MIN_CACHE_HIT_PCT` | 前缀缓存命中率百分比（0-100）。设置为 `0` 以禁用。 | `60` |
| `MAX_LATENCY_ALLOWED_MS` | 允许的最大 P99 端到端延迟（毫秒）。设置为非常大的数（例如 `100000000000`）以忽略延迟约束。 | `500` |
| `NUM_SEQS_LIST` | 要测试的 `max-num-seqs` 值的空格分隔字符串。 | `"128 256"` |
| `NUM_BATCHED_TOKENS_LIST` | 要测试的 `max-num-batched-tokens` 值的空格分隔字符串。 | `"1024 2048 4096"` |

**注意**：默认的 `NUM_SEQS_LIST` 和 `NUM_BATCHED_TOKENS_LIST` 是针对中等大小输入/输出设置的。对于非常短的上下文（例如 20 个输入 token、20 个输出 token），您可能需要测试更大的 `max-num-seqs` 值。

## 如何运行

1. **配置**：编辑脚本并在[配置](#配置)部分设置变量。
2. **执行**：运行脚本。由于该过程可能需要很长时间，强烈建议使用终端复用器（如 `tmux` 或 `screen`）以防止连接断开时脚本停止。

```bash
cd <脚本所在目录>
bash auto_tune.sh
```

    请注意，`bash auto_tune.sh` 命令的完整或部分路径不能包含关键字 `vllm`，否则 `pkill -f vllm` 命令也会终止此脚本本身。

## 示例用例

以下是一些如何为不同目标配置脚本的示例：

### 1. 最大化吞吐量（无延迟约束）

- **目标**：找到最佳的 `max-num-seqs` 和 `max-num-batched-tokens`，以获得 1800 个输入 token 和 20 个输出 token 的最高可能吞吐量。
- **配置**：

```bash
INPUT_LEN=1800
OUTPUT_LEN=20
MAX_MODEL_LEN=2048
MIN_CACHE_HIT_PCT=0
MAX_LATENCY_ALLOWED_MS=100000000000 # 一个非常大的数
```

### 2. 在延迟要求下最大化吞吐量

- **目标**：在 P99 端到端延迟必须低于 500ms 的情况下，找到最佳服务器参数。
- **配置**：

```bash
INPUT_LEN=1800
OUTPUT_LEN=20
MAX_MODEL_LEN=2048
MIN_CACHE_HIT_PCT=0
MAX_LATENCY_ALLOWED_MS=500
```

### 3. 在带前缀缓存和延迟要求下最大化吞吐量

- **目标**：假设前缀缓存命中率为 60% 且延迟要求为 500ms，找到最佳服务器参数。
- **配置**：

```bash
INPUT_LEN=1800
OUTPUT_LEN=20
MAX_MODEL_LEN=2048
MIN_CACHE_HIT_PCT=60
MAX_LATENCY_ALLOWED_MS=500
```

## 输出

脚本运行完成后，您可以在 `$BASE/auto-benchmark/` 内新建的带时间戳目录中找到结果。

- **日志文件**：该目录（`$BASE/auto-benchmark/YYYY_MM_DD_HH_MM/`）包含每次运行的详细日志：
    - `vllm_log_...txt`：vLLM 服务器在每个参数组合下的日志输出。
    - `bm_log_...txt`：每次基准测试运行的 `vllm bench serve` 命令的日志输出。

- **最终结果摘要**：在日志目录中创建一个名为 `result.txt` 的文件。它包含每个测试组合的摘要，并以找到的全局最佳参数作为结论。

```text
# result.txt 内容示例
hash:a1b2c3d4...
max_num_seqs: 128, max_num_batched_tokens: 2048, request_rate: 10.0, e2el: 450.5, throughput: 9.8, goodput: 9.8
max_num_seqs: 128, max_num_batched_tokens: 4096 不满足延迟要求 500
...
best_max_num_seqs: 256, best_num_batched_tokens: 2048, best_throughput: 12.5, profile saved in: /home/user/vllm/auto-benchmark/2024_08_01_10_30/profile
```

  如果无法找到最佳参数，最后一行将是 `best_max_num_seqs: 0, best_num_batched_tokens: 0, best_throughput: 0`。这可能是由于服务器未正常启动，或延迟要求过于严格导致的。

- **Profiler 追踪**：在日志目录内创建一个名为 `profile` 的目录。它包含来自最佳性能运行的 profiler 追踪文件（例如 TPU 的 `.xplane.pb` 或 GPU 的 `.json` 追踪）。

## 工作原理

脚本按照系统化的流程寻找最优参数：

1. **查找最大 GPU 内存利用率**：脚本首先确定最高安全的 `gpu-memory-utilization`（从 0.98 开始递减），该值在启动服务器时不会导致内存不足（OOM）错误。这确保了基准测试运行在不崩溃的情况下使用最大可用内存。

2. **迭代和基准测试**：然后进入嵌套循环，遍历配置列表中提供的 `max-num-seqs` 和 `max-num-batched-tokens` 的每种组合。

3. **延迟感知的吞吐量搜索**：对于每个参数组合：
    - 启动 vLLM 服务器。
    - 首先使用无限请求率（`--request-rate inf`）运行基准测试。
    - 如果得到的 P99 端到端延迟在 `MAX_LATENCY_ALLOWED_MS` 限制内，则此吞吐量被视为该配置的最大值。
    - 如果延迟过高，脚本会通过迭代递减请求率直到满足延迟约束来进行搜索。这为给定参数和延迟要求找到了最高的可持续吞吐量。

4. **跟踪最佳结果**：在整个过程中，脚本跟踪迄今为止产生最高有效吞吐量的参数组合。

5. **Profile 收集**：对于性能最佳的运行，脚本保存 vLLM profiler 输出，可用于使用 TensorBoard 等工具进行深入性能分析。

## 批量自动调优

`batch_auto_tune.sh` 脚本允许您通过单个配置文件依次运行多个 `auto_tune.sh` 实验。它遍历一组参数集，为每组执行 `auto_tune.sh`，并将结果记录回输入文件。

### 前提条件

- **jq**：此脚本需要 `jq` 来解析 JSON 配置文件。
- **gcloud**：如果您计划将结果上传到 Google Cloud Storage，则必须安装并认证 `gcloud` CLI。

### 如何运行

1. **创建 JSON 配置文件**：创建一个文件（例如 `runs_config.json`），其中包含 JSON 对象数组。每个对象定义单个 `auto_tune.sh` 运行的参数。

2. **执行脚本**：

    ```bash
    bash batch_auto_tune.sh <path_to_json_file> [gcs_upload_path]
    ```

    - `<path_to_json_file>`：**必需。** JSON 配置文件的路径。
    - `[gcs_upload_path]`：**可选。** GCS 路径（例如 `gs://my-bucket/benchmark-results`），每次运行的详细结果和 profile 将上传到此路径。如果此参数为空，结果将在本地文件系统上可用（请参见日志中的 `RESULT_FILE=/path/to/results/file.txt`）。

### 配置文件

JSON 配置文件应包含一个对象数组。每个对象的键对应于 `auto_tune.sh` 的配置变量（请参见上方的[配置表](#配置)）。这些键将在每次运行时转换为大写的环境变量。

以下是一个包含两个基准测试配置的 `runs_config.json` 示例：

```json
[
  {
    "base": "/home/user",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "system": "TPU", # 或 GPU
    "tp": 8,
    "input_len": 128,
    "output_len": 2048,
    "max_model_len": 2300,
    "num_seqs_list": "128 256",
    "num_batched_tokens_list": "8192 16384"
  },
  {
    "base": "/home/user",
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "system": "TPU", # 或 GPU
    "tp": 8,
    "input_len": 4000,
    "output_len": 16,
    "max_model_len": 4096,
    "num_seqs_list": "64 128",
    "num_batched_tokens_list": "4096 8192",
    "max_latency_allowed_ms": 500
  }
]
```

### 输出

脚本会原地修改输入的 JSON 文件，将每次运行的结果添加到相应的对象中。添加的字段如下：

- `run_id`：运行的唯一标识符，从时间戳派生。
- `status`：运行的结果（`SUCCESS`、`FAILURE` 或 `WARNING_NO_RESULT_FILE`）。
- `results`：`auto_tune.sh` 运行的 `result.txt` 文件内容。
- `gcs_results`：存储运行产物的 GCS URL（如果提供了 GCS 路径）。

运行完成后，还会在控制台打印成功和失败运行的摘要。
