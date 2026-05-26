# 使用 GPT-OSS 进行 GPQA 评估

此目录包含使用 GPT-OSS 评估包和 vLLM 服务器的 GPQA 评估测试。

## 使用方法

### 使用 pytest 运行测试（类似 buildkite）

```bash
# H200
pytest -s -v tests/evals/gpt_oss/test_gpqa_correctness.py \
    --config-list-file=configs/models-h200.txt

# B200
pytest -s -v tests/evals/gpt_oss/test_gpqa_correctness.py \
    --config-list-file=configs/models-b200.txt
```

## 配置格式

`configs/` 目录中的模型配置文件使用以下 YAML 格式：

```yaml
model_name: "openai/gpt-oss-20b"
metric_threshold: 0.568          # 最低预期准确率
reasoning_effort: "low"          # 推理努力程度（默认值："low"）
server_args: "--tensor-parallel-size 2"  # 服务器参数
startup_max_wait_seconds: 1800   # 等待服务器启动的最长时间（默认值：1800）
env:                             # 环境变量（可选）
  SOME_VAR: "value"
```

`server_args` 字段接受可以传递给 `vllm serve` 的任何参数。

`env` 字段接受要为服务器进程设置的环境变量字典。

## 添加新模型

1. 在 `configs/` 目录中创建一个新的 YAML 配置文件
2. 将文件名添加到相应的 `models-*.txt` 文件中

## Tiktoken 编码文件

vLLM 服务器所需的 tiktoken 编码文件会在首次运行时自动从 OpenAI 的公共 blob 存储下载：

- `cl100k_base.tiktoken`
- `o200k_base.tiktoken`

文件缓存在 `data/` 目录中。运行评估时，`TIKTOKEN_ENCODINGS_BASE` 环境变量会自动设置为指向此目录。
