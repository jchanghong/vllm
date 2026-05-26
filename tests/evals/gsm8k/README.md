# GSM8K 准确率评估

此目录包含 lm-eval-harness GSM8K 评估的替代方案，使用独立的 GSM8K 脚本和 vLLM 服务器以获得更好的性能和可控性。

## 使用方法

### 使用 pytest 运行测试（类似 buildkite）

```bash
pytest -s -v tests/evals/gsm8k/test_gsm8k_correctness.py \
    --config-list-file=configs/models-small.txt
```

### 运行独立的评估脚本

```bash
# 先启动 vLLM 服务器
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000

# 运行评估
python tests/evals/gsm8k/gsm8k_eval.py --port 8000
```

## 配置格式

`configs/` 目录中的模型配置文件使用以下 YAML 格式：

```yaml
model_name: "Qwen/Qwen2.5-1.5B-Instruct"
accuracy_threshold: 0.54  # 最低预期准确率
num_questions: 1319       # 问题数量（默认值：完整测试集）
num_fewshot: 5            # 从训练集中选取的 few-shot 示例数量
server_args: "--max-model-len 4096 --tensor-parallel-size 2"  # 服务器参数
env:                      # 环境变量（可选）
  VLLM_USE_FLASHINFER_MOE_FP4: "1"
```

`server_args` 字段接受可以传递给 `vllm serve` 的任何参数。

`env` 字段接受要为服务器进程设置的环境变量字典。
