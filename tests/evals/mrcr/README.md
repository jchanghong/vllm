# MRCR 长上下文准确率评估

使用 OpenAI 的公开 [`openai/mrcr`](https://huggingface.co/datasets/openai/mrcr) 数据集对长上下文行为进行冒烟测试。模型会看到一个包含多个近似重复"针"的长对话，并且必须逐字复现特定的早期助手回复，并在前面加上一个随机的反猜测字符串。

**评分：** 如果响应不是以 `random_string_to_prepend` 开头，则得分为 0；否则去除前缀后，报告与参考答案的平均 `SequenceMatcher.ratio()`。

## 使用方法

```bash
# Pytest（启动服务器）
pytest -s -v tests/evals/mrcr/test_mrcr_correctness.py \
    --config-list-file=configs/models-small.txt

# 独立运行（服务器已启动；模型和上下文自动发现）
vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3 --port 8000
python tests/evals/mrcr/mrcr_eval.py --port 8000
```

## 配置

```yaml
model_name: "Qwen/Qwen3-0.6B"
# 每个针的阈值可捕获聚合结果可能隐藏的特定桶回归（滑动窗口、
# 分块预填充、前缀缓存）。也接受标量值
#（例如 `match_ratio_threshold: 0.20`），并与平均匹配率进行比较。
match_ratio_threshold:
  2: 0.30
  4: 0.15
  8: 0.10
num_samples: 30
needles: [2, 4, 8]
# max_prompt_tokens: 32768       # 可选；默认为服务器 max_model_len - max_tokens - 256
max_tokens: 2048
concurrency: 8
server_args: "--max-model-len 32768 --reasoning-parser qwen3"
```

## 备注

- 样本从三个 parquet 分片中流式获取（`{N}needle/{N}needle_0.parquet`）；仅获取前几个行组，而非完整的 1.4 GB 仓库。
- `max_prompt_tokens` 默认为 `max_model_len - max_tokens - 256`，即填充服务器公布的最大上下文长度。在服务器上设置 `--max-model-len` 以控制冒烟测试的上下文长度；在客户端覆盖 `--max-prompt-tokens` 以限制在此值以下。
- 样本长度通过 `n_chars × 4 ≤ max_prompt_tokens` 进行预过滤，然后通过服务器的 `/tokenize` 端点在实际的聊天模板下进行验证。
- 推理模型：使用 `--reasoning-parser <name>`（例如 `qwen3`、`deepseek_r1`）启动服务器，以便 `<think>` 内容进入 `message.reasoning_content`，不会污染评分答案。
