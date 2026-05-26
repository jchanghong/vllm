# 结构化输出

此脚本演示了 vLLM 与 OpenAI 兼容服务器的各种结构化输出能力。
它可以单独运行某种约束类型，也可以运行所有类型。
它同时支持流式响应和并发的非流式请求。

要使用此示例，您必须启动一个运行任意模型的 vLLM 服务器。

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct
```

要服务推理模型，可以使用以下命令：

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --reasoning-parser deepseek_r1
```

如果您想使用 `uv` 独立运行此脚本，可以使用以下命令：

```bash
uvx --from git+https://github.com/vllm-project/vllm#subdirectory=examples/features/structured_outputs \
    structured-outputs
```

有关更多信息，请参阅[功能文档](https://docs.vllm.ai/en/latest/features/structured_outputs.html)。

!!! tip
    如果 vLLM 正在远程运行，请在运行脚本前设置 `OPENAI_BASE_URL=<remote_url>`。

## 用法

运行所有约束，非流式：

```bash
uv run structured_outputs_offline.py
```

运行所有约束，流式：

```bash
uv run structured_outputs_offline.py --stream
```

运行特定约束，例如 `structural_tag` 和 `regex`，流式：

```bash
uv run structured_outputs_offline.py \
    --constraint structural_tag regex \
    --stream
```

运行所有约束，使用推理模型并流式：

```bash
uv run structured_outputs_offline.py --reasoning --stream
```
