# `vllm-rs` CLI 快速入门

从仓库根目录使用一条 `vllm-rs serve` 命令启动托管模式的 Qwen3：

```bash
HF_HUB_OFFLINE=1 \
VLLM_CPU_KVCACHE_SPACE=2 \
VLLM_HOST_IP=127.0.0.1 \
VLLM_LOOPBACK_IP=127.0.0.1 \
cargo run --bin vllm-rs -- serve \
  Qwen/Qwen3-0.6B \
  --python ../vllm/.venv/bin/python \
  --max-model-len 512 \
  -- \
  --dtype float16
```

这将启动：

- 一个托管的无头 Python `vllm` 引擎
- 位于 `127.0.0.1:8000` 的 Rust 兼容 OpenAI 前端

所有 Python 引擎参数必须放在 `--` 之后。`--` 之前的参数由 Rust
前端自身解析。

之后，您可以向 Rust 前端发送 OpenAI 风格的请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": true
  }'
```

如果您已经自行启动了无头 `vllm`，请使用 `frontend` 子命令：

```bash
cargo run --bin vllm-rs -- frontend \
  --handshake-address tcp://127.0.0.1:62100 \
  Qwen/Qwen3-0.6B
```
