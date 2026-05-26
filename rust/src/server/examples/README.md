# 服务器冒烟测试

启动一个新的无头 `vllm` 引擎：

```bash
source ../vllm/.venv/bin/activate
HF_HUB_OFFLINE=1 \
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_CPU_KVCACHE_SPACE=2 \
VLLM_HOST_IP=127.0.0.1 \
VLLM_LOOPBACK_IP=127.0.0.1 \
python3 -m vllm.entrypoints.cli.main serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size-local 1 \
  --max-model-len 512 \
  --dtype float16
```

运行 Rust 服务器冒烟测试：

```bash
cargo run -p vllm-server --example external_engine_openai_qwen -- \
  --handshake-address tcp://127.0.0.1:62100
```

该示例在临时本地端口上启动 Rust 兼容 OpenAI 的服务器，
通过 `async-openai` Rust 客户端连接它，列出模型，然后验证
流式聊天补全能生成助手角色块、最终答案
内容块以及终止结束块。此示例特意使用
`async-openai` 的标准类型化 `create_stream` API 而非 BYOT，因此它不会
检查非标准的 `reasoning_content` 字段，尽管 Rust
服务器可能会为支持推理的模型（如 Qwen3）发出该字段。有关推理
行为本身，请使用 `vllm-chat` 冒烟测试或 `vllm-server`
路由测试。

重要提示：每次运行冒烟测试时，都必须重启 `vllm`。当前的无头
引擎在客户端关闭后无法安全地处理前端重连。
