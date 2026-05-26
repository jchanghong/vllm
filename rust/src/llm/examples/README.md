# LLM 冒烟测试

启动无头 `vllm`：

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

通过 `vllm-llm` 生成接口运行 Rust 冒烟测试：

```bash
cargo run -p vllm-llm --example external_engine_smoke -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1
```

重要提示：每次运行冒烟测试时都必须重启 `vllm`，因为 vLLM 引擎无法管理前端关闭及其后的重连。换句话说，请勿重用现有的 `vllm` 实例（如果有）。
