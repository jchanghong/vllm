# Engine-Core 冒烟测试

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
  --enable-sleep-mode \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size-local 1 \
  --max-model-len 512 \
  --dtype float16
```

通过 `vllm-engine-core-client` 工具接口运行 Rust 冒烟测试：

```bash
cargo run -p vllm-engine-core-client --example external_engine_utility_call -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1
```

如果当前引擎设置不支持睡眠模式，请跳过冒烟测试中的 `sleep` / `wake_up` 部分：

```bash
cargo run -p vllm-engine-core-client --example external_engine_utility_call -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1 \
  --skip-sleep-wake
```

通过原始 engine-core 请求路径运行用于样例 logprobs 解码的 Rust 冒烟测试：

```bash
cargo run -p vllm-engine-core-client --example external_engine_logprobs -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1
```

该冒烟测试请求一个较小的已生成 token 的 `logprobs` 载荷，外加在一个更长的
提示上的 prompt logprobs，因此它针对真实引擎同时练习了内联和辅助帧解码路径。
Rust 客户端将这些载荷解码为语义化的逐位置记录，而不是暴露
原始的 ndarray/tensor 连线格式。

重要提示：每次运行冒烟测试时都必须重启 `vllm`，因为 vLLM 引擎无法管理前端关闭及其后的重连。换句话说，请勿重用现有的 `vllm` 实例（如果有）。
