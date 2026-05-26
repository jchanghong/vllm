# 聊天冒烟测试

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

通过 `vllm-chat` 接口运行 Rust 聊天冒烟测试：

```bash
cargo run -p vllm-chat --example external_engine_chat_qwen -- \
  --handshake-address tcp://127.0.0.1:62100 \
  --host 127.0.0.1 \
  --prompt 'What is the capital of France? Answer with one word.'
```

该示例现在默认使用 `Qwen/Qwen3-0.6B`。当前的 `vllm-chat`
请求模型保持文本优先，支持纯字符串内容或
OpenAI 风格的文本块，而输出端现在会发出结构化的助手
事件，并自动为支持的模型分离推理块。工具
使用和多模态输入仍不在范围之内。它使用 Rust
`tokenizers` 库作为分词器本身，并加载标准的 Hugging Face
配置文件来获取聊天模板和 EOS 元数据。

重要提示：每次运行冒烟测试时，都必须重启 `vllm`。当前的无头
引擎在客户端关闭后无法安全地处理前端重连。
