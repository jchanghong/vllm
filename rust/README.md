# vllm-frontend-rs

这是 vLLM 的 Rust 直接替代前端。当前目标是用 Rust 重构北向服务层，同时仍通过 ZMQ 在现有引擎边界上与核心 Python vLLM 引擎进程通信。

它仍应被视为实验性的，尚未实现完整功能。我们正在努力从 Python 前端添加更多功能。

原始提交历史（在移入主 vllm 仓库之前）请参见 <https://github.com/Inferact/vllm-frontend-rs>。

## 架构

该组件组织为一个 Cargo 工作空间，包含多个 crate，自底向上分层：

```text
┌─────────────────────────────────┐
│  vllm-cmd / vllm-rs             │  CLI 入口点：
│                                 │  Python vLLM 前端子进程
│                                 │  Rust 托管引擎服务模式
├─────────────────────────────────┤
│  vllm-server                    │  兼容 OpenAI 的 HTTP API (axum)
├─────────────────────────────────┤
│  vllm-chat                      │  聊天补全：模板渲染、
│                                 │  结构化助手事件、
│                                 │  推理与工具解析
├─────────────────────────────────┤
│  vllm-text                      │  分词器与增量逆分词器
├─────────────────────────────────┤
│  vllm-llm                       │  引擎客户端上层的
│                                 │  轻量 token 输入/输出外观
├─────────────────────────────────┤
│  vllm-engine-core-client        │  无头 vLLM 引擎的
│                                 │  ZMQ 传输 + MessagePack 协议
└─────────────────────────────────┘
```

`vllm-rs` 作为 Rust 前端子进程集成到 Python `vllm` 中。
Python 负责进程启动，并将 Rust API 服务器作为 Python 监督的工作进程启动，
同时将继承的监听套接字和传输地址传递给 `vllm-rs`。

例如：

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B
```

### 外部引擎

当 Python 引擎在其他地方启动，且本节点应仅运行 Rust 前端时，
`vllm-rs serve` 可以使用 `--data-parallel-size-local 0` 独立运行。前端仍然使用
全局的 `--data-parallel-size` 来确定它期望加入共享握手的引擎数量。

```bash
vllm serve Qwen/Qwen3-0.6B \
  --headless \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1
```

然后启动纯 Rust 前端服务器：

```bash
vllm-rs serve Qwen/Qwen3-0.6B \
  --data-parallel-address 127.0.0.1 \
  --data-parallel-rpc-port 62100 \
  --data-parallel-size 1 \
  --data-parallel-size-local 0
```

单独构建 `vllm-rs`：

```bash
# 从本地检出目录
cargo install --path src/cmd --bin vllm-rs
```

### 示例请求

两种启动路径完成后，都可以使用任何兼容 OpenAI 的客户端：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": true
  }'
```
