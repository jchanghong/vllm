# 分布式 KV 缓存传输

本文件夹实现了跨 vLLM 实例的分布式 KV 缓存传输。
目前主要用例是分离式预填充（disaggregated prefilling）。

## 抽象层

KV 缓存传输包含三层抽象：

- **KV 管道**：用于 torch.tensor 传输的 FIFO 管道。关键 API：`send_tensor` 和 `recv_tensor`。
- **KV 查找缓冲区**：用于 KV 缓存的查找缓冲区。键：令牌（tokens），值：KV 缓存（和/或隐藏状态）。关键 API：`insert` 和 `drop_select`（类似于 SQL 语义）。
- **KV 连接器**：将 KV 管道和 KV 查找缓冲区连接到 vLLM 的连接器。关键 API：`send_kv_caches_and_hidden_states` 和 `recv_kv_caches_and_hidden_states`。

为什么需要 KV 查找缓冲区：仅靠 FIFO 管道是不够的，因为预填充 vLLM 工作节点处理请求的顺序可能与解码 vLLM 工作节点不同。假设 QPS 非常高，预填充工作节点可能按 A -> B -> C 的顺序处理请求，但解码工作节点可能先处理请求 C。这种情况无法被 FIFO 管道自然处理，因此我们提供 KV 查找缓冲区来帮助将 FIFO 管道转换为查找缓冲区。

注意：KV 管道层是可绕过的：如果你的分布式通信服务已经支持基于键值的查找（如 Redis 或 RDMA 数据库），可以跳过这一层。

注意：如果你不仅想传输 KV 缓存，还想调整 vLLM 的模型执行流程（例如，允许 vLLM 在某些令牌上接收 KV 缓存，并在剩余令牌上执行预填充），你可以同时绕过 KV 管道层和 KV 查找缓冲区层，直接在 KV 连接器层上实现。请记住，由于 vLLM 的模型输入不断变化，当 vLLM 有新的更新时，这种实现很可能会失效。

## 分离式预填充

示例用法见[此文件](../../../examples/disaggregated/disaggregated_prefill.sh)。

以下是我们运行分离式预填充的示意图。

![分离式预填充工作流程](./disagg_prefill_workflow.jpg)
