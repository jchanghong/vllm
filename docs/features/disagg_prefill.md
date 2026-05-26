# 分离式预填充（实验性）

本页介绍 vLLM 中的分离式预填充功能。

!!! note
    此功能为实验性功能，可能会发生变化。

## 为什么需要分离式预填充？

两个主要原因：

- **分别调整首令牌延迟（TTFT）和令牌间延迟（ITL）**。分离式预填充将 LLM 推理的预填充和解码阶段放在不同的 vLLM 实例中。这使您可以灵活地分配不同的并行策略（例如 `tp` 和 `pp`）来调整 TTFT 而不影响 ITL，或调整 ITL 而不影响 TTFT。
- **控制尾部 ITL**。如果没有分离式预填充，vLLM 可能会在处理一个请求的解码过程中插入一些预填充任务。这会导致更高的尾部延迟。分离式预填充帮助您解决此问题并控制尾部 ITL。使用适当块大小的分块预填充也能达到相同的目的，但在实践中很难确定正确的块大小值。因此，分离式预填充是控制尾部 ITL 更可靠的方法。

!!! note
    分离式预填充 **不会** 提高吞吐量。

## 使用示例

请参阅 [examples/disaggregated/disaggregated_prefill.sh](../../examples/disaggregated/disaggregated_prefill.sh) 了解分离式预填充的示例用法。

目前支持 6 种类型的连接器：

- **ExampleConnector**：请参阅 [examples/disaggregated/example_connector/run.sh](../../examples/disaggregated/example_connector/run.sh) 了解 ExampleConnector 分离式预填充的示例用法。
- **LMCacheConnectorV1**：请参阅 [examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh](../../examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh) 了解使用 NIXL 作为底层 KV 传输的 LMCacheConnectorV1 分离式预填充的示例用法。
- **NixlConnector**：请参阅 [tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh](../../tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh) 了解 NixlConnector 分离式预填充的示例用法，它支持完全异步发送/接收。有关详细使用指南，请参阅 [NixlConnector 使用指南](nixl_connector_usage.md)。有关功能兼容性详情，请参阅 [NixlConnector 兼容性矩阵](nixl_connector_compatibility.md)。
- **P2pNcclConnector**：请参阅 [examples/disaggregated/p2p_nccl_xpyd/disagg_example_p2p_nccl_xpyd.sh](../../examples/disaggregated/p2p_nccl_xpyd/disagg_example_p2p_nccl_xpyd.sh) 了解 P2pNcclConnector 分离式预填充的示例用法。
- **MooncakeConnector**：请参阅 [examples/disaggregated/mooncake_connector/run_mooncake_connector.sh](../../examples/disaggregated/mooncake_connector/run_mooncake_connector.sh) 了解 MooncakeConnector 分离式预填充的示例用法。有关详细使用指南，请参阅 [MooncakeConnector 使用指南](mooncake_connector_usage.md)。
- **MultiConnector**：利用 `KVTransferConfig` 中已有的 `kv_connector_extra_config: dict[str, Any]` 将所有需要的连接器存储为有序的 kwargs 列表。例如：

  ```bash
  --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both"},{"kv_connector":"ExampleConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"local_storage"}}]}}'
  ```

对于 NixlConnector，您还可以指定一个或多个 NIXL_Backend。例如：

  ```bash
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_buffer_device":"cuda", "kv_connector_extra_config":{"backends":["UCX", "GDS"]}}'
  ```

- **OffloadingConnector**：支持将 KV 数据卸载到 CPU 内存，自定义 CPU 块大小（以 token 为单位）和要分配的 CPU 总字节数：

  ```bash
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"block_size": 64, "cpu_bytes_to_use": 1000000000}}'
  ```

- **FlexKVConnectorV1**：请参阅 [examples/disaggregated/flexkv_connector/prefix_caching_flexkv.py](../../examples/disaggregated/flexkv_connector/prefix_caching_flexkv.py) 了解 FlexKVConnectorV1 的示例用法。FlexKV 是一个用于超大规模 LLM 推理的分布式 KV 存储和多级缓存管理系统。

  ```bash
  --kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
  ```

## 基准测试

请参阅 [benchmarks/disagg_benchmarks](../../benchmarks/disagg_benchmarks) 了解分离式预填充的基准测试。

## 开发

我们通过运行 2 个 vLLM 实例来实现分离式预填充。一个用于预填充（我们称之为预填充实例），一个用于解码（我们称之为解码实例），然后使用连接器将预填充的 KV 缓存和结果从预填充实例传输到解码实例。

所有分离式预填充的实现代码位于 `vllm/distributed/kv_transfer` 下。

分离式预填充的关键抽象：

- **Connector（连接器）**：连接器允许 **KV 消费者** 从 **KV 生产者** 检索一批请求的 KV 缓存。
- **LookupBuffer（查找缓冲区）**：LookupBuffer 提供两个 API：`insert`（插入）KV 缓存和 `drop_select`（删除选择）KV 缓存。`insert` 和 `drop_select` 的语义类似于 SQL，其中 `insert` 将 KV 缓存插入缓冲区，而 `drop_select` 返回符合给定条件的 KV 缓存并将其从缓冲区中移除。
- **Pipe（管道）**：用于张量传输的单向 FIFO 管道。它支持 `send_tensor` 和 `recv_tensor`。

!!! note
    `insert` 是非阻塞操作，但 `drop_select` 是阻塞操作。

下图说明了上述 3 种抽象的组织方式：

![分离式预填充抽象层](../assets/features/disagg_prefill/abstraction.jpg)

分离式预填充的工作流程如下：

![分离式预填充工作流程](../assets/features/disagg_prefill/overview.jpg)

图中的 `buffer` 对应 LookupBuffer 中的 `insert` API，`drop_select` 对应 LookupBuffer 中的 `drop_select` API。

现在，vLLM 中的每个进程都会有一个对应的连接器。具体来说，我们有：

- 调度器连接器：与调度器进程位于同一进程中的连接器。它调度 KV 缓存传输操作。
- 工作节点连接器：位于工作进程中的连接器。它们执行 KV 缓存传输操作。

下图说明了上述 2 种连接器的组织方式：

![分离式预填充高层设计](../assets/features/disagg_prefill/high_level_design.png)

下图展示了工作节点连接器如何与注意力模块协同工作，实现逐层的 KV 缓存存储和加载：

![分离式预填充工作流程](../assets/features/disagg_prefill/workflow.png)

## 第三方贡献

分离式预填充与基础设施高度相关，因此 vLLM 依赖第三方连接器来实现生产级别的分离式预填充（vLLM 团队将积极审查并合并第三方连接器的新 PR）。

我们推荐三种实现方式：

- **完全自定义连接器**：实现您自己的 `Connector`，调用第三方库来发送和接收 KV 缓存，以及更多功能（例如编辑 vLLM 的模型输入以执行自定义预填充等）。这种方法给您最大的控制权，但存在与未来 vLLM 版本不兼容的风险。
- **数据库式连接器**：实现您自己的 `LookupBuffer`，并像 SQL 一样支持 `insert` 和 `drop_select` API。
- **分布式 P2P 连接器**：实现您自己的 `Pipe`，并像 `torch.distributed` 一样支持 `send_tensor` 和 `recv_tensor` API。
