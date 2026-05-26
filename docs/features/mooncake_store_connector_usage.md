# MooncakeStoreConnector 使用指南

MooncakeStoreConnector 是一个 KV 缓存连接器，使用 [MooncakeDistributedStore](https://github.com/kvcache-ai/Mooncake) 作为共享 KV 缓存池。与在预填充器和解码器之间进行直接点对点 KV 传输的 `MooncakeConnector` 不同，MooncakeStoreConnector 支持将 KV 缓存卸载到外部分布式存储，支持：

- **CPU/磁盘卸载**：通过 Mooncake 的传输引擎将 KV 缓存卸载到 CPU 内存或磁盘，扩展有效的 KV 缓存容量。
- **跨实例前缀缓存**：基于哈希的去重允许通过存储共享缓存 KV 块，使多个 vLLM 实例受益。
- **单节点和多节点部署**：既可作为独立 KV 缓存扩展使用，也可在分离式预填充-解码设置中使用。

## 前提条件

### 安装 Mooncake

通过 pip 安装 mooncake：

```bash
uv pip install mooncake-transfer-engine
```

更多安装说明和从源码构建，请参考 [Mooncake 官方仓库](https://github.com/kvcache-ai/Mooncake)。

### 启动 Mooncake Master 服务器

Mooncake master 管理元数据并协调分布式存储。在启动 vLLM 之前启动它：

```bash
mooncake_master --port 50051
```

默认端口：

- RPC：50051

多个 vLLM 实例可以共享同一个 master 服务器。

### 配置 Mooncake

创建一个 JSON 配置文件（例如 `mooncake_config.json`）：

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "80GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
```

- `mode`：拓扑选择。`"embedded"`（默认，PR-40900 基线）使每个 vLLM 等级在进程中贡献 `global_segment_size` 到池中。`"standalone-store"` 使等级成为纯请求者——外部 `mooncake_client` 进程拥有 CPU 池和（可选）SSD 层。
- `protocol`：使用 `"rdma"` 获得最佳性能。`"tcp"` 可作为回退方案。
- `global_segment_size`：贡献给分布式池的 CPU 内存（每个 GPU）。在 `embedded` 模式下必须 `> 0`，在 `standalone-store` 模式下必须为 `0`。
- `local_buffer_size`：用于此节点自身操作的私有缓冲区（每个 GPU）。
- `enable_offload`：当为 `true` 时，vLLM 分配一个 DirectIO 暂存缓冲区，以便大型预填充不会超出拥有者的 SSD 写入预算。请同时设置 `mooncake_master` 上相应的 `--enable_offload=true` 标志和外部 `mooncake_client`（如有）的相应标志。

通过环境变量设置配置路径：

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```

## 用法

### 单节点 KV 缓存卸载

使用 MooncakeStoreConnector 将 KV 缓存卸载到 CPU 内存，扩展有效缓存大小：

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
```

### 分离式预填充-解码（XpYd）

在分离式预填充-解码模式下，使用 `MultiConnector` 结合 `MooncakeConnector`（点对点 KV 传输）和 `MooncakeStoreConnector`（共享 KV 缓存池）。这同时支持预填充器和解码器之间的直接 P2P 传输，以及通过分布式存储的跨实例前缀缓存共享。
**预填充器节点：**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50052 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8100 \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_producer"
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both"
                }
            ]
        }
    }'
```

**解码器节点：**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50053 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8200 \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_consumer"
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_consumer"
                }
            ]
        }
    }'
```

**代理：**

需要一个分离代理在预填充器和解码器节点之间路由请求。代理分配 `do_remote_prefill=True` / `do_remote_decode=True` 以协调通过 `MooncakeConnector` 的 P2P 传输。有关代理设置详情，请参考 [MooncakeConnector 使用指南](mooncake_connector_usage.md)。

### 磁盘卸载

磁盘卸载最常在 `standalone-store` 模式下运行：外部 `mooncake_client` 进程拥有 CPU 池和 SSD 层，每个 vLLM 等级是纯请求者。这避免了每个等级重复 SSD 池，并将 DirectIO 预算跟踪保持在单个进程上。

端到端磁盘卸载需要对齐三件事：

1. **`mooncake_master`** 以 `--enable_offload=true` 启动。
2. **`mooncake_client`**（拥有者）以 `--enable_offload=true` 启动，并通过 `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` 指定 SSD 路径。
3. **vLLM 侧**在 JSON 配置文件中设置 `"enable_offload": true`（这由连接器读取，**不是**环境变量）。

vLLM 侧的 `mooncake_config.json` 示例：

```json
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": 0,
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "mlx5_0",
  "enable_offload": true
}
```

通过以下方式将此等级指向本地拥有者段：

```bash
export MOONCAKE_PREFERRED_SEGMENT=127.0.0.1:50053
```

拥有者的 SSD 目录、磁盘驱逐策略和 DirectIO 暂存缓冲区大小通过标准 Mooncake 环境变量在 `mooncake_client` 侧控制（`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`、`MOONCAKE_BUCKET_EVICTION_POLICY`、`MOONCAKE_USE_URING`、`MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`、`MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` 等）。这些与 vLLM JSON 配置无关。

## 环境变量

| 变量 | 描述 | 默认值 |
| --- | --- | --- |
| `MOONCAKE_CONFIG_PATH` | Mooncake JSON 配置文件的路径 | （必需） |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | MooncakeConnector P2P 传输的引导端口（仅分离模式） | 8998 |
| `MOONCAKE_PREFERRED_SEGMENT` | 将此等级的副本固定到特定的拥有者段（`host:port`）；在 `standalone-store` 模式下使用 | — |
| `MOONCAKE_REQUESTER_LOCAL_HOSTNAME` | 覆盖 vLLM 等级向 Mooncake 注册为请求者的主机名。默认为等级解析的 IP。 | — |
| `VLLM_MOONCAKE_STORE_TIER_LOG` | 当为 `1` 时，记录每批次的层摘要（内存与磁盘命中数）以进行可观测性 | 禁用 |
| `VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO` | 在单次 `batch_get_into_multi_buffers` 调用中，请求者将填满的拥有者 DirectIO 暂存缓冲区的比例。越低 → 越保守的预分割，更多轮次。 | 0.9 |

## KV 传输配置

### KV 角色选项

- **kv_producer**：用于将 KV 缓存存储到池中的实例。
- **kv_consumer**：用于从池中加载 KV 缓存的实例。
- **kv_both**：实例同时存储和加载 KV 缓存。用于单节点 CPU 卸载或预填充器实例。

### kv_connector_extra_config

- `load_async`（bool）：启用异步加载以获得更好的计算-I/O 重叠。默认值：`true`。
- `enable_cross_layers_blocks`（bool）：启用跨层块打包以减少存储操作。默认值：`false`。
- `lookup_rpc_port`（int）：ZMQ 查找 RPC 套接字的自定义端口。默认值：`0`。

## 注意事项

### 跨进程的可重现块哈希

`MooncakeStoreConnector` 依赖于跨共享分布式存储的所有 vLLM 进程的一致块哈希。由于 Python 默认在每个进程中随机化其哈希种子，相同的提示词在不同进程上可能产生不同的块哈希——从而阻止跨进程的前缀缓存命中。

在每个共享存储的实例上设置固定的 `PYTHONHASHSEED`（DP 等级、独立的预填充器/解码器节点，以及指向同一 Mooncake 存储的任何其他 vLLM 进程）：

```bash
PYTHONHASHSEED=0 vllm serve ...
```
