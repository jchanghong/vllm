# MooncakeConnector 使用指南

## 关于 Mooncake

Mooncake 旨在通过在高速互连的 DRAM/SSD 资源上构建多级缓存池，来提升大语言模型（LLM）的推理效率，尤其是在慢速对象存储环境中。与传统缓存系统相比，Mooncake 利用（GPUDirect）RDMA 技术以零拷贝方式直接传输数据，同时最大化利用单机上多 NIC 资源。

有关 Mooncake 的更多详情，请参阅 [Mooncake 项目](https://github.com/kvcache-ai/Mooncake) 和 [Mooncake 文档](https://kvcache-ai.github.io/Mooncake/)。

## 前提条件

### 安装

通过 pip 安装 mooncake：`uv pip install mooncake-transfer-engine`。

更多安装说明请参阅 [Mooncake 官方仓库](https://github.com/kvcache-ai/Mooncake)。

## 使用方法

### 预填充节点 (192.168.0.2)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8010 --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

### 解码节点 (192.168.0.3)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8020 --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

### 代理

```bash
python examples/disaggregated/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py --prefill http://192.168.0.2:8010 --decode http://192.168.0.3:8020
```

现在您可以通过端口 8000 向代理服务器发送请求。

## 环境变量

- `VLLM_MOONCAKE_BOOTSTRAP_PORT`：Mooncake 引导服务器的端口
    - 默认值：8998
    - 仅预填充实例需要
    - 对于无头实例，必须与主实例相同
    - 每个实例在其主机上需要唯一的端口；在不同主机上使用相同的端口号是可以的

- `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`：自动释放预填充实例上特定请求的 KV 缓存的超时时间（秒）。（可选）
    - 默认值：480
    - 如果请求被中止且解码器尚未通知预填充实例，预填充实例将在此超时后释放其 KV 缓存块，以避免无限期持有它们。

## KV 传输配置

### KV 角色选项

- **kv_producer**：用于生成 KV 缓存的预填充实例
- **kv_consumer**：用于消费预填充实例的 KV 缓存的解码实例
- **kv_both**：启用对称功能，连接器既可以作为生产者也可以作为消费者。这为实验性设置和角色区分未预先确定的场景提供了灵活性。

### kv_connector_extra_config

- **num_workers**：一个预填充工作节点用于通过 mooncake 传输 KV 缓存的线程池大小。（默认值 10）
- **mooncake_protocol**：Mooncake 连接器协议。（默认值 "rdma"）

## 示例脚本/代码

请参阅 vLLM 仓库中的以下示例脚本：

- [run_mooncake_connector.sh](../../examples/disaggregated/mooncake_connector/run_mooncake_connector.sh)
- [mooncake_connector_proxy.py](../../examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py)
