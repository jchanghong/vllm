# IPC 引擎

IPC 权重传输引擎使用 **CUDA IPC**（进程间通信）句柄在训练器和推理工作进程之间直接共享同一** GPU **上的 GPU 内存。这避免了任何数据复制，使其成为训练和推理共置时的最高效选项。支持多 GPU 设置——权重由每个 GPU 全部收集，并由正确的共置进程提取。

## 何时使用 IPC

- 训练和推理共享**相同 GPU**（共置）

## 工作原理

1. 训练器为每个权重创建 CUDA 张量，并使用 `torch.multiprocessing.reductions.reduce_tensor` 生成 IPC 句柄。在多 GPU 设置（例如 FSDP）中，每个训练器 rank 必须先将每一层的完整张量全部收集到自己的 GPU 上，然后才能生成 IPC 句柄。
2. 每个 GPU 的 IPC 句柄通过 **Ray**、**HTTP** 或**自定义可调用对象**发送到推理引擎。每个 rank 只读取对应其自身 GPU 的句柄。
3. 推理工作进程使用 `rebuild_cuda_tensor` 从句柄重建张量，直接从训练器的 GPU 内存中读取。

!!! warning
    IPC 句柄涉及发送序列化的 Python 对象。使用 HTTP 传输时，必须在服务器和客户端两端都设置 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`。这是因为 IPC 句柄需要经过 pickle 序列化和 base64 编码以进行 HTTP 传输。

## 打包（分块）传输

默认情况下，所有权重在单个 API 调用中发送。对于大型模型，这要求完整模型同时驻留在两侧的 GPU 内存中。设置 `packed=True` 可以启用具有有界 GPU 内存的**分块传输**：

- 权重被连接成固定大小的打包缓冲区（由 `packed_buffer_size_bytes` 控制）。
- 每个块在单个 `start_weight_update` / `finish_weight_update` 括号内作为单独的 `update_weights` 调用发送，因此无论块的数量如何，逐层重新加载过程在开始时初始化一次，在结束时最终确定一次。
- 每个块被消费后，该块的 GPU 内存可以被回收。

```python
trainer_args = IPCTrainerSendWeightsArgs(
    send_mode="ray",
    llm_handle=llm_actor_handle,
    packed=True,
    packed_buffer_size_bytes=256 * 1024 * 1024,  # 256 MB 块
)
```

## 初始化

IPC 后端在任一端都不需要初始化。`init_transfer_engine` 调用对 IPC 来说是一个空操作。

## 发送权重

IPC 支持两种传输模式来传递句柄：

### Ray 模式

当 vLLM 作为 Ray actor 运行时使用：

```python
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerSendWeightsArgs,
    IPCWeightTransferEngine,
)

trainer_args = IPCTrainerSendWeightsArgs(
    send_mode="ray",
    llm_handle=llm_actor_handle,
)
# 开始
ray.get(llm_actor_handle.start_weight_update.remote(is_checkpoint_format=True))
# 发送权重
IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
# 完成
ray.get(llm_actor_handle.finish_weight_update.remote())
```

在 Ray 模式下，引擎直接调用 `llm_handle.update_weights.remote(...)`，通过 Ray 的序列化传递 IPC 句柄。

### HTTP 模式

当 vLLM 作为 HTTP 服务器运行时使用：

```python
trainer_args = IPCTrainerSendWeightsArgs(
    send_mode="http",
    url="http://localhost:8000",
)

# 开始
base_url = "http://localhost:8000"
url = f"{base_url}/start_weight_update"
response = requests.post(url, json={"is_checkpoint_format": True}, timeout=60)
response.raise_for_status()
# 发送权重
IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
# 完成
url = f"{base_url}/finish_weight_update"
response = requests.post(url, json={}, timeout=60)
response.raise_for_status()
```

在 HTTP 模式下，IPC 句柄经过 pickle 序列化、base64 编码，并作为 JSON 发送到 `/update_weights` 端点。由于工作进程通过 `pickle.loads` 反序列化负载，vLLM 服务器必须使用 `VLLM_ALLOW_INSECURE_SERIALIZATION=1` 启动。

```python
def my_custom_sender(update_info: IPCWeightTransferUpdateInfo):
    # 将 update_info 传递给 vLLM 的自定义逻辑
    ...

trainer_args = IPCTrainerSendWeightsArgs(
    send_mode=my_custom_sender,
)

IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

可配置字段的完整列表请参见 [`IPCTrainerSendWeightsArgs`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/weight_transfer/ipc_engine.py)。

## 示例

- [使用 IPC 权重同步的 RLHF（离线，Ray）](../../../examples/rl/rlhf_ipc.py) - 使用 Ray 放置组和 CUDA IPC 句柄在单 GPU 上共置训练和推理
- [使用 IPC 权重同步的 RLHF（在线服务，HTTP）](../../../examples/rl/rlhf_http_ipc.py) - 与 vLLM HTTP 服务器的权重传输，其中服务器和训练器共享同一 GPU
