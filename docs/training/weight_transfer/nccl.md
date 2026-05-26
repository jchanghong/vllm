# NCCL 引擎

NCCL 权重传输引擎使用 [NCCL](https://developer.nvidia.com/nccl) broadcast 操作将权重从训练器传输到推理工作进程。它支持训练器和推理引擎在不同 GPU 上运行的**多节点**和**多 GPU** 设置。

## 何时使用 NCCL

- 训练和推理在**独立 GPU** 上（可能跨节点）
- **张量并行**推理，多个工作进程都需要更新的权重
- 需要通过 NVLink 或 InfiniBand 进行高带宽、低延迟的权重传输

## 工作原理

1. 训练器和所有推理工作进程使用 `StatelessProcessGroup`（vLLM 的独立于 torch.distributed 的组抽象）加入一个共享的 NCCL 进程组。
2. 训练器同时向所有工作进程广播权重。每个工作进程接收并增量加载权重。
3. 可选地，**打包张量广播**将多个小张量批量合并到更大的缓冲区中，使用双倍/三倍缓冲和 CUDA 流重叠以提高吞吐量。此实现基于 [NeMo-RL 的 packed tensor](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/utils/packed_tensor.py)。

## 初始化

NCCL 需要显式的进程组设置。训练器和推理工作进程必须就 master 地址、端口和 world size 达成一致。

### 推理端

```python
from vllm.distributed.weight_transfer.base import WeightTransferInitRequest

# rank_offset 用于说明训练器占据了 rank 0
llm.init_weight_transfer_engine(
    WeightTransferInitRequest(
        init_info=dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,  # 训练器 + 所有推理工作进程
        )
    )
)
```

### 训练器端

```python
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
)

group = NCCLWeightTransferEngine.trainer_init(
    dict(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,
    )
)
```

!!! note
    `trainer_init` 总是将训练器分配到 rank 0。推理工作进程从 `rank_offset`（通常为 1）开始。

## 发送权重

```python
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)

trainer_args = NCCLTrainerSendWeightsArgs(
    group=group,
    packed=True,  # 使用打包广播以提高效率
)

NCCLWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

可配置字段的完整列表请参见 [`NCCLTrainerSendWeightsArgs`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/weight_transfer/nccl_engine.py)。

### 打包张量广播

当 `packed=True` 时，多个权重张量在广播之前被打包成大的连续缓冲区。这减少了 NCCL 操作的数量，并使用双倍/三倍缓冲以及专用 CUDA 流来实现打包、广播和解包之间的重叠。

训练器端（`NCCLTrainerSendWeightsArgs`）和推理端（`NCCLWeightTransferUpdateInfo`）必须使用匹配的 `packed_buffer_size_bytes` 和 `packed_num_buffers` 值。

## 接收权重（推理端）

推理端使用四阶段协议触发权重接收——`init_weight_transfer_engine`、`start_weight_update`、`update_weights`、`finish_weight_update`。初始化阶段如上[所示](#initialization)。其余三个步骤如下：

```python
from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

# 1. 开始权重更新
llm.start_weight_update(is_checkpoint_format=True)

# 2. 接收权重（对于分块传输可以多次调用）
llm.update_weights(
    WeightTransferUpdateRequest(
        update_info=dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=True,
        )
    )
)

# 3. 完成权重更新
llm.finish_weight_update()
```

`names`、`dtype_names` 和 `shapes` 列表描述每个参数。它们必须与训练器迭代其参数时的顺序相匹配。

`start_weight_update` 必须在 `update_weights` 之前调用，`finish_weight_update` 必须在所有权重块传输完毕后调用。`is_checkpoint_format` 标志控制是否应用逐层重新加载处理（`True` 用于检查点格式的权重，`False` 用于预处理的内核格式权重）。

## 示例

- [使用 NCCL 权重同步的 RLHF（离线，Ray）](../../../examples/rl/rlhf_nccl.py) - 训练器在一个 GPU 上，2 路张量并行 vLLM 引擎在另外两个 GPU 上，使用打包 NCCL 权重广播
- [使用异步权重同步的 RLHF（离线，Ray）](../../../examples/rl/rlhf_async_new_apis.py) - 异步生成，带有飞行中暂停、权重同步、恢复以及针对新模型的验证
- [使用 NCCL 权重同步的 RLHF（在线服务，HTTP）](../../../examples/rl/rlhf_http_nccl.py) - 使用 HTTP 控制面和 NCCL 数据面与运行中的 vLLM HTTP 服务器进行权重传输
