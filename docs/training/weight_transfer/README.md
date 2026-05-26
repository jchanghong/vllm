# 权重传输

vLLM 提供了一个可插拔的权重传输系统，用于在强化学习（RL）工作流中将模型权重从训练进程同步到推理引擎。这对于 RLHF、GRPO 和其他在线 RL 方法至关重要，在这些方法中，策略模型在训练期间迭代更新，更新后的权重必须反映在推理引擎中以进行 rollout 生成。

## 架构

权重传输系统遵循**四阶段协议**，采用可插拔后端设计：

1. **初始化**（`init_weight_transfer_engine`）：建立训练器和推理工作进程之间的通信通道。在训练循环开始前调用一次。
2. **开始**（`start_weight_update`）：准备推理引擎进行权重更新。
3. **权重更新**（`update_weights`）：将更新后的权重从训练器传输到推理引擎。可以调用一次或多次（例如，用于分块传输）。
4. **完成**（`finish_weight_update`）：完成权重更新（例如，对检查点格式的权重进行后处理）。在所有权重传输完毕后调用一次。

## 可用后端

| 后端 | 传输方式 | 使用场景 |
| ------- | --------- | -------- |
| [NCCL](nccl.md) | NCCL broadcast | 训练和推理使用独立 GPU |
| [IPC](ipc.md) | CUDA IPC handles | 训练和推理共置于同 GPU |

## 配置

通过 `WeightTransferConfig` 指定权重传输后端。后端决定了哪个引擎处理权重同步。

### 编程方式（离线推理）

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(
    model="my-model",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),  # 或 "ipc"
)
```

### CLI（在线服务）

```bash
vllm serve my-model \
    --weight-transfer-config '{"backend": "nccl"}'
```

`backend` 字段接受 `"nccl"`（默认）或 `"ipc"`。

## API 端点

将 vLLM 作为 HTTP 服务器运行时，以下端点可用于权重传输：

| 端点 | 方法 | 描述 |
| -------- | ------ | ----------- |
| `/init_weight_transfer_engine` | POST | 使用后端特定信息初始化权重传输引擎 |
| `/start_weight_update` | POST | 开始权重更新 |
| `/update_weights` | POST | 传输一批权重及后端特定元数据 |
| `/finish_weight_update` | POST | 完成权重更新并运行后处理 |
| `/pause` | POST | 在权重同步前暂停生成以处理进行中的请求 |
| `/resume` | POST | 在权重同步后恢复生成 |
| `/get_world_size` | GET | 获取推理工作进程数量（用于 NCCL world size 计算） |

!!! note
    HTTP 权重传输端点需要设置 `VLLM_SERVER_DEV_MODE=1`。

## 训练器端 API

两个后端都提供了训练器调用以发送权重的静态方法。通用模式如下：

```python
# 1. 初始化传输引擎（后端特定）
EngineClass.trainer_init(init_info)

# 2. 在推理端开始权重更新
llm.start_weight_update(is_checkpoint_format=True)

# 3. 将权重发送到推理工作进程
EngineClass.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=backend_specific_args,
)

# 4. 在推理端完成权重更新
llm.finish_weight_update()
```

请参阅 [NCCL](nccl.md) 和 [IPC](ipc.md) 页面了解后端特定的训练器 API 和完整示例。

## 扩展系统

权重传输系统设计为可扩展的。您可以通过继承 `WeightTransferEngine` 并将其注册到工厂来实现自定义后端。详情请参阅[基类](base.md)页面。
