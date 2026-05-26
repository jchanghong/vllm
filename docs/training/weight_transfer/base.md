# 基类与自定义引擎

权重传输系统构建在一个抽象基类之上，该基类定义了 vLLM 的工作进程基础设施与传输后端之间的契约。您可以通过继承 `WeightTransferEngine` 并将其注册到 `WeightTransferEngineFactory` 来实现自定义后端。

## WeightTransferEngine

`WeightTransferEngine` 是一个泛型抽象类，由两个数据类类型参数化：

- **`TInitInfo`**（继承 `WeightTransferInitInfo`）：后端特定的初始化参数。
- **`TUpdateInfo`**（继承 `WeightTransferUpdateInfo`）：后端特定的权重更新元数据。

### 抽象方法

子类必须实现以下四个方法：

| 方法 | 端 | 描述 |
| ------ | ---- | ----------- |
| `init_transfer_engine(init_info)` | 推理端 | 在每个推理工作进程上初始化通信通道 |
| `receive_weights(update_info, load_weights)` | 推理端 | 接收权重并增量调用 `load_weights` |
| `shutdown()` | 推理端 | 清理资源 |
| `trainer_send_weights(iterator, trainer_args)` | 训练器端 | 从训练器进程发送权重的静态方法 |

### 请求类

API 层的请求类使用普通字典提供与后端无关的序列化。引擎的 `parse_init_info` 和 `parse_update_info` 方法将这些字典转换为类型化的数据类。

```python
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)

# 初始化请求（dict 被转换为后端特定的 TInitInfo）
init_request = WeightTransferInitRequest(
    init_info={"master_address": "10.0.0.1", "master_port": 29500, ...}
)

# 更新请求（dict 被转换为后端特定的 TUpdateInfo）
update_request = WeightTransferUpdateRequest(
    update_info={"names": [...], "dtype_names": [...], "shapes": [...]}
)
```

### WeightTransferUpdateInfo

基本的 `WeightTransferUpdateInfo` 是一个用于后端特定更新信息的标记类：

```python
@dataclass
class WeightTransferUpdateInfo(ABC):
    pass
```

## 实现自定义引擎

要创建自定义权重传输后端：

### 1. 定义信息数据类

```python
from dataclasses import dataclass
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

@dataclass
class MyInitInfo(WeightTransferInitInfo):
    endpoint: str
    token: str

@dataclass
class MyUpdateInfo(WeightTransferUpdateInfo):
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    # 根据需要添加自定义字段
```

### 2. 实现引擎

```python
from collections.abc import Callable, Iterator
from typing import Any
import torch

class MyWeightTransferEngine(WeightTransferEngine[MyInitInfo, MyUpdateInfo]):
    init_info_cls = MyInitInfo
    update_info_cls = MyUpdateInfo

    def init_transfer_engine(self, init_info: MyInitInfo) -> None:
        # 使用 init_info.endpoint 等建立与训练器的连接
        ...

    def receive_weights(
        self,
        update_info: MyUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        # 接收每个权重并增量调用 load_weights
        for name, dtype_name, shape in zip(
            update_info.names, update_info.dtype_names, update_info.shapes
        ):
            dtype = getattr(torch, dtype_name)
            weight = self._fetch_weight(name, shape, dtype)
            load_weights([(name, weight)])

    def shutdown(self) -> None:
        # 清理资源
        ...

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any],
    ) -> None:
        # 从训练器进程发送权重
        for name, tensor in iterator:
            # 通过自定义传输方式发送张量
            ...
```

!!! important
    传递给 `receive_weights` 的 `load_weights` 可调用对象应该**增量地**调用（一次一个或几个权重），而不是先累积所有权重。这样可以避免大模型出现 GPU 内存不足错误。

### 3. 注册到工厂

```python
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

# 选项 1：延迟加载（推荐用于内置引擎）
WeightTransferEngineFactory.register_engine(
    "my_backend",
    "my_package.my_module",
    "MyWeightTransferEngine",
)

# 选项 2：直接注册类
WeightTransferEngineFactory.register_engine(
    "my_backend",
    MyWeightTransferEngine,
)
```

注册后，用户可以通过 `WeightTransferConfig(backend="my_backend")` 选择您的后端。

## WeightTransferEngineFactory

工厂使用带有延迟加载的注册表模式。内置引擎（`nccl` 和 `ipc`）在导入时注册，但它们的模块仅在实际请求后端时才加载。这避免了在不需要时导入重型依赖（如 NCCL 通信器）。

```python
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

# 从配置创建引擎
engine = WeightTransferEngineFactory.create_engine(
    config=weight_transfer_config,
    parallel_config=parallel_config,
)
```
