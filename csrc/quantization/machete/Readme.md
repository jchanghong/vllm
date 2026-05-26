# Machete（基于 CUTLASS 的混合精度 GEMM）

Machete 是 Marlin 内核的精神继承者，但针对 Hopper 架构进行了优化，并基于 CUTLASS。由于基于 CUTLASS，与 Marlin 相比，添加新的类型对和后处理逻辑更加容易。

## 概述

Machete 有效执行以下操作

```python
scale_type = w_s.dtype
compute_type = a.dtype
out = (w_q.to(scale_type) * w_s - w_z.to(scale_type)) @ a
```

其中 `w_q` 是量化后的权重矩阵，`w_s` 是量化缩放因子，`w_z` 是量化零点。

> **_注意：_** `w_z` 在缩放因子之后添加，以便我们可以
使用 FMA 操作，但这意味着如果提供的零点假定在应用缩放因子之前先减去它们，则它们必须预先应用缩放因子。

## API

Machete 的主要优化是将权重矩阵重新打包，以更贴合张量核心的布局，从而在加载权重矩阵时允许更宽的共享内存加载。这意味着在调用 `machete_gemm` 之前，权重矩阵必须预先打包。流程大致如下：

```python
from vllm import _custom_ops as ops

...
W_q_packed = ops.machete_prepack_B(w_q, wtype)
output = ops.machete_gemm(
    a,
    b_q=W_q_packed,
    b_type=wtype,
    b_scales=w_s,
    b_group_size=group_size
)
```

## 代码生成

由于 Machete 基于 CUTLASS，我们可以使用相同的内核模板生成多个类型对和不同的 tile 形状。我们使用 `generate.py` 生成此模板的多个实例化。

新的类型对（`TypeConfig`）可以附加到 `impl_configs`（在 `generate()` 中），这些配置将自动生成（假设它们可以无障碍地得到支持）。对于每个 `TypeConfig`，您还必须提供一个 `ImplConfig`，它将 `TypeConfig` 与 `ScheduleConfig`、`Specialization` 列表以及默认启发式函数绑定在一起。`ScheduleConfig`（包含 tile 形状、tile 调度程序等信息）对于不同的问题形状可能表现不同，几乎没有一种 `ScheduleConfig` 可以适用于所有问题形状，因此为不同的潜在问题形状生成不同的 `ScheduleConfig` 通常是有益的。这就是启发式函数的用武之地。对于每个 `TypeConfig`，应提供一个默认的启发式函数。它将不同的问题形状映射到不同的 `ScheduleConfig`，并在用户未向 `machete_gemm` 提供 `schedule` 参数时使用。`Specialization` 定义了要生成的功能组合，例如 `with_zeropoints`、`with_scales` 等。我们可以通过限制生成的功能组合集来减少编译时间和最终的二进制文件大小。
