# CUTLASS 后处理逻辑 (Epilogues)

## 介绍

本文档描述了各种 CUTLASS 后处理逻辑，用于将去量化操作融合到 GEMM 中。

目前，我们仅支持权重的对称量化，
以及激活值的对称和非对称量化。
两者都可以按张量或按通道（权重）/按 token（激活值）进行量化。

共有 4 种后处理逻辑：

1. `ScaledEpilogue`：激活值对称量化，无偏置。
1. `ScaledEpilogueBias`：激活值对称量化，支持偏置。
1. `ScaledEpilogueAzp`：激活值非对称按张量量化，支持偏置。
1. `ScaledEpilogueAzpPerToken`：激活值非对称按 token 量化，支持偏置。

我们没有为不带偏置的激活值非对称量化实现后处理逻辑，以减少最终的二进制文件大小。
相反，如果未传递偏置，后处理逻辑将使用 0 作为偏置。
这会引入一个冗余的加法操作（和运行时检查），但对性能影响很小。

## 底层线性代数

更多详情请参见[激活量化 RFC](https://github.com/vllm-project/vllm/issues/3975)。

如果 $` \widehat X `$ 是量化后的 $` X `$，我们的矩阵将变为以下形式

```math
A = s_a (\widehat A - J_a z_a)
```

```math
B = s_b \widehat B
```

```math
D = A B + C
```

```math
D = s_a s_b \widehat D + C
```

这里，D 是 GEMM 的输出，C 是偏置。
A 是激活值，支持非对称量化，
B 是权重，仅支持对称量化。
$ s_a $ 和 $s_b$ 分别是激活值和权重的缩放因子。
$ z_a $ 是激活值的零点，$ J_a $ 是与 A 维度相同的全 1 矩阵。
需要额外的后处理逻辑来支持权重的非对称量化。

进一步展开，我们可以计算 $` \widehat D `$ 如下：

```math
A B = s_a ( \widehat A - J_a z_a ) s_b \widehat B
```

```math
A B = s_a s_b \left( \widehat A \widehat B - J_a z_a \widehat B \right)
```

```math
\widehat D = \widehat A \widehat B - z_a J_a \widehat B
```

注意 $` \widehat A \widehat B `$ 是 GEMM 的原始输出，
而 $` J_a \widehat B `$ 可以预先计算。
它的每一行等于 $` \mathbf 1 \widehat B `$，即 $` \widehat B `$ 列和的行向量。

## 后处理逻辑

### `ScaledEpilogue`

此后处理逻辑计算无偏置的激活值对称量化，即 $` C = 0 `$ 且 $` z_a = 0 `$。
GEMM 的输出为：

```math
\widehat D = \widehat A \widehat B
```

```math
D = s_a s_b \widehat D
```

```math
D = s_a s_b \widehat A \widehat B
```

后处理逻辑参数：

- `scale_a` 是激活值的缩放因子，可以是按张量（标量）或按 token（列向量）。
- `scale_b` 是权重的缩放因子，可以是按张量（标量）或按通道（行向量）。

### `ScaledEpilogueBias`

此后处理逻辑计算带偏置的激活值对称量化，即 $` z_a = 0 `$。
GEMM 的输出为：

```math
\widehat D = \widehat A \widehat B
```

```math
D = s_a s_b \widehat D + C 
```

```math
D = s_a s_b \widehat A \widehat B + C
```

后处理逻辑参数：

- `scale_a` 是激活值的缩放因子，可以是按张量（标量）或按 token（列向量）。
- `scale_b` 是权重的缩放因子，可以是按张量（标量）或按通道（行向量）。
- `bias` 是偏置，始终按通道（行向量）。

### `ScaledEpilogueAzp`

此后处理逻辑计算带偏置的激活值非对称按张量量化。
GEMM 的输出为：

```math
\widehat D = \widehat A \widehat B - z_a J_a \widehat B
```

```math
D = s_a s_b \widehat D + C 
```

```math
D = s_a s_b \left( \widehat A \widehat B - z_a J_a \widehat B \right) + C
```

因为 $` z_a `$ 是一个标量，零点项 $` z_a J_a \widehat B `$ 的每一行都等于 $` z_a \mathbf 1 B `$。
该项被预先计算并作为行向量存储在 `azp_with_adj` 中。

后处理逻辑参数：

- `scale_a` 是激活值的缩放因子，可以是按张量（标量）或按 token（列向量）。
    - 通常由于零点是按张量的，因此这也会是按张量的。
- `scale_b` 是权重的缩放因子，可以是按张量（标量）或按通道（行向量）。
- `azp_with_adj` 是预先计算的零点项（$` z_a J_a \widehat B `$），是按通道的（行向量）。
- `bias` 是偏置，始终按通道（行向量）。

为了高效使用这些内核，用户必须离线预计算 `azp_with_adj` 项并将其传递给内核。

### `ScaledEpilogueAzpPerToken`

此后处理逻辑计算带偏置的激活值非对称按 token 量化。

GEMM 的输出与上述相同，但 $` z_a `$ 是一个列向量。
这意味着零点项 $` z_a J_a \widehat B `$ 变为 $` z_a `$ 和 $` \mathbf 1 \widehat B `$ 的外积。

后处理逻辑参数：

- `scale_a` 是激活值的缩放因子，可以是按张量（标量）或按 token（列向量）。
    - 通常由于零点是按 token 的，因此这也会是按 token 的。
- `scale_b` 是权重的缩放因子，可以是按张量（标量）或按通道（行向量）。
- `azp_adj` 是预先计算的零点调整项（$` \mathbf 1 \widehat B `$），是按通道的（行向量）。
- `azp` 是零点（`z_a`），是按 token 的（列向量）。
- `bias` 是偏置，始终按通道（行向量）。

为了高效使用这些内核，用户必须离线预计算 `azp_adj` 项并将其传递给内核。

后处理逻辑执行以下计算（其中 `Dq` 是 GEMM 的原始量化输出）：

```math
out = scale_a * scale_b * (Dq - azp_adj * azp) + bias
```
