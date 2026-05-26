# Fusion torch.compile 传递

vLLM 在编译时（通过自定义的 [`torch.compile`](torch_compile.md) Inductor 传递）应用一组内核/算子融合，以将优化与模型定义分离，并避免破坏模型代码中的层抽象。这些融合由 [`PassConfig`][vllm.config.compilation.PassConfig] 中的字段控制，并在适当的[优化级别](optimization_levels.md)下自动启用。

## 快速参考

下表将每个融合映射到其控制标志/配置旋钮、融合的操作、默认启用的级别以及指示性的加速比。Fullgraph 列表示该融合是否需要整个模型图可见（通过 Inductor 分区或 `splitting_ops=[]`），最后一列表示该融合是在所有 `num_tokens` 下激活，还是仅在低端或高端激活。

!!! info
    加速比严重依赖于具体的模型、批量大小和硬件。如果手动调优性能，请始终在有和无该融合的情况下对您的确切用例进行基准测试，以验证影响。

| 融合 | `PassConfig` 标志 | 融合的操作 | 默认级别 | 端到端加速比 | Fullgraph | `num_tokens` |
| - | - | - | - | - | - | - |
| [AllReduce + RMSNorm](#allreduce--rmsnorm-fuse_allreduce_rms) | `fuse_allreduce_rms` | 全规约 → RMSNorm（+残差相加）（→ 量化） | O2（Hopper/Blackwell + TP > 1） | 5-20% | 否 | 低 |
| [MiniMax QK Norm](#minimax-qk-norm-fuse_minimax_qk_norm) | `fuse_minimax_qk_norm` | Q/K 方差全规约 → Q/K RMSNorm | 默认关闭 | 2-3% | 否 | 低 |
| [Attention + Quant](#attention--quantization-fuse_attn_quant) | `fuse_attn_quant` | 注意力输出 → FP8/NVFP4 量化 | 默认关闭 | 3-7% | 是 | 始终 |
| [MLA Attention + Quant](#attention--quantization-fuse_attn_quant) | `fuse_attn_quant` | MLA 注意力输出 → FP8/NVFP4 量化 | 默认关闭 | 待定 | 是 | 始终 |
| [RoPE + KV-Cache Update](#rope--kv-cache-update-fuse_rope_kvcache) | `fuse_rope_kvcache` | 旋转嵌入 → KV 缓存写入 | O2（仅 ROCm/AITER） | 2-4% | 否 | 低 |
| [QK Norm + RoPE](#qk-norm--rope-enable_qk_norm_rope_fusion) | `enable_qk_norm_rope_fusion` | Q/K RMSNorm → 旋转嵌入 | 默认关闭 | 2-3% | 否 | 低 |
| [Sequence Parallelism](#sequence-parallelism-enable_sp) | `enable_sp` | AllReduce → ReduceScatter + AllGather | 默认关闭 | AsyncTP 的前提条件 | 是 | 高 |
| [AsyncTP GEMM + collective](#asynctp-gemm--collective-overlap-fuse_gemm_comms) | `fuse_gemm_comms` | GEMM → reduce-scatter / all-gather → GEMM | 默认关闭 | 7-10% | 是 | 高 |
| [RMSNorm + Quant](#rmsnorm--quantization-fuse_norm_quant) | `fuse_norm_quant` | RMSNorm（+残差相加）→ FP8/FP4 量化 | O1（条件性） | 1-4% | 否 | 始终 |
| [SiLU+Mul + Quant](#silumul--quantization-fuse_act_quant) | `fuse_act_quant` | SiLU+Mul 激活 → FP8/FP4 量化 | O1（条件性） | 1-4% | 否 | 始终 |
| [RMSNorm + Padding](#rmsnorm--padding-fuse_act_padding) | `fuse_act_padding` | 残差相加 + RMSNorm → 填充 | O1（仅 ROCm/AITER） | 待定 | 否 | 始终 |
| [MLA Dual RMSNorm](#mla-dual-rmsnorm-fuse_mla_dual_rms_norm) | `fuse_mla_dual_rms_norm` | 配对的 Q + KV RMSNorm → 单内核 | O1（仅 ROCm/AITER） | ~2% | 否 | 始终 |

## 支持矩阵

下表列出了每个平台上每个融合支持的量化方案。**—** 表示该平台不支持该融合。最新及正在进行的工作请参见追踪问题：[#36066](https://github.com/vllm-project/vllm/issues/36066)

| 融合 | SM100（Blackwell） | SM90（Hopper） | SM89（Ada） | SM80（Ampere） | ROCm |
| - | - | - | - | - | - |
| `fuse_allreduce_rms` | FP16/BF16、FP8 static、NVFP4 | FP16/BF16、FP8 static | — | — | — |
| `fuse_minimax_qk_norm`\* | FP16/BF16 | FP16/BF16 | FP16/BF16 | FP16/BF16 | — |
| `fuse_attn_quant`\* | FP8 static\*、NVFP4\* | FP8 static\* | FP8 static\* | — | FP8 static\* |
| `fuse_attn_quant`（MLA）\* | FP8 static\*、FP8 per-group\*、NVFP4\* | FP8 static\*、FP8 per-group\* | FP8 static\*、FP8 per-group\* | — | FP8 static\*（未测试） |
| `fuse_rope_kvcache` | — | — | — | — | FP16/BF16 |
| `enable_qk_norm_rope_fusion` | FP16/BF16 | FP16/BF16 | FP16/BF16† | FP16/BF16† | — |
| `enable_sp` | FP16/BF16、FP8 static† | FP16/BF16、FP8 static | FP16/BF16† | FP16/BF16† | — |
| `fuse_gemm_comms` | FP16/BF16、FP8 static† | FP16/BF16、FP8 static | FP16/BF16† | FP16/BF16† | — |
| `fuse_norm_quant` | FP8 static、FP8 per-token、FP8 per-group | FP8 static、FP8 per-token、FP8 per-group | FP8 static、FP8 per-token、FP8 per-group | — | FP8 static、FP8 per-token、FP8 per-group |
| `fuse_act_quant` | FP8 static、NVFP4 | FP8 static、FP8 per-group（128/64） | FP8 static、FP8 per-group（128/64） | — | FP8 per-group |
| `fuse_act_padding` | — | — | — | — | FP16/BF16 |
| `fuse_mla_dual_rms_norm` | — | — | — | — | BF16 |

\* `fuse_attn_quant` 的支持取决于所使用的注意力后端；并非所有后端都支持融合量化输出。有关每个后端的详细信息，请参见 [`fuse_attn_quant` 部分](#attention--quantization-fuse_attn_quant)。

\* `fuse_minimax_qk_norm` 是特定于 `MiniMaxM2ForCausalLM` 模型的传递。它还需要张量并行（`tp_size > 1`）和 CUDA 自定义算子 `minimax_allreduce_rms_qk`。

† `enable_sp` 和 `fuse_gemm_comms` 目前仅在 SM90 上自动配置；其他架构支持需要显式设置 `PassConfig.sp_min_token_num`。SM100 支持还需要设置 `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`。

## 启用/禁用融合

融合通过 `PassConfig` 暴露，该配置嵌套在 `CompilationConfig` 中：

```python
from vllm import LLM
from vllm.config import CompilationConfig, PassConfig

llm = LLM(
    model="...",
    optimization_level=2, # 默认优化级别
    compilation_config=CompilationConfig(
        pass_config=PassConfig(
            fuse_norm_quant=True,
            fuse_act_quant=True,
            fuse_allreduce_rms=False,  # 禁用特定融合
        )
    ),
)
```

融合也可以通过命令行标志在任何 `vllm ...` 命令中启用：

```bash
# 启用 O2 默认值，但关闭 allreduce 融合
vllm serve meta-llama/Llama-3.1-8B-Instruct -O2 -cc.pass_config.fuse_allreduce_rms=False

# 以上等价于更详细的写法：
vllm serve meta-llama/Llama-3.1-8B-Instruct -O2 --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'

# 其他命令中使用相同语法，例如 vllm bench：
vllm bench latency --model=meta-llama/Llama-3.1-8B-Instruct -O2 -cc.pass_config.fuse_allreduce_rms=False
```

用户显式设置的字段始终优先于优化级别默认值。

## 融合详情

### AllReduce + RMSNorm（`fuse_allreduce_rms`）

!!! warning
    TP+DP 和 TP+PP 组合目前存在问题
    （[#34458](https://github.com/vllm-project/vllm/issues/34458) 和
    [#35426](https://github.com/vllm-project/vllm/issues/35426)）。
    仅在安装了 FlashInfer 的 NVIDIA Hopper（SM90）和 Blackwell（SM100）上支持。

**融合内容。** 将张量并行全规约集合通信与后续的残差相加、RMSNorm 以及可选的量化步骤融合为单个 FlashInfer / TRT-LLM 通信内核。
此融合仅对较小的 `num_tokens` 有益，因此仅在较低的编译范围内执行。

覆盖的模式：

- `AllReduce → RMSNorm(+residual_add)`：CUDA sm90+ 与 FlashInfer
- `AllReduce → RMSNorm(+residual_add) → FP8 static quant`：CUDA sm90+ 与 FlashInfer
- `AllReduce → RMSNorm(+residual_add) → NVFP4 dynamic quant`：CUDA sm100+ 与 FlashInfer

使用融合内核的张量大小上限取决于硬件（SM90/SM100 上 TP=2 时为 64 MB），可通过 `PassConfig.fi_allreduce_fusion_max_size_mb` 配置。

**代码位置。**

- 传递：[`vllm/compilation/passes/fusion/allreduce_rms_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/allreduce_rms_fusion.py)
- FlashInfer 全规约：[`vllm/distributed/device_communicators/flashinfer_all_reduce.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/flashinfer_all_reduce.py)
- 基准测试：[`benchmarks/kernels/benchmark_fused_collective.py`](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_fused_collective.py)

### Attention + Quantization（`fuse_attn_quant`）

!!! info
    `fuse_attn_quant` 目前默认不在任何优化级别启用，必须显式设置。它需要整个模型图可见（Inductor 分区或 `splitting_ops=[]`）。

**融合内容。** 将注意力输出量化直接在注意力计算之后融合，消除了注意力输出的全精度内存往返。此融合支持标准 `Attention` 和 `MLAAttention`（用于 DeepSeek-V2/V3/R1 模型）。覆盖的模式：

`Attention → FP8 static quant`：

- `TRITON_ATTN`：CUDA、ROCm
- `FLASHINFER`：安装了 FlashInfer 的 CUDA sm100+
- `ROCM_ATTN`：ROCm
- `ROCM_AITER_UNIFIED_ATTN`：带 AITER 的 ROCm

`Attention → NVFP4 dynamic quant`：

- `FLASHINFER`：安装了 FlashInfer 的 CUDA sm100+

`MLAAttention → FP8 static、FP8 per-group、NVFP4 dynamic quant`

MLA 融合在图级别对 `unified_mla_attention_with_output` 算子进行操作，适用于所有 MLA 解码和预填充后端组合。与标准 `Attention` 后端（内核直接写入 FP8 输出）不同，目前没有 MLA 预填充或解码后端支持直接的 FP8/FP4 输出。融合写入中间缓冲区并在单独的步骤中量化，因此尚未消除内存往返。

!!! info
    MLA 注意力融合预计不会产生可测量的加速。一旦 MLA 预填充/解码内核支持直接的 FP8/FP4 输出，这种情况将得到改善。

其他注意力后端尚不支持融合输出量化。

**代码位置。**

- 传递（Attention）：[`vllm/compilation/passes/fusion/attn_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/attn_quant_fusion.py)
- 传递（MLAAttention）：[`vllm/compilation/passes/fusion/mla_attn_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/mla_attn_quant_fusion.py)
- 注意力后端：[`vllm/v1/attention/backends/`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/)

### RoPE + KV-Cache Update（`fuse_rope_kvcache`）

!!! info
    仅 ROCm/AITER。在 NVIDIA CUDA 或 CPU 上不可用。由于 AITER 融合内核性能问题，该融合默认仅在 `num_tokens ≤ 256` 时启用。此阈值可通过 `PassConfig.rope_kvcache_fusion_max_token_num` 配置。

**融合内容。** 将旋转位置嵌入内核与 KV 缓存分散/写入融合为单个内核，避免了对键和值张量的单独读取和写入。

需要：启用 AITER 的 AMD ROCm、激活的 `rotary_embedding` 自定义算子（自动），以及在图中可见的 `kv_cache` 更新操作：通过使用 Inductor 图分区或从 `splitting_ops` 中移除。如果满足这些条件，融合会在优化级别 O1 及以上自动启用。

**代码位置。**

- 传递：[`vllm/compilation/passes/fusion/rope_kvcache_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rope_kvcache_fusion.py)

### MiniMax QK Norm（`fuse_minimax_qk_norm`）

!!! info
    这是特定于 MiniMax 的编译传递。目前仅在以下所有条件满足时启用：模型架构为 `MiniMaxM2ForCausalLM`、启用了张量并行（`tp_size > 1`）、且 CUDA 自定义算子 `minimax_allreduce_rms_qk` 可用。默认不在任何优化级别启用。

**融合内容。** 融合 MiniMax M2 的 Q/K 归一化路径，该路径在对 Q 和 K 应用 RMS 归一化之前，对每个 token 的 Q/K 方差执行全规约。

此传递与 [`enable_qk_norm_rope_fusion`](#qk-norm--rope-enable_qk_norm_rope_fusion) 不同：`fuse_minimax_qk_norm` 针对 MiniMax M2 的张量并行全规约 + RMSNorm 序列，而 `enable_qk_norm_rope_fusion` 针对多个其他模型使用的后续 Q/K RMSNorm + RoPE 序列。

示例：

```bash
vllm serve MiniMaxAI/MiniMax-M2.5 \
  --tensor-parallel-size 4 \
  --compilation-config '{"mode": 3, "pass_config": {"fuse_minimax_qk_norm": true}}'
```

**代码位置。**

- 传递：[`vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/minimax_qk_norm_fusion.py)
- CUDA 算子：[`csrc/minimax_reduce_rms_kernel.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/minimax_reduce_rms_kernel.cu)（`minimax_allreduce_rms_qk`）
- 工作空间辅助：[`vllm/model_executor/layers/mamba/lamport_workspace.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/lamport_workspace.py)

### Sequence Parallelism（`enable_sp`）

**融合内容。** 用 reduce-scatter + 本地 RMSNorm + all-gather 替换全规约集合通信，将序列维度分割到各 TP 等级。这重构了图，使得后续的 AsyncTP 传递可以将 reduce-scatter / all-gather 与周围的 GEMM 融合。

序列并行本身并不直接提升性能；它是 AsyncTP 传递（`fuse_gemm_comms`）的前提条件。SP 仅在某个最小 token 阈值以上应用，该阈值根据设备能力和模型 `hidden_size` 自动配置。目前仅在 H100/SM90 上对 `hidden_size >= 8192` 的模型激活。阈值可通过 `PassConfig.sp_min_token_num` 配置。

一般变换：

```text
输入 → AllReduce → RMSNorm → 输出
变为：
输入 → ReduceScatter → 本地 RMSNorm → AllGather → 输出
```

覆盖的模式：

- 第一个块：`AllReduce → RMSNorm` → `ReduceScatter → RMSNorm → AllGather`
- 中间块：`AllReduce → fused_add_RMSNorm` → `ReduceScatter → fused_add_RMSNorm → AllGather`
- 两者均可选地带有 `→ FP8 static quant` 后缀

需要：`use_inductor_graph_partition=True` **或** 使用可被 `tensor_parallel_size` 整除的静态大小的逐段编译。

支持的硬件：仅在 NVIDIA CUDA 上测试，可能在 ROCm 上有效。FP8 all-gather 需要 sm90+。

**代码位置。**

- 传递：[`vllm/compilation/passes/fusion/sequence_parallelism.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/sequence_parallelism.py)

### AsyncTP GEMM + Collective Overlap（`fuse_gemm_comms`）

!!! info
    需要 `enable_sp=True`（自动启用）。如果未应用序列并行，此传递为无操作。

**融合内容。** 在序列并行变换图之后，使用 `torch.ops.symm_mem` 对称内存原语将 GEMM 内核与周围的 reduce-scatter（输出投影）和 all-gather（输入投影）融合，使通信和计算重叠。这种重叠仅对较大的 `num_tokens` 有益，因此融合（及其前置的 SP）仅在高于 `PassConfig.sp_min_token_num` 的较高编译范围内执行。

覆盖的模式：

- `GEMM → reduce-scatter` → `fused_matmul_reduce_scatter`
- `all-gather → GEMM` → `all_gather_matmul`
- 两种模式的 FP8 scaled 变体

支持的硬件：支持对称内存（`torch.distributed._symmetric_memory`）的 NVIDIA CUDA。

在 B200 上，不支持 FP8 FlashInfer scaled MM 的模式匹配，因此必须禁用它
（[#27893](https://github.com/vllm-project/vllm/issues/27893)）

```shell
VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel ...
```

**代码位置。**

- 传递：[`vllm/compilation/passes/fusion/collective_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/collective_fusion.py)
- 序列并行传递：[`vllm/compilation/passes/fusion/sequence_parallelism.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/sequence_parallelism.py)

### QK Norm + RoPE（`enable_qk_norm_rope_fusion`）

!!! info
    仅适用于在旋转位置嵌入前对 Q 和 K 应用逐头 RMSNorm 的模型（例如 Qwen）。由于 H100 上的性能问题，默认不在任何优化级别启用：[#34391](https://github.com/vllm-project/vllm/issues/34391)

**融合内容。** 将以下序列融合为单个 `fused_qk_norm_rope` CUDA 内核：拆分 QKV → 重塑 → Q/K RMSNorm → 重塑 → 旋转嵌入。

```text
# 未融合：
q, k, v = split(qkv)
q_norm = rms_norm(q.view(heads))
k_norm = rms_norm(k.view(kv_heads))
q_rope, k_rope = rotary_embedding(q_norm, k_norm, ...)

# 融合后：
fused_qk_norm_rope(qkv, ...)
```

支持的硬件：仅 CUDA（sm80+），仅在 sm90 和 sm100 上测试。

**代码位置。**

- 传递：[`vllm/compilation/passes/fusion/qk_norm_rope_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py)
- CUDA 内核：[`csrc/ops.h`](https://github.com/vllm-project/vllm/blob/main/csrc/ops.h)（`fused_qk_norm_rope`）

### RMSNorm + Quantization（`fuse_norm_quant`）

!!! warning
    在 NVIDIA 上，Inductor 实际生成的融合内核比我们的自定义 CUDA 内核更快。因此，此融合仅在 `rms_norm` 或 `quant_fp8` 使用自定义内核时启用。

**融合内容。** 将自定义的 `rms_norm` / `fused_add_rms_norm` 操作与后续的量化合并为单个融合内核，消除了中间的全精度激活张量的读取/写入。融合两种变体：

- *Plain RMSNorm + quant*：`rms_norm(x) → quant_fp8(y)`
- *Fused-add RMSNorm + quant*：`fused_add_rms_norm(x, residual) → quant_fp8(y)` — 同时就地更新残差。

注意，AITER 融合目前位于 `vllm.compilation.passes.fusion.rocm_aiter_fusion` 中的单独传递中。

支持的量化方案/硬件组合：

- FP8 static per-tensor：CUDA & HIP 内核
- FP8 dynamic per-token：CUDA & HIP 内核、AITER
- FP8 dynamic per-token-group（128/64）：CUDA & HIP 内核、AITER

**代码位置。**

- 传递：[`vllm/compilation/passes/fusion/rms_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rms_quant_fusion.py)
- ROCm AITER 传递：[`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)
- CUDA/HIP 内核：[`csrc/layernorm_quant_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_quant_kernels.cu)

### SiLU+Mul + Quantization（`fuse_act_quant`）

!!! warning
    与 `fuse_norm_quant` 相同：在 NVIDIA 上，Inductor 生成的融合内核比我们的自定义算子更快。此融合仅在 `silu_and_mul` 或 `quant_fp8` 使用自定义内核时启用，或用于 NVFP4 量化模型（其中 FP4 量化始终是自定义算子）。

**融合内容。** 将 `silu_and_mul` 门控上投影激活与后续量化融合为单个内核，避免了全精度后激活张量的具体化。

注意，AITER 融合位于 `vllm.compilation.passes.fusion.rocm_aiter_fusion` 中的单独传递中。

支持的量化方案/硬件组合：

- FP8 static per-tensor：CUDA & HIP 内核
- FP8 dynamic per-group（128/64）：CUDA 内核（sm89+，在 sm100+ 上使用 DeepGemm 时不激活）
- NVFP4 dynamic：仅 CUDA sm100+ 与 FlashInfer
- FP8 per-token-group（128）：仅 ROCm AITER

**代码位置。**

- 传递：[`vllm/compilation/passes/fusion/act_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/act_quant_fusion.py)
- ROCm AITER 传递：[`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)
- CUDA/HIP 内核：[`csrc/quantization/`](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/)
- 融合的 SiLU+Mul+BlockQuant 内核：[`csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu)

### RMSNorm + Padding（`fuse_act_padding`）

!!! info
    仅 ROCm/AITER。针对 GPT-OSS 模型。

**融合内容。** 将残差相加 + RMSNorm 与后续的填充操作融合，该填充操作将隐藏维度填充为下游 AITER Triton GEMM 内核所需的倍数。

需要：启用 AITER RMSNorm 的 AMD ROCm。当隐藏大小为 2880 且 AITER Triton GEMM **未**启用时，默认在优化级别 O1 及以上启用。

**代码位置。**

- 传递：[`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)（`RocmAiterTritonAddRMSNormPadFusionPass`）

### MLA Dual RMSNorm（`fuse_mla_dual_rms_norm`）

!!! info
    仅 ROCm/AITER。针对 DeepSeek-V3 / Kimi-K2 MLA 注意力。

!!! note
    当使用 `rms_norm` 的原生实现（目前 CUDA 和 ROCm 上的默认设置）时，Inductor 的内置融合已自动处理这些范数的合并。此显式传递针对 AITER 的自定义 `rms_norm` 算子激活的情况，Inductor 无法自行融合该算子。

**融合内容。** 将 MLA 注意力中配对的 `q_a_layernorm` 和 `kv_a_layernorm` RMS 归一化操作融合为单个通过 AITER 的 `fused_qk_rmsnorm` HIP 内核调用，将每个 MLA 层的内核启动开销从 2 次减少到 1 次。

```text
# 未融合：
q_c, kv_lora = split(projected, [q_dim, kv_dim])
kv_c, k_pe   = split(kv_lora,  [kv_c_dim, k_pe_dim])
q_c  = rms_norm(q_c,  q_weight,  eps)
kv_c = rms_norm(kv_c, kv_weight, eps)

# 融合后：
q_c, kv_lora = split(projected, [q_dim, kv_dim])
kv_c, k_pe   = split(kv_lora,  [kv_c_dim, k_pe_dim])
q_normed, kv_normed = fused_mla_dual_rms_norm(
    q_c, q_weight, kv_c, kv_weight, eps1, eps2)
```

需要：启用 AITER 的 AMD ROCm。当 AITER 可用时，默认在优化级别 O1 及以上启用。

**代码位置。**

- 传递：[`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)（`MLADualRMSNormFusionPass`）
- 自定义算子：[`vllm/_aiter_ops.py`](https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py)（`fused_mla_dual_rms_norm`）
- AITER 内核：[`fused_qk_rmsnorm`](https://github.com/ROCm/aiter/pull/2442)

## 另请参阅

- [优化级别](optimization_levels.md) — 设置融合默认值的高级预设。
- [vLLM 中的 torch.compile](torch_compile.md) — Inductor 传递管道的工作原理。
- [注意力后端](attention_backends.md) — 特定于注意力的内核选择。
