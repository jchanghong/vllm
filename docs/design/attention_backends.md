# 注意力后端功能支持

本文档由 `tools/pre_commit/generate_attention_backend_docs.py` 自动生成。
它展示了每个已注册的注意力后端基于 `AttentionBackend.validate_configuration()` 中的检查所支持的功能。

**请勿手动编辑此文件。** 运行以下命令重新生成：

```bash
python tools/pre_commit/generate_attention_backend_docs.py
```

## 设置注意力后端

### 命令行

从命令行指定后端有两种方式：

**方式 1：使用 `--attention-backend`（简单）**

```bash
vllm serve <model> --attention-backend FLASH_ATTN
```

**方式 2：使用 `--attention-config.backend` / `-ac.backend`（结构化配置）**

```bash
# 点号表示法
vllm serve <model> --attention-config.backend FLASH_ATTN
vllm serve <model> -ac.backend FLASH_ATTN

# JSON 格式
vllm serve <model> --attention-config '{"backend": "FLASH_ATTN"}'
vllm serve <model> -ac '{"backend": "FLASH_ATTN"}'
```

> **注意：** `--attention-backend` 和 `--attention-config.backend` 是互斥的。请使用其中之一，不要同时使用。

### Python API

使用 `AttentionConfig` 配合 `LLM` 类：

```python
from vllm import LLM
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# 方法 1：使用 AttentionConfig 配合枚举
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN),
)

# 方法 2：使用 attention_backend 参数配合字符串
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    attention_backend="FLASH_ATTN",
)
```

## 后端选择行为

### 手动选择

当您通过 `--attention-backend` 或 `AttentionConfig` 显式设置后端时：

1. 后端会根据您的配置（模型 dtype、head 大小、计算能力等）进行**验证**
2. 如果后端**不支持**您的配置，将抛出错误并说明具体原因
3. 如果验证通过，则使用该后端

选择不兼容后端时的错误示例：

```text
ValueError: Selected backend FLASHMLA is not valid for this configuration.
Reason: ['compute capability not supported']
```

### 自动选择

当未指定后端时（默认情况）：

1. vLLM 按**优先级顺序**遍历后端（见下方表格）
2. 每个后端都会根据您的配置进行验证
3. 选择**第一个兼容的后端**
4. 如果没有后端兼容，将抛出错误，列出所有后端及其不兼容原因

## 后端优先级（CUDA）

当未显式选择后端时，vLLM 会从这些按优先级排序的列表中选择第一个兼容的后端。

优先级 **1 = 最高**（优先尝试）。

### 标准注意力（MHA、MQA、GQA）

**Blackwell（SM 10.x）：**

| 优先级 | 后端 |
| -------- | ------- |
| 1 | `FLASHINFER` |
| 2 | `FLASH_ATTN` |
| 3 | `TRITON_ATTN` |
| 4 | `FLEX_ATTENTION` |
| 5 | `TURBOQUANT` |

**Ampere/Hopper（SM 8.x-9.x）：**

| 优先级 | 后端 |
| -------- | ------- |
| 1 | `FLASH_ATTN` |
| 2 | `FLASHINFER` |
| 3 | `TRITON_ATTN` |
| 4 | `FLEX_ATTENTION` |
| 5 | `TURBOQUANT` |

### MLA 注意力（DeepSeek 风格）

**Blackwell（SM 10.x）：**

| 优先级 | 后端 |
| -------- | ------- |
| 1 | `FLASHINFER_MLA` |
| 2 | `TOKENSPEED_MLA` |
| 3 | `CUTLASS_MLA` |
| 4 | `FLASH_ATTN_MLA` |
| 5 | `FLASHMLA` |
| 6 | `TRITON_MLA` |
| 7 | `FLASHINFER_MLA_SPARSE`**\*** |
| 8 | `FLASHMLA_SPARSE` |

**Ampere/Hopper（SM 8.x-9.x）：**

| 优先级 | 后端 |
| -------- | ------- |
| 1 | `FLASH_ATTN_MLA` |
| 2 | `FLASHMLA` |
| 3 | `FLASHINFER_MLA` |
| 4 | `TRITON_MLA` |
| 5 | `FLASHMLA_SPARSE` |

> **\*** 对于稀疏 MLA，FP8 KV cache 始终优先选择 `FLASHINFER_MLA_SPARSE`。使用 BF16 KV cache 时，对于低 query-head 数量（<= 16），优先选择 `FLASHINFER_MLA_SPARSE`，否则优先选择 `FLASHMLA_SPARSE`。
>
> **注意：** ROCm 和 CPU 平台有自己的选择逻辑。详见平台相关文档。

## 图例

| 列 | 描述 |
| ------ | ----------- |
| **Dtypes** | 支持的模型数据类型（fp16、bf16、fp32） |
| **KV Dtypes** | 支持的 KV cache 数据类型（`auto`、`fp8`、`fp8_e4m3` 等） |
| **Block Sizes** | 支持的 KV cache 块大小（%N 表示 N 的倍数） |
| **Head Sizes** | 支持的注意力 head 大小 |
| **Sink** | 注意力下沉支持（用于 StreamingLLM） |
| **Non-Causal** | 解码器模型的非因果（双向）注意力支持 |
| **Sparse** | 稀疏注意力支持（仅 MLA） |
| **MM Prefix** | 多模态前缀全注意力支持 |
| **DCP** | 解码上下文并行支持（`--decode-context-parallel-size`） |
| **Attention Types** | 支持的注意力模式（Decoder、Encoder、Enc-Dec） |
| **Compute Cap.** | 所需的 CUDA 计算能力（非 CUDA 后端为 N/A） |

**符号：** ✅ = 支持，❌ = 不支持

## 标准注意力（MHA、MQA、GQA）后端

| 后端 | 版本 | Dtypes | KV Dtypes | Block Sizes | Head Sizes | Sink | Non-Causal | MM Prefix | DCP | Attention Types | Compute Cap. |
| ------- | ------- | ------ | --------- | ----------- | ---------- | ---- | ---------- | --------- | --- | --------------- | ------------ |
| `CPU_ATTN` | | fp16, bf16, fp32 | `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | %16 | 32, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512 | ❌ | ❌ | ❌ | ❌ | All | N/A |
| `FLASHINFER` | Native† | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | 16, 32, 64 | 64, 128, 256, 512 | ❌ | ❌ | ❌ | ✅ | Decoder | 7.x-9.x |
| `FLASHINFER` | TRTLLM† | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `nvfp4` | 16, 32, 64 | 64, 128, 256, 512 | ✅ | ❌ | ❌ | ✅ | Decoder | 10.x |
| `FLASH_ATTN` | FA2* | fp16, bf16 | `auto`, `float16`, `bfloat16` | %16 | Any | ❌ | ✅ | ❌ | ✅ | All | ≥8.0 |
| `FLASH_ATTN` | FA3* | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | %16 | Any | ✅ | ✅ | ❌ | ✅ | All | 9.x |
| `FLASH_ATTN` | FA4* | fp16, bf16 | `auto`, `float16`, `bfloat16` | %16 | Any | ✅ | ✅ | ❌ | ✅ | All | ≥10.0 |
| `FLASH_ATTN_DIFFKV` | | fp16, bf16 | `auto` | Any | Any | ❌ | ❌ | ❌ | ✅ | Decoder | Any |
| `FLEX_ATTENTION` | | fp16, bf16, fp32 | `auto`, `float16`, `bfloat16` | %16 | Any | ❌ | ✅ | ✅ | ❌ | Decoder, Encoder Only | Any |
| `ROCM_AITER_FA` | | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | 16, 32 | 64, 128, 256 | ❌ | ✅ | ❌ | ❌ | Decoder | N/A |
| `ROCM_AITER_UNIFIED_ATTN` | | fp16, bf16 | `auto` | %16 | Any | ✅ | ❌ | ✅ | ❌ | All | N/A |
| `ROCM_ATTN` | | fp16, bf16, fp32 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | %16 | 32, 64, 80, 96, 128, 160, 192, 224, 256 | ❌ | ✅ | ✅ | ❌ | Decoder, Encoder, Encoder Only | N/A |
| `TRITON_ATTN` | | fp16, bf16, fp32 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `int8_per_token_head`, `fp8_per_token_head` | %16 | Any | ✅ | ❌ | ✅ | ❌ | All | Any |
| `TURBOQUANT` | | fp16, bf16 | `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc` | 16, 32, 64, 128 | Any | ❌ | ❌ | ❌ | ❌ | Decoder | Any |

> **†** FlashInfer 在 Blackwell（SM100）上使用 TRTLLM 注意力，支持下沉。通过 `--attention-config.use_trtllm_attention=0` 禁用。
>
> **\*** 通过 `--attention-config.flash_attn_version=2`、`3` 或 `4` 指定 FlashAttention 版本。默认在 SM100+（Blackwell）上为 FA4，SM90（Hopper）上为 FA3，其他情况下为 FA2。

## MLA（多头潜在注意力）后端

MLA 对预填充和解码阶段使用不同的后端。

### 预填充后端

要显式选择预填充后端，使用
`-ac.mla_prefill_backend=<BACKEND>`（例如 `FLASH_ATTN`、`FLASHINFER`）。
否则，预填充后端会在运行时根据硬件和配置自动选择。

| 后端 | 描述 | Dtypes | Compute Cap. | 备注 |
| ------- | ----------- | ------ | ------------ | ----- |
| `FLASH_ATTN`‡ | FlashAttention varlen（FA2/FA3/FA4） | fp16, bf16 | Any | SM100+ 上为 FA4，SM90 上为 FA3，其他为 FA2 |
| `TRTLLM_RAGGED` | TensorRT-LLM ragged attention | fp16, bf16 | 10.x | 仅 DeepSeek R1 维度 |
| `FLASHINFER` | FlashInfer CUTLASS 后端 | fp16, bf16 | 10.x | 仅 DeepSeek R1 维度 |
| `TOKENSPEED_MLA` | | fp16, bf16 | 10.x | 仅 DeepSeek R1 维度 |

> **‡** TRT-LLM Ragged 是 Blackwell（SM100）上的默认选项。
> 在其他 GPU 上，默认使用 FlashAttention。

### 解码后端

MLA 解码后端使用标准的
`-ac.backend=<BACKEND>` 参数选择（例如 `FLASHMLA`、`TRITON_MLA`）。

| 后端 | Dtypes | KV Dtypes | Block Sizes | Head Sizes | Sink | Non-Causal | Sparse | MM Prefix | DCP | Attention Types | Compute Cap. |
| ------- | ------ | --------- | ----------- | ---------- | ---- | ---------- | ------ | --------- | --- | --------------- | ------------ |
| `CUTLASS_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 128 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 10.x |
| `FLASHINFER_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 32, 64 | Any | ❌ | ❌ | ❌ | ❌ | ❌ | Decoder | 10.x |
| `FLASHINFER_MLA_SPARSE` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 32, 64 | 576 | ❌ | ❌ | ✅ | ❌ | ❌ | Decoder | 10.x |
| `FLASHMLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 64 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 9.x-10.x |
| `FLASHMLA_SPARSE` | bf16 | `auto`, `bfloat16`, `fp8_ds_mla` | 64 | 576 | ❌ | ❌ | ✅ | ❌ | ❌ | Decoder | 9.x-10.x |
| `FLASH_ATTN_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16` | %16 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 9.x |
| `ROCM_AITER_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | %1 | Any | ❌ | ❌ | ❌ | ❌ | ❌ | Decoder | N/A |
| `ROCM_AITER_MLA_SPARSE` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 1, 64 | Any | ❌ | ❌ | ✅ | ❌ | ❌ | Decoder | N/A |
| `ROCM_AITER_TRITON_MLA` | fp16, bf16 | `auto` | Any | Any | ❌ | ❌ | ❌ | ❌ | ❌ | Decoder | N/A |
| `TOKENSPEED_MLA` | fp16, bf16 | `fp8`, `fp8_e4m3` | 32, 64 | Any | ❌ | ❌ | ❌ | ❌ | ❌ | Decoder | 10.x |
| `TRITON_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | %16 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | Any |
| `XPU_MLA_SPARSE` | fp16, bf16 | `auto`, `float16`, `bfloat16` | Any | 576 | ❌ | ❌ | ✅ | ❌ | ❌ | Decoder | Any |
