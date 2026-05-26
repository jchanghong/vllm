# NixlConnector 兼容性矩阵

本页记录了 **使用 NixlConnector 进行分离式预填充** 的功能兼容性。有关一般使用说明，请参阅 [NixlConnector 使用指南](nixl_connector_usage.md)。有关分离式预填充的概述，请参阅 [分离式预填充](disagg_prefill.md)。

!!! note
    本页反映代码库的当前状态，并可能随着功能的演进而变化。标记为 🟠 或 ❌ 的条目可能链接到跟踪问题。有关即将推出的功能开发，请参阅 [NIXL 连接器路线图](https://github.com/vllm-project/vllm/issues/33702)。

**图例：**

- ✅ = 完全支持
- 🟠 = 部分支持（参见脚注）
- ❌ = 不支持
- ❔ = 未知 / 尚未验证
- 🚧 = 正在进行中

!!! info "普遍支持的功能"
    以下功能在使用 NixlConnector PD 分离式服务时适用于 **所有** 模型架构：

    [分块预填充](../configuration/optimization.md#chunked-prefill) |
    [APC（前缀缓存）](automatic_prefix_caching.md) |
    [数据并行](../serving/data_parallel_deployment.md) |
    CUDA graph |
    Logprobs |
    Prompt Logprobs |
    [提示嵌入](prompt_embeds.md) |
    多个 NIXL 后端（UCX、GDS、LIBFABRIC 等）

## 模型架构 x 能力

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| 模型类型 | <abbr title="基础预填充/解码分离">基础 PD</abbr> | <abbr title="推测解码">推测解码</abbr> | <abbr title="异构张量并行（P TP != D TP）">异构 TP</abbr> | <abbr title="跨层块优化">跨层块</abbr> | <abbr title="滑动窗口注意力">SWA</abbr> | <abbr title="CPU 主机缓冲区卸载（例如 TPU）">主机缓冲区</abbr> | <abbr title="P 和 D 使用不同块大小">异构块大小</abbr> |
| - | - | - | - | - | - | - | - |
| 密集型 Transformer | ✅ | ✅<sup>1</sup> | ✅ | ✅<sup>2</sup> | ✅ | ✅ | 🟠<sup>3</sup> |
| MLA（例如 DeepSeek-V2/V3） | ✅ | ✅<sup>1</sup> | 🟠<sup>4</sup> | ✅<sup>2</sup> | ✅ | ✅ | 🟠<sup>3</sup> |
| 稀疏 MLA（例如 DeepSeek-V3.2） | ✅ | ✅<sup>1</sup> | 🟠<sup>4</sup> | ✅<sup>2</sup> | ✅ | ✅ | 🟠<sup>3</sup> |
| 混合 SSM / Mamba | ✅ | ❔ | 🚧<sup>5</sup> | ❌ | ✅ | ✅ | ❌<sup>6</sup> |
| MoE | ✅ | ✅<sup>1</sup> | ✅ | ✅<sup>2</sup> | ✅ | ✅ | 🟠<sup>3</sup> |
| 多模态 | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ |
| 编码器-解码器 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

<sup>1</sup> P 和 D 实例必须使用相同的推测配置。

<sup>2</sup> 需要 `FLASH_ATTN` 或 `FLASHINFER` 后端 **以及** `HND` KV 缓存布局。通过 `--kv-transfer-config '{"kv_connector_extra_config": {"enable_cross_layers_blocks": "True"}}'` 启用。

<sup>3</sup> 仅在 **不** 需要 HMA 时支持（即非混合模型）。块 ID 会自动重新映射。仅支持 P 块大小 < D 块大小。

<sup>4</sup> MLA KV 缓存在 TP 工作节点间复制，因此异构 TP 可以工作，但不会进行头部拆分。当 P TP > D TP 时，仅执行单次读取（跳过冗余的排名）。D TP > P TP 也能工作。

<sup>5</sup> 混合 SSM (Mamba) 模型需要 **同构 TP** (`P TP == D TP`)。Mamba 层尚不支持异构 TP。

<sup>6</sup> HMA（混合模型必需）不支持不同的远程块大小。

## 配置说明

### P 和 D 之间必须匹配的内容

默认情况下，在握手期间会检查 **兼容性哈希值**。P 和 D 实例必须就以下方面达成一致：

- vLLM 版本和 NIXL 连接器版本
- 模型（架构、dtype、KV 头数量、头大小、隐藏层数量）
- 注意力后端
- KV 缓存 dtype (`cache_dtype`)

!!! warning
    使用 `--kv-transfer-config '{"kv_connector_extra_config": {"enforce_handshake_compat": false}}'` 禁用哈希值检查风险自负。

### P 和 D 之间可以安全不同的内容

- `tensor-parallel-size`（异构 TP，受上述模型限制约束）
- `block-size`（异构块大小，受上述限制约束）
- KV 缓存块数量（由每个实例的可用内存决定）

### KV 缓存布局

- NixlConnector 默认使用 **`HND`** 布局以获得最佳传输性能（非 MLA 模型）。
- 支持 `NHD` 布局，但 **不** 允许异构 TP 头部拆分。
- 实验性 `HND` ↔ `NHD` 置换：通过 `--kv-transfer-config '{"enable_permute_local_kv": true}'` 启用。HMA 不支持。

### 量化 KV 缓存

[量化 KV 缓存](quantization/quantized_kvcache.md)（例如 FP8）要求 P 和 D 实例使用 **相同的** `cache_dtype`。不匹配的缓存 dtype 将在握手期间导致兼容性哈希值检查失败。

- **静态量化**（从检查点加载缩放因子）：✅ 支持。每个实例独立从模型检查点加载缩放因子。
- **动态量化**（运行时计算缩放因子）：❌ 不支持。每块缩放因子不会随 KV 缓存数据一起传输。
- **打包布局缩放因子**（缩放因子内联存储在权重中）：✅ 支持。缩放因子与 KV 缓存块一起传输。
