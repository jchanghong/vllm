# 优化级别

## 概述

vLLM 提供 4 个优化级别（`-O0`、`-O1`、`-O2`、`-O3`），允许用户在启动时间和性能之间进行权衡：

- `-O0`：无优化。最快的启动时间，但性能最低。
- `-O1`：快速优化。简单的编译和快速融合，以及 PIECEWISE cudagraphs。
- `-O2`：默认优化。附加编译范围、额外融合、FULL_AND_PIECEWISE cudagraphs。
- `-O3`：激进优化。目前等同于 `-O2`，但将来可能包括额外的耗时或实验性优化。

所有优化级别的默认值都可以通过手动设置底层标志来实现。
用户设置的标志优先于优化级别的默认值。

## 级别汇总及使用示例

```bash
# CLI 使用
vllm serve RedHatAI/Llama-3.2-1B-FP8 -O1

# Python API 使用
from vllm.entrypoints.llm import LLM

llm = LLM(
    model="RedHatAI/Llama-3.2-1B-FP8",
    optimization_level=2 # 等同于 -O2
)
```

### `-O0`：无优化

尽可能快地启动——无自动调优、无编译、无 cudagraphs。
此级别适合开发和调试的初始阶段。

设置：

- `-cc.cudagraph_mode=NONE`
- `-cc.mode=NONE`（也会导致 `-cc.custom_ops=["none"]`）
- `-cc.pass_config.fuse_...=False`（所有融合禁用）
- `--kernel-config.enable_flashinfer_autotune=False`

### `-O1`：快速优化

优先考虑快速启动，但仍启用编译和 cudagraphs 等基本优化。
此级别对于大多数开发场景来说是一个很好的平衡，您希望获得更快的启动速度，但
仍然确保您的代码不会破坏 cudagraphs 或编译。

设置：

- `-cc.cudagraph_mode=PIECEWISE`
- `-cc.mode=VLLM_COMPILE`
- `--kernel-config.enable_flashinfer_autotune=True`

融合：

- `-cc.pass_config.fuse_norm_quant=True`*
- `-cc.pass_config.fuse_act_quant=True`*
- `-cc.pass_config.fuse_act_padding=True`†
- `-cc.pass_config.fuse_mla_dual_rms_norm=True`†

\* 这些融合仅在其中一个算子使用自定义内核时启用，否则 Inductor 融合效果更好。</br>
† 这些融合仅适用于 ROCm 并且需要 AITER。

### `-O2`：完全优化（默认）

优先考虑性能，以增加启动时间为代价。
此级别推荐用于生产工作负载，因此是默认选项。
此级别的融合由于额外的编译范围，可能需要更长时间。

设置（在 `-O1` 之上新增）：

- `-cc.cudagraph_mode=FULL_AND_PIECEWISE`
- `-cc.pass_config.fuse_allreduce_rms=True`
- `-cc.pass_config.fuse_rope_kvcache=True`†

† 这些融合仅适用于 ROCm 并且需要 AITER。

### `-O3`：激进优化

此级别目前与 `-O2` 相同，但将来可能包括
更耗时或实验性的额外优化。

## 故障排除

### 常见问题

1. **启动时间过长**：使用 `-O0` 或 `-O1` 以获得更快的启动速度
2. **编译错误**：使用 `debug_dump_path` 获取额外的调试信息
3. **性能问题**：确保在生产环境中使用 `-O2`
