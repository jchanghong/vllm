# Multi-LoRA 调优

**注意**：应通过导出 `VLLM_TUNED_CONFIG_FOLDER=/path/to/configs` 来指定 LoRA 配置文件夹。
如果没有设置此项，收缩/扩展核将使用默认配置。

## 调优过程

Multi-LoRA 收缩/扩展 Triton 核调优遵循与
[Triton MoE 调优](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py)
类似的方法论。

1. 定义搜索空间。以下是一个搜索空间示例：

   ```python
   block_m_range = [16, 32, 64, 128, 256]
   block_n_range = [32, 64, 128, 256]
   block_k_range = [32, 64, 128, 256]
   num_warps_range = [4, 8]
   num_stage_range = [2, 3, 4, 5]
   num_ctas_range = [1]
   split_k_range = [4, 8, 16, 32, 64]
   ```

2. 获取目标模型在特定 TP 大小下使用的所有隐藏状态大小和 num_slices 值。

   例如，你可以通过检查
   [add_lora_linear](https://github.com/vllm-project/vllm/blob/main/vllm/lora/punica_wrapper/punica_gpu.py#L181)
   来获取这些信息：

   ```python
   print(f"x_shape: {x.view(-1, x.shape[-1]).shape}")
   print(f"num_slices: {len(output_slices)}")
   for i in range(len(output_slices)):
       print(f"a{i} shape: {lora_a_stacked[i].shape}")
       print(f"b{i} shape: {lora_b_stacked[i].shape}")
   print("y_shape", y.shape)
   ```

3. 使用预定义搜索空间中生成的不同核配置，通过网格搜索对收缩/扩展核运行时间进行基准测试，
   以找到最优核配置。
   vLLM 的 [benchmark_lora.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_lora.py)
   可用于为不同形状搜索配置。

## 配置文件

### 文件命名

| 核类型 | 文件名模板 | 示例 |
| ------------------------- | ------------------------------------------- | -------------------------------------------- |
| shrink | `{gpu_name}_SHRINK.json` | `NVIDIA_H200_SHRINK.json` |
| expand | `{gpu_name}_EXPAND_{add_input}.json` | `NVIDIA_H200_EXPAND_TRUE.json` |
| fused_moe_lora_w13_shrink | `{gpu_name}_FUSED_MOE_LORA_W13_SHRINK.json` | `NVIDIA_H200_FUSED_MOE_LORA_W13_SHRINK.json` |
| fused_moe_lora_w13_expand | `{gpu_name}_FUSED_MOE_LORA_W13_EXPAND.json` | `NVIDIA_H200_FUSED_MOE_LORA_W13_EXPAND.json` |
| fused_moe_lora_w2_shrink | `{gpu_name}_FUSED_MOE_LORA_W2_SHRINK.json` | `NVIDIA_H200_FUSED_MOE_LORA_W2_SHRINK.json` |
| fused_moe_lora_w2_expand | `{gpu_name}_FUSED_MOE_LORA_W2_EXPAND.json` | `NVIDIA_H200_FUSED_MOE_LORA_W2_EXPAND.json` |

`gpu_name` 可以通过调用 `torch.cuda.get_device_name()` 自动检测。

### JSON 结构

最优核配置文件以 JSON 格式保存，结构为 `config_data[max_loras][num_slices][m][k][n][i]`，
其中 `i` 是 `fused_moe_lora` 配置中的可选维度，表示 MoE 层的中间大小。
