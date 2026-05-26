# 视觉编码器（ViT）CUDA Graphs

vLLM 中的 [CUDA Graphs](cuda_graphs.md) 基础设施主要针对**解码器**（语言模型）的前向传播。vLLM 还支持独立于解码器，将**编码器**（视觉 transformer）的前向传播捕获为 CUDA Graphs。这基于 <https://github.com/vllm-project/vllm/pull/35963>。

!!! note
    编码器 CUDA Graphs 与解码器 CUDA Graphs 是正交的——两者可以同时启用。编码器 graphs 捕获视觉编码器的执行（例如 Qwen3-VL 中的 ViT），而解码器 graphs 捕获语言模型的执行，如 [CUDA Graphs 设计文档](cuda_graphs.md)中所述。

## 动机

视觉编码器推理会在主机端产生 CUDA 内核启动开销。当批次大小较小或图像尺寸较小时，这种开销更为显著。

编码器 CUDA Graphs 通过在模型初始化期间在多个 token 预算级别预捕获完整的编码器前向传播，然后在运行时回放相应的 graph，从而消除这种开销。

## 设计

编码器 CUDA Graph 系统使用基于**预算的捕获/回放**策略，由 [EncoderCudaGraphManager][vllm.v1.worker.encoder_cudagraph.EncoderCudaGraphManager] 管理。该系统包含以下核心组件：

* [EncoderCudaGraphManager][vllm.v1.worker.encoder_cudagraph.EncoderCudaGraphManager]：协调编码器 CUDA Graphs 的捕获、回放、贪心打包和数据并行执行。
* [SupportsEncoderCudaGraph][vllm.model_executor.models.interfaces.SupportsEncoderCudaGraph]：一个运行时可检查的协议，模型实现它以选择加入编码器 CUDA Graphs。
* [BudgetGraphMetadata][vllm.v1.worker.encoder_cudagraph.BudgetGraphMetadata]：保存单个 token 预算级别的已捕获 CUDA Graph 及其相关 I/O 缓冲区。

### 基于预算的 graph 捕获

多个 CUDA Graphs 在不同的 **token 预算**级别（例如 `[2048, 4096, 8192, 13824]`）预捕获。每个预算定义一个固定的 token 容量，所有预算共享相同的最大批次大小（图像数量）。每个级别的 `BudgetGraphMetadata` 存储 graph 以及预先分配的输入、元数据和输出缓冲区：

```python
@dataclass
class BudgetGraphMetadata:
    token_budget: int
    max_batch_size: int
    max_frames_per_batch: int
    graph: torch.cuda.CUDAGraph
    input_buffer: torch.Tensor       # 例如 pixel_values
    metadata_buffers: dict[str, torch.Tensor]  # 例如 embeddings, seq metadata
    output_buffer: torch.Tensor      # 编码器隐藏状态
```

预算通过 `get_encoder_cudagraph_budget_range()` 根据模型提供的范围自动生成为 2 的幂级别，即使最大值不在 2 的幂边界上，也始终包含最大预算。用户也可以通过 `CompilationConfig` 中的 `encoder_cudagraph_token_budgets` 显式指定预算。

### 运行时贪心装箱

当一批图像到达时，管理器按输出 token 数量（最小优先）对图像进行排序，并贪心地将尽可能多的图像打包到每个子批次中，同时保持在**最大** token 预算和最大批次大小范围内。一旦子批次最终确定（下一张图像将超出任一约束），管理器找到适合该子批次总 token 数的**最小**预算，并回放相应的 CUDA Graph。重复此过程直到批次耗尽。超出所有预算的图像回退到即时执行模式。

对于每次 graph 回放：

1. 将预先分配的 `input_buffer` 置零，然后将输入张量（例如 `pixel_values`）复制到其中。
2. 将 `metadata_buffers` 置零，然后切片复制预计算的值（例如旋转位置编码、序列元数据）。
3. 回放 CUDA Graph。
4. 从 `output_buffer` 克隆输出（克隆是必要的，因为该缓冲区在多次回放中被重用）。

### 数据并行支持

当 `mm_encoder_tp_mode="data"` 时，管理器通过 `get_load_balance_assignment` 在 TP rank 之间使用负载均衡分配来分发图像，在每个 rank 上本地执行，然后通过 `tensor_model_parallel_all_gather` 按原始顺序收集结果。

### 视频推理支持

继 <https://github.com/vllm-project/vllm/pull/35963>（支持图像推理的 ViT 完整 CUDA graph）之后，<https://github.com/vllm-project/vllm/pull/38061> 将编码器 CUDA graph 框架扩展到支持 Qwen3-VL 的视频推理。以前，CUDA graph 捕获/回放路径只处理图像输入（`pixel_values` + `image_grid_thw`）。视频输入使用不同的键（`pixel_values_videos` + `video_grid_thw`），并且需要更大的 `cu_seqlens` 缓冲区，因为每个视频项贡献多个帧（`T` 个注意力序列）。该 PR 通用了协议和管理器，通过单个共享的 graph 管理器处理两种模态。

!!! note
    当 EVS（高效视频采样）修剪启用时，视频 CUDA graphs 会自动禁用，因为 EVS 使 token 数量依赖于数据，与 CUDA graph 捕获不兼容。

    现在也支持每个提示的混合输入（图像+视频）。

## 通过 `SupportsEncoderCudaGraph` 进行模型集成

模型通过实现 [SupportsEncoderCudaGraph][vllm.model_executor.models.interfaces.SupportsEncoderCudaGraph] 协议来选择加入编码器 CUDA Graphs。该协议封装了所有模型特定的逻辑，使管理器保持模型无关。该协议定义了以下方法：

* `get_encoder_cudagraph_config()` —— 返回静态配置（支持的模态、输入键、缓冲区键、输出隐藏层大小）。
* `get_encoder_cudagraph_budget_range(vllm_config)` —— 返回用于自动推断 token 预算的 `(min_budget, max_budget)`。
* `get_encoder_cudagraph_num_items(mm_kwargs)` —— 返回批次中的项（例如图像）数量。
* `get_encoder_cudagraph_per_item_output_tokens(mm_kwargs)` —— 返回每项输出 token 数量，用于贪心打包。
* `get_encoder_cudagraph_per_item_input_sizes(mm_kwargs)` —— 返回每项输入大小（例如 patch 数量），用于 DP 负载均衡。
* `select_encoder_cudagraph_items(mm_kwargs, indices)` —— 按索引提取子批次项，用于贪心打包和 DP 分片。
* `prepare_encoder_cudagraph_capture_inputs(...)` —— 为 graph 捕获创建虚拟输入。
* `prepare_encoder_cudagraph_replay_buffers(...)` —— 在回放前从实际批次输入计算新的缓冲区值。
* `encoder_cudagraph_forward(...)` —— 使用预计算缓冲区的前向传播（在捕获和回放期间调用）。
* `encoder_eager_forward(...)` —— 当没有合适的 graph 时的回退即时前向传播。
* `get_input_modality(...)` —— 返回输入的模态。
* `get_max_frames_per_video()` —— 返回模型特定的每视频最大帧数。
* `postprocess_encoder_output(...)` —— 后处理编码器输出，默认直接调用 `scatter_output_slices`。

!!! note
    `SupportsEncoderCudaGraph` 协议设计为模型无关。新的视觉编码器模型可以通过实现协议方法选择加入，而无需修改管理器。

**支持的模型：**

| 架构 | 模型 | 图像的 CG | 视频的 CG |
| ------------ | ------ | ------------ | ------------ |
| `Qwen2VLForConditionalGeneration` | `Qwen2-VL` | ✅︎ | ✅︎ |
| `Qwen2_5_VLForConditionalGeneration` | `Qwen2.5-VL` | ✅︎ | ✅︎ |
| `Qwen3VLForConditionalGeneration` | `Qwen3-VL` | ✅︎ | ✅︎ |
| `Qwen3_5ForConditionalGeneration` | `Qwen3.5` | ✅︎ | ✅︎ |
| `Step3VLForConditionalGeneration` | `Step3-VL` | ✅︎ | ❌︎ |

!!! note
    编码器 CUDA Graphs 目前已在 Blackwell GPU 上使用 `--mm-encoder-attn-backend=FLASH_ATTN` 和 `--mm-encoder-attn-backend=FLASHINFER` 进行了测试。
    对于 Qwen2-VL 和 Qwen2.5-VL，仅测试了 FA2 和 FA3。

## 配置

`CompilationConfig` 中的三个字段控制编码器 CUDA Graphs：

* `cudagraph_mm_encoder`（`bool`，默认 `False`）—— 为多模态编码器启用 CUDA Graph 捕获。启用后，将每个 token 预算级别的完整编码器前向传播捕获为 CUDA Graph。
* `encoder_cudagraph_token_budgets`（`list[int]`，默认 `[]`）—— 用于捕获的 token 预算级别。如果为空（默认），则从模型架构自动推断为 2 的幂级别。用户提供的值会覆盖自动推断。
* `encoder_cudagraph_max_vision_items_per_batch`（`int`，默认 `0`）—— 捕获期间每批次的最大图像/视频数量。如果为 0（默认），则自动推断为 `max_budget // min_budget`。
* `encoder_cudagraph_max_frames_per_batch`（`int`，默认 `None`）—— 捕获期间每批次的最大视频帧数量。如果为 `None`（默认），则自动推断为 `encoder_cudagraph_max_vision_items_per_batch * max_frames_per_video`（`max_frames_per_video` 是根据其 `processing_info` 决定的模型特定值）。如果我们限制每个提示的视频数量为 `0`，它也将被设置为 `0`（即回退到仅图像模式）。

## 使用指南

### 图像推理

通过 `compilation_config` 启用编码器 CUDA Graphs：

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true}'
```

使用显式预算：

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824], "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

Python 示例：

```python
import vllm

compilation_config = {
    "cudagraph_mm_encoder": True,
    # 可选：覆盖自动推断的预算
    # "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824],
    # "encoder_cudagraph_max_vision_items_per_batch": 8,
}

model = vllm.LLM(
    model="Qwen/Qwen3-VL-32B",
    compilation_config=compilation_config,
)
```

管理器会跟踪命中/未命中统计信息并定期记录。"命中"表示图像通过 CUDA Graph 回放处理；"未命中"表示即时回退（图像超出了所有预算）。

### 视频推理

通过 `compilation_config` 启用编码器 CUDA Graphs：

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true}'
```

使用显式预算：

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824], "encoder_cudagraph_max_vision_items_per_batch": 8, "encoder_cudagraph_max_frames_per_batch": 64}'
```

Python 示例：

```python
import vllm

compilation_config = {
    "cudagraph_mm_encoder": True,
    # 可选：覆盖自动推断的预算
    # "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824],
    # "encoder_cudagraph_max_vision_items_per_batch": 8,
    # "encoder_cudagraph_max_frames_per_batch": 64,
}

model = vllm.LLM(
    model="Qwen/Qwen3-VL-32B",
    compilation_config=compilation_config,
)
```

## 关于性能

以下基准测试在 Blackwell GPU（GB200）上使用 `vllm bench mm-processor` 运行。详见 [#35963](https://github.com/vllm-project/vllm/pull/35963)。

### 单 GPU（1x GB200）

模型：`Qwen/Qwen3-VL-30B-A3B-Instruct`，数据集：`lmarena-ai/VisionArena-Chat`（3000 条提示，300 条预热），`max_model_len=32768`。

| 后端 | 平均延迟改进 | P99 延迟改进 |
| :------ | :----------------------- | :---------------------- |
| FLASH_ATTN | +11.8%（5.13→4.52ms） | +31.6%（9.16→6.26ms） |
| FLASHINFER | +19.6%（5.42→4.36ms） | +40.3%（10.87→6.49ms） |

复现方法：

```bash
vllm bench mm-processor \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 3000 --num-warmups 300 \
  --max-model-len 32768 --seed 42 \
  --mm-encoder-attn-backend FLASH_ATTN \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4864], "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

### 多 GPU（4x GB200，TP=4，DP=4）

模型：`Qwen/Qwen3-VL-32B-Instruct`，数据集：`random-mm`（1000 条提示，200 条预热，每次请求 20 张 336x336 图像），`max_model_len=8192`。

| 后端 | 平均延迟改进 | P99 延迟改进 |
| :------ | :----------------------- | :---------------------- |
| FLASH_ATTN | +18.4%（28.39→23.16ms） | +14.0%（238.78→205.28ms） |
| FLASHINFER | +44.4%（23.24→12.91ms） | +84.9%（172.41→26.05ms） |

复现方法：

```bash
vllm bench mm-processor \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --dataset-name random-mm \
  --random-mm-base-items-per-request 20 \
  --random-mm-num-mm-items-range-ratio 0.0 \
  --random-mm-bucket-config '{"(336,336,1)": 1.0}' \
  --num-prompts 1000 --num-warmups 200 \
  --max-model-len 8192 --seed 42 \
  --mm-encoder-attn-backend FLASHINFER \
  --tensor-parallel-size 4 --mm-encoder-tp-mode data \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4864], "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

!!! note
    在 GPU（A100）上进行视频推理的基准测试详情请参见 [#38061](https://github.com/vllm-project/vllm/pull/38061)。
