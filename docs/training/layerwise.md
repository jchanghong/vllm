# 什么是逐层（重新）加载？

逐层重新加载是用于处理将新的权重数据加载到现有权重数据目标中，而不会触发 cuda 图和其他运行时工件重新编译的系统。该系统用于支持 [QeRL](https://arxiv.org/pdf/2510.11696) 风格的后训练流程，其中全精度训练器权重被量化并加载到目标 vLLM 实例中，以进行快速、高探索性的 rollout。核心实现可在 [layerwise.py](../../vllm/model_executor/model_loader/reload/layerwise.py) 中找到。

![逐层](../assets/training/layerwise.png)

## 用于 QeRL 的逐层重新加载

为了将新权重加载到现有权重数据目标中，权重必须经过以下操作：

- 传输：权重必须从训练器模型传输到目标节点/设备
- 融合：权重分区必须融合，例如 qkv/gate_up
- 处理：这通常意味着在线量化和特定于内核的填充或步长对齐
- 分片：权重必须根据所选的并行策略进行分片
- 复制：权重必须复制到现有的权重数据目标中

逐层重新加载通过以下步骤实现这一点：

1. 权重从训练器**传输**到目标（参见[权重传输](weight_transfer/README.md)）
2. 权重通过 `model.load_weights` 加载，在此过程中它们被**分片**和**融合**
3. 一旦某一层的所有权重加载完毕，即对权重进行**在线处理**
4. 权重被**复制**到现有的权重数据目标中

有关实现的更多信息，请参见[底层 `layerwise` API](#low-level-layerwise-api)。

## 带在线量化的逐层加载

在线量化是指用户提供全精度权重，这些权重在加载到模型时被即时量化。逐层重新加载系统通过将在线量化视为一个**处理**步骤来应对，该步骤在首次加载和重新加载时都以在线方式处理。一个典型的在线量化方法实现应该如下所示：

```python
class Fp8OnlineLinearMethod(Fp8LinearMethod):
    """Fp8LinearMethod 的在线版本，加载全精度检查点
    并在加载过程中量化权重。"""

    uses_meta_device: bool = True

    def create_weights(self, layer: torch.nn.Module, ...):
        # 权重在加载期间被物化和处理
        layer.weight = ModelWeightParameter(
            data=torch.empty(..., device="meta"),
            weight_loader=weight_loader,
        )

        # 设置在线处理
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        layer.weight, layer.weight_scale = ops.scaled_fp8_quant(layer.weight)

        # 防止重复处理（例如，在权重重新加载期间）
        layer._already_called_process_weights_after_loading = True
```

## 示例用法

### 高层权重传输 API

逐层重新加载系统与后训练权重传输系统集成。要将逐层重新加载与权重传输系统结合使用，请参考[此处](../../examples/rl/)的示例。逐层重新加载由 `WeightTransferUpdateInfo.is_checkpoint_format` 标志控制，默认设置为 `True`。

### 中层 `reload_weights` API

逐层重新加载也通过 `reload_weights` API 暴露。可以使用以下代码调用此接口：

```python
from vllm import LLM

llm = LLM("Qwen/Qwen3-0.6B")
llm.collective_rpc("reload_weights")
```

此接口还允许指定一个 `weights_path`，用于选择要加载的检查点路径：

```python
from vllm import LLM

# 用于测试的微调模型检查点
mul_path = "inference-optimization/Qwen3-0.6B-debug-multiply"
add_path = "inference-optimization/Qwen3-0.6B-debug-add"

llm = LLM("Qwen/Qwen3-0.6B")
llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_path})
llm.generate("3 4 = ")  # 12

llm.collective_rpc("reload_weights", kwargs={"weights_path": add_path})
llm.generate("3 4 = ")  # 7
```

最后，可以直接提供一个 `weights_iterator`。这个迭代器可以是惰性定义的，也可以是急切定义的。

```python
from vllm import LLM

weights_iterator = [("q_proj", ...), ("k_proj", ...), ...]

llm = LLM("Qwen/Qwen3-0.6B")
llm.collective_rpc("reload_weights", kwargs={"weights_iterator": weights_iterator})
```

### 底层 `layerwise` API

[layerwise.py](../../vllm/model_executor/model_loader/reload/layerwise.py) 实现了以下函数来执行其生命周期：

| 函数 | 用途 | 量化重新加载 | 在线量化 |
| - | - | - | - |
| `record_metadata_for_reloading` | 记录张量元数据，以便层可以在 meta 设备上恢复 | 由 `BaseModelLoader` 调用 | 由 `BaseModelLoader` 调用 |
| `restore_layer_on_meta` | 在重新加载开始时将层恢复到模型格式 | 由 `initialize_layerwise_reload` 调用 | 不调用。在线量化权重已通过 `...OnlineLinearMethod.create_weights` 在 meta 设备上创建 |
| `initialize_online_processing` | 使用 `online_process_loader` 包装器包装权重加载器，该包装器缓冲权重直到所有层权重加载完毕 | 由 `initialize_layerwise_reload` 调用 | 由 `...OnlineLinearMethod.create_weights` 调用 |
| `_layerwise_process` | 在所有权重加载完毕后处理层 | 由加载期间的 `online_process_loader` 调用 | 由加载期间的 `online_process_loader` 调用 |
| `_copy_and_restore_kernel_tensors` | 将处理后的权重复制到原始张量位置，以影响已编译的 cuda 图等 | 由 `process_weights_after_loading` 之后的 `_layerwise_process` 调用 | 不调用。尚未编译 cuda 图 |
| `finalize_layerwise_processing` | 捕获任何未加载所有权重的层（例如注意力权重或带填充的权重） | 由 `BaseModelLoader` 调用 | 由 `BaseModelLoader` 调用 |

您可以通过调用 `initialize_layerwise_reload`、加载权重、然后调用 `finalize_layerwise_processing` 来直接接入此生命周期：

```python
from vllm import LLM
from vllm.model_executor.model_loader.reload import initialize_layerwise_reload, finalize_layerwise_processing

llm = LLM("Qwen/Qwen3-0.6B")

# 此模型路径需要 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 且不稳定
model = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.get_model()

# 逐层重新加载
initialize_layerwise_reload(model)
model.load_weights(...)
finalize_layerwise_processing(model, llm.model_config)
```

## 排查内存使用过高问题

逐层重新加载允许用户在权重加载到模型时增量加载和处理权重。该系统依赖于在设备上缓冲层权重，直到某一层的所有权重加载完毕。然而，如果没有卸载机制，如果权重以乱序方式加载，这种方法必然会导致过多的缓冲。

因此，用户在重新加载权重时必须注意权重的顺序。权重应该"按顺序"加载，这意味着在开始加载下一层权重之前，每一层的所有权重都已完全加载。"乱序"加载可能导致层权重在加载其他层权重时一直处于缓冲状态，从而导致内存使用过高。在下面的示例中，q_proj、k_proj、v_proj 和 up_proj 同时被缓冲，比在 q_proj、k_proj 和 v_proj 之后加载 up_proj 使用了更多内存。

| 正确加载 | 错误加载 |
| - | - |
| ![逐层](../assets/training/layerwise_good_loading.png) | ![逐层](../assets/training/layerwise_bad_loading.png) |

如果权重以乱序方式加载，用户会看到如下所示的警告：

```console
WARNING [layerwise.py:198] Allocating 28.5 MB of device memory to buffers to load ["QKVParallelLinear", "MergedColumnParallelLinear"] layers. This extra memory usage can be avoided by ordering weights by their parent layer when reloading.
```
