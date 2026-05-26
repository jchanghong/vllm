# 隐藏状态提取 (Hidden State Extraction)

隐藏状态提取功能允许 vLLM 在推理过程中保存目标模型的中间层激活值。这对于训练 [EAGLE](eagle.md) 风格的草稿模型、知识蒸馏或离线分析模型内部状态非常有用。

!!! note
    可以通过传递 `num_hidden_layers` 作为层 ID 来保存最后一层的输出隐藏状态。请注意，这些状态_未_经过输出归一化处理。

## 离线示例

```python
import tempfile

from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector,
)

with tempfile.TemporaryDirectory() as tmpdir:
    llm = LLM(
        model="Qwen/Qwen3-8B",
        enable_chunked_prefill=False,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4],
                },
            },
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": tmpdir,
            },
        ),
    )

    outputs = llm.generate(
        ["The future of AI is"],
        SamplingParams(max_tokens=1),
    )

    for output in outputs:
        path = output.kv_transfer_params["hidden_states_path"]
        obj = example_hidden_states_connector.load_hidden_states(path)
        print(f"token_ids: {obj['token_ids'].shape}")
        print(f"hidden_states: {obj['hidden_states'].shape}")
```

完整示例请参见 [`examples/features/speculative_decoding/extract_hidden_states_offline.py`](../../../examples/features/speculative_decoding/extract_hidden_states_offline.py)。

## 在线示例

为获得更好的性能，在客户端会在生成后立即清理文件的在线使用场景中，建议使用 RAM 挂载的文件系统，例如 `/dev/shm/`。

```bash
vllm serve Qwen/Qwen3-8B \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4]}}}' \
    --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/dev/shm/hidden_states"}}' \
    --no-enable-chunked-prefill
```

## 配置

`kv_connector_extra_config` 字典接受以下选项：

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| `shared_storage_path` | `/tmp` | 隐藏状态文件的保存目录 |
| `num_writer_threads` | `8` | 异步磁盘写入的线程池大小 |
| `use_synchronization_lock` | `True` | 使用文件锁，使并发读取器在写入完成前阻塞。在不需要同步的批量生成场景中可以禁用。 |

## 输出格式

每个请求生成一个 `.safetensors` 文件，包含：

- **`hidden_states`** — 形状 `[num_tokens, num_extracted_layers, hidden_size]`
- **`token_ids`** — 形状 `[num_tokens]`

文件路径在 `output.kv_transfer_params["hidden_states_path"]` 中返回。使用连接器模块中的 `load_hidden_states()` 函数来读取文件并进行适当的同步。

!!! note
    分块预填充与此功能不兼容，必须禁用。
