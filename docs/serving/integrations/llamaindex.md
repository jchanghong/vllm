# LlamaIndex

vLLM 也可以通过 [LlamaIndex](https://github.com/run-llama/llama_index) 使用。

要安装 LlamaIndex，请运行

```bash
pip install llama-index-llms-vllm -q
```

要在单个或多个 GPU 上运行推理，请使用 `llamaindex` 中的 `Vllm` 类。

```python
from llama_index.llms.vllm import Vllm

llm = Vllm(
    model="microsoft/Orca-2-7b",
    tensor_parallel_size=4,
    max_new_tokens=100,
    vllm_kwargs={"gpu_memory_utilization": 0.5},
)
```

有关更多详细信息，请参考此[教程](https://docs.llamaindex.ai/en/latest/examples/llm/vllm/)。
