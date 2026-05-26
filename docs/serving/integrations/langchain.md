# LangChain

vLLM 也可以通过 [LangChain](https://github.com/langchain-ai/langchain) 使用。

要安装 LangChain，请运行

```bash
pip install langchain langchain_community -q
```

要在单个或多个 GPU 上运行推理，请使用 `langchain` 中的 `VLLM` 类。

??? code

    ```python
    from langchain_community.llms import VLLM

    llm = VLLM(
        model="Qwen/Qwen3-4B",
        trust_remote_code=True,  # hf 模型必须
        max_new_tokens=128,
        top_k=10,
        top_p=0.95,
        temperature=0.8,
        # 用于分布式推理
        # tensor_parallel_size=...,
    )

    print(llm("What is the capital of France ?"))
    ```

有关更多详细信息，请参考此[教程](https://python.langchain.com/docs/integrations/llms/vllm)。
