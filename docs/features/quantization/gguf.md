# GGUF

!!! warning
    请注意，vLLM 中的 GGUF 支持目前处于高度实验性阶段且优化不足，可能与其他功能不兼容。目前，你可以使用 GGUF 作为减少内存占用的一种方式。如果遇到任何问题，请向 vLLM 团队报告。

!!! warning
    目前，vLLM 仅支持加载单文件 GGUF 模型。如果你有多文件 GGUF 模型，可以使用 [gguf-split](https://github.com/ggerganov/llama.cpp/pull/6135) 工具将其合并为单文件模型。

要在 vLLM 中运行 GGUF 模型，你可以使用 `repo_id:quant_type` 格式直接从 HuggingFace 加载。例如，从 [unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) 加载 Q4_K_M 量化模型：

```bash
# 建议使用基础模型的 tokenizer，以避免耗时且易出错的 tokenizer 转换。
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M --tokenizer Qwen/Qwen3-0.6B
```

你也可以添加 `--tensor-parallel-size 2` 来启用 2 个 GPU 的张量并行推理：

```bash
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
   --tokenizer Qwen/Qwen3-0.6B \
   --tensor-parallel-size 2
```

或者，你可以下载并使用本地 GGUF 文件：

```bash
wget https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
vllm serve ./Qwen3-0.6B-Q4_K_M.gguf --tokenizer Qwen/Qwen3-0.6B
```

!!! warning
    我们建议使用基础模型的 tokenizer 而非 GGUF 模型的 tokenizer。因为从 GGUF 转换 tokenizer 耗时且不稳定，特别是对于词汇量较大的模型。

GGUF 假设 HuggingFace 可以将元数据转换为配置文件。如果 HuggingFace 不支持你的模型，你可以手动创建配置并通过 `hf-config-path` 传递：

```bash
# 如果 HuggingFace 不支持你的模型，你可以手动提供 HuggingFace 兼容的配置路径
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
   --tokenizer Qwen/Qwen3-0.6B \
   --hf-config-path Qwen/Qwen3-0.6B
```

你也可以直接通过 LLM 入口点使用 GGUF 模型：

??? code

      ```python
      from vllm import LLM, SamplingParams

      # 在本脚本中，我们演示如何向 chat 方法传递输入：
      conversation = [
         {
            "role": "system",
            "content": "You are a helpful assistant",
         },
         {
            "role": "user",
            "content": "Hello",
         },
         {
            "role": "assistant",
            "content": "Hello! How can I assist you today?",
         },
         {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
         },
      ]

      # 创建采样参数对象。
      sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

      # 使用 repo_id:quant_type 格式创建 LLM。
      llm = LLM(
         model="unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
         tokenizer="Qwen/Qwen3-0.6B",
      )
      # 根据提示生成文本。输出是 RequestOutput 对象的列表，
      # 包含提示、生成的文本和其他信息。
      outputs = llm.chat(conversation, sampling_params)

      # 打印输出。
      for output in outputs:
         prompt = output.prompt
         generated_text = output.outputs[0].text
         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
      ```
