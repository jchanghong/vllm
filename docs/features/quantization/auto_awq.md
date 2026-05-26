# AutoAWQ

> ⚠️ **警告：**
    `AutoAWQ` 库已弃用。该功能已被 vLLM 项目采纳到 [`llm-compressor`](https://github.com/vllm-project/llm-compressor/tree/main/examples/awq) 中。
    对于推荐的量化工作流程，请参阅 [`llm-compressor`](https://github.com/vllm-project/llm-compressor/tree/main/examples/awq) 中的 AWQ 示例。关于弃用的更多详细信息，请参考原始 [AutoAWQ 仓库](https://github.com/casper-hansen/AutoAWQ)。

要创建新的 4 位量化模型，你可以利用 [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)。
量化将模型的精度从 BF16/FP16 降低到 INT4，从而有效减少模型的总内存占用。
主要优势在于更低的延迟和内存使用。

你可以通过安装 AutoAWQ 或选择 [Huggingface 上的 6500+ 个模型](https://huggingface.co/models?search=awq) 来量化自己的模型。

```bash
pip install autoawq
```

安装 AutoAWQ 后，你就可以量化模型了。有关更多详细信息，请参阅 [AutoAWQ 文档](https://casper-hansen.github.io/AutoAWQ/examples/#basic-quantization)。以下是如何量化 `mistralai/Mistral-7B-Instruct-v0.2` 的示例：

??? code

    ```python
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    quant_path = "mistral-instruct-v0.2-awq"
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

    # 加载模型
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 量化
    model.quantize(tokenizer, quant_config=quant_config)

    # 保存量化后的模型
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'模型已量化并保存至 "{quant_path}"')
    ```

要在 vLLM 中运行 AWQ 模型，你可以使用 [TheBloke/Llama-2-7b-Chat-AWQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-AWQ)，命令如下：

```bash
python examples/deployment/llm_engine_example.py \
    --model TheBloke/Llama-2-7b-Chat-AWQ \
    --quantization awq
```

AWQ 模型也直接通过 LLM 入口点支持：

??? code

    ```python
    from vllm import LLM, SamplingParams

    # 示例提示。
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # 创建采样参数对象。
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # 创建 LLM。
    llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")
    # 根据提示生成文本。输出是 RequestOutput 对象的列表，
    # 包含提示、生成的文本和其他信息。
    outputs = llm.generate(prompts, sampling_params)
    # 打印输出。
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```
