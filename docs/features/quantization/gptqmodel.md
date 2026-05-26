# GPTQModel

要创建新的 4 位或 8 位 GPTQ 量化模型，你可以利用 ModelCloud.AI 的 [GPTQModel](https://github.com/ModelCloud/GPTQModel)。

量化将模型的精度从 BF16/FP16（16 位）降低到 INT4（4 位）或 INT8（8 位），从而显著减少模型的总内存占用，同时提升推理性能。

兼容的 GPTQModel 量化模型可以利用 `Marlin` 和 `Machete` vLLM 自定义内核，在 Ampere (A100+) 和 Hopper (H100+) Nvidia GPU 上最大化批处理每秒事务数 (`tps`) 和令牌延迟性能。
这两个内核由 vLLM 和 NeuralMagic（现为 Redhat 的一部分）高度优化，可为量化的 GPTQ 模型提供世界级的推理性能。

GPTQModel 是世界上少数支持 `动态` 按模块量化的工具包之一，LLM 模型中的不同层和/或模块可以使用自定义量化参数进一步优化。`动态` 量化已完全集成到 vLLM 中，并得到 ModelCloud.AI 团队的支持。有关此功能和其他高级功能的更多详细信息，请参阅 [GPTQModel 自述文件](https://github.com/ModelCloud/GPTQModel?tab=readme-ov-file#dynamic-quantization-per-module-quantizeconfig-override)。

## 安装

你可以通过安装 [GPTQModel](https://github.com/ModelCloud/GPTQModel) 或选择 [Huggingface 上的 5000+ 个模型](https://huggingface.co/models?search=gptq) 来量化自己的模型。

```bash
pip install -U gptqmodel --no-build-isolation -v
```

## 量化模型

安装 GPTQModel 后，你就可以量化模型了。有关更多详细信息，请参阅 [GPTQModel 自述文件](https://github.com/ModelCloud/GPTQModel/?tab=readme-ov-file#quantization)。

以下是如何量化 `meta-llama/Llama-3.2-1B-Instruct` 的示例：

??? code

    ```python
    from datasets import load_dataset
    from gptqmodel import GPTQModel, QuantizeConfig

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
    ).select(range(1024))["text"]

    quant_config = QuantizeConfig(bits=4, group_size=128)

    model = GPTQModel.load(model_id, quant_config)

    # 增加 `batch_size` 以匹配 GPU/VRAM 规格，加速量化过程
    model.quantize(calibration_dataset, batch_size=2)

    model.save(quant_path)
    ```

## 在 vLLM 中运行量化模型

要在 vLLM 中运行 GPTQModel 量化模型，你可以使用 [DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2](https://huggingface.co/ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2)，命令如下：

```bash
python examples/deployment/llm_engine_example.py \
    --model ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2
```

## 通过 vLLM 的 Python API 使用 GPTQModel

GPTQModel 量化模型也直接通过 LLM 入口点支持：

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
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9)

    # 创建 LLM。
    llm = LLM(model="ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2")

    # 根据提示生成文本。输出是 RequestOutput 对象的列表，
    # 包含提示、生成的文本和其他信息。
    outputs = llm.generate(prompts, sampling_params)

    # 打印输出。
    print("-"*50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-"*50)
    ```
