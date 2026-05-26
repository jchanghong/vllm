# TorchAO

TorchAO 是一个用于 PyTorch 的架构优化库，它为推理和训练提供高性能数据类型、优化技术和内核，具有与 torch.compile、FSDP 等原生 PyTorch 功能的可组合性。一些基准数据可以在[这里](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks)找到。

我们建议安装最新的 torchao  nightly 版本：

```bash
# 安装最新的 TorchAO nightly 构建版本
# 选择与你系统匹配的 CUDA 版本（cu126、cu128 等）
pip install \
    --pre torchao>=10.0.0 \
    --index-url https://download.pytorch.org/whl/nightly/cu126
```

## 量化 HuggingFace 模型

你可以使用 torchao 量化自己的 huggingface 模型，例如 [transformers](https://huggingface.co/docs/transformers/main/en/quantization/torchao) 和 [diffusers](https://huggingface.co/docs/diffusers/en/quantization/torchao)，并使用以下示例代码将检查点保存到 huggingface hub，如[这个示例](https://huggingface.co/jerryzh168/llama3-8b-int8wo)所示：

??? code

    ```Python
    import torch
    from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
    from torchao.quantization import Int8WeightOnlyConfig

    model_name = "meta-llama/Meta-Llama-3-8B"
    quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_text = "What are we having for dinner?"
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    hub_repo = # 你的 HUB REPO ID
    tokenizer.push_to_hub(hub_repo)
    quantized_model.push_to_hub(hub_repo, safe_serialization=False)
    ```

另外，你也可以使用 [TorchAO 量化空间](https://huggingface.co/spaces/medmekk/TorchAO_Quantization) 通过简单的 UI 对模型进行量化。
