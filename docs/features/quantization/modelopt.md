# NVIDIA Model Optimizer

[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) 是一个旨在优化模型以在 NVIDIA GPU 上进行推理的库。它包含了用于大型语言模型 (LLM)、视觉语言模型 (VLM) 以及扩散模型的训练后量化 (PTQ) 和量化感知训练 (QAT) 工具。

我们建议通过以下方式安装该库：

```bash
pip install nvidia-modelopt
```

## 支持的 ModelOpt checkpoint 格式

vLLM 通过 `hf_quant_config.json` 检测 ModelOpt checkpoint，并支持以下 `quantization.quant_algo` 取值：

- `FP8`：per-tensor 权重 scale（+ 可选的静态激活 scale）。
- `FP8_PER_CHANNEL_PER_TOKEN`：per-channel 权重 scale 和动态 per-token 激活量化。
- `FP8_PB_WO`（ModelOpt 可能输出 `fp8_pb_wo`）：block 缩放的 FP8 仅权重（通常为 128×128 blocks）。
- `NVFP4`：ModelOpt NVFP4 checkpoint（使用 `quantization="modelopt_fp4"`）。
- `MXFP8`：ModelOpt MXFP8 checkpoint（使用 `quantization="modelopt_mxfp8"`）。

## 使用 PTQ 量化 HuggingFace 模型

您可以使用 Model Optimizer 仓库中提供的示例脚本来量化 HuggingFace 模型。LLM PTQ 的主要脚本通常位于 `examples/llm_ptq` 目录中。

以下示例展示了如何使用 modelopt 的 PTQ API 量化一个模型：

??? code

    ```python
    import modelopt.torch.quantization as mtq
    from transformers import AutoModelForCausalLM

    # 从 HuggingFace 加载模型
    model = AutoModelForCausalLM.from_pretrained("<path_or_model_id>")

    # 选择量化配置，例如 FP8
    config = mtq.FP8_DEFAULT_CFG

    # 定义用于校准的前向循环函数
    def forward_loop(model):
        for data in calib_set:
            model(data)

    # PTQ，原地替换量化模块
    model = mtq.quantize(model, config, forward_loop)
    ```

模型量化后，您可以使用导出 API 将其导出为量化 checkpoint：

```python
import torch
from modelopt.torch.export import export_hf_checkpoint

with torch.inference_mode():
    export_hf_checkpoint(
        model,  # 量化后的模型。
        export_dir,  # 导出文件存放的目录。
    )
```

然后，量化后的 checkpoint 可以使用 vLLM 进行部署。例如，以下代码展示了如何使用 vLLM 部署 `nvidia/Llama-3.1-8B-Instruct-FP8`（这是从 `meta-llama/Llama-3.1-8B-Instruct` 派生的 FP8 量化 checkpoint）：

??? code

    ```python
    from vllm import LLM, SamplingParams

    def main():
        model_id = "nvidia/Llama-3.1-8B-Instruct-FP8"

        # 加载 modelopt checkpoint 时请确保指定 quantization="modelopt"
        llm = LLM(model=model_id, quantization="modelopt", trust_remote_code=True)

        sampling_params = SamplingParams(temperature=0.8, top_p=0.9)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    if __name__ == "__main__":
        main()
    ```

## 运行兼容 OpenAI 的服务器

要通过兼容 OpenAI 的 API 提供本地 ModelOpt checkpoint 服务：

```bash
vllm serve <path_to_exported_checkpoint> \
  --quantization modelopt \
  --host 0.0.0.0 --port 8000
```

## 测试（本地 checkpoint）

vLLM 的 ModelOpt 单元测试以本地 checkpoint 路径为条件，默认在 CI 中跳过。要在本地运行测试：

```bash
export VLLM_TEST_MODELOPT_FP8_PC_PT_MODEL_PATH=<path_to_fp8_pc_pt_checkpoint>
export VLLM_TEST_MODELOPT_FP8_PB_WO_MODEL_PATH=<path_to_fp8_pb_wo_checkpoint>
pytest -q tests/quantization/test_modelopt.py
```
