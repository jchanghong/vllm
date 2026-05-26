# FP8 W8A8

vLLM 支持使用 GPU（如 Nvidia H100 和 AMD MI300x）上的硬件加速进行 FP8（8 位浮点）权重和激活值量化。目前，仅 Hopper 和 Ada Lovelace GPU 正式支持 W8A8。Turing/Ampere GPU 支持 W8A16（仅权重 FP8），利用 Marlin 内核实现。对模型进行 FP8 量化可将模型内存需求减少 2 倍，吞吐量提升高达 1.6 倍，同时对精度的影响极小。

请访问 HF 上[可直接与 vLLM 配合使用的流行 LLM 的 FP8 量化 checkpoint 集合](https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127)。

硬件通常支持的 FP8 类型有两种不同的表示形式，各有不同的适用场景：

- **E4M3**：包含 1 个符号位、4 个指数位和 3 个尾数位。可存储的值范围最大为 +/-448 和 `nan`。
- **E5M2**：包含 1 个符号位、5 个指数位和 2 个尾数位。可存储的值范围最大为 +/-57344、+/- `inf` 和 `nan`。增加动态范围的代价是存储值的精度较低。

!!! note
    FP8 计算支持计算能力 >= 8.9 的 NVIDIA GPU（Ada Lovelace、Hopper）。
    FP8 模型可在计算能力 >= 7.5（Turing）的 GPU 上以仅权重 W8A16 模式运行，利用 FP8 Marlin。

## 安装

要使用 vLLM 生成高性能的 FP8 量化模型，您需要安装 [llm-compressor](https://github.com/vllm-project/llm-compressor/) 库：

```bash
pip install llmcompressor
```

## 量化流程

量化过程包括三个主要步骤：

1. 加载模型
2. 应用量化
3. 在 vLLM 中评估精度

### 1. 加载模型

使用标准的 `transformers` AutoModel 类加载您的模型和 tokenizer：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
```

### 2. 应用量化

对于 FP8 量化，我们可以通过简单的 RTN 量化来恢复精度。我们建议使用 `FP8_DYNAMIC` 方案定位所有 `Linear` 层，该方案使用：

- 静态、per-channel 的权重量化
- 动态、per-token 的激活值量化

由于简单的 RTN 不需要数据来进行权重量化，且激活值是动态量化的，因此此量化流程不需要任何校准数据。

??? code

    ```python
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    # 配置简单的 PTQ 量化
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )

    # 应用量化算法。
    oneshot(model=model, recipe=recipe)

    # 保存模型：Meta-Llama-3-8B-Instruct-FP8-Dynamic
    SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    ```

### 3. 评估精度

安装 `vllm` 和 `lm-evaluation-harness` 用于评估：

```bash
pip install vllm "lm-eval[api]>=0.4.12"
```

在 `vllm` 中加载并运行模型：

```python
from vllm import LLM

llm = LLM("./Meta-Llama-3-8B-Instruct-FP8-Dynamic")
result = llm.generate("Hello my name is")
print(result[0].outputs[0].text)
```

使用 `lm_eval` 评估精度（例如在 `gsm8k` 的 250 个样本上）：

!!! note
    量化模型可能对 `bos` token 的存在敏感。`lm_eval` 默认不会添加 `bos` token，因此运行评估时请确保包含 `add_bos_token=True` 参数。

```bash
MODEL=$PWD/Meta-Llama-3-8B-Instruct-FP8-Dynamic
lm_eval \
  --model vllm \
  --model_args pretrained=$MODEL,add_bos_token=True \
  --tasks gsm8k  --num_fewshot 5 --batch_size auto --limit 250
```

以下是一个结果分数示例：

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
| --- |------:| -------------- |-----:| --------- | - |----:| - |-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.768|±  |0.0268|
|     |       |strict-match    |     5|exact_match|↑  |0.768|±  |0.0268|
```

## 故障排除与支持

如果您遇到任何问题或有功能请求，请在 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) GitHub 仓库中提交 issue。

## 在线动态量化

将原始精度的 BF16/FP16 模型动态量化为 FP8 可以通过 vLLM 实现，无需任何校准数据。您可以通过在命令行中指定 `--quantization="fp8"` 或在 LLM 构造函数中设置 `quantization="fp8"` 来启用此功能。

在此模式下，所有 Linear 模块（除了最后的 `lm_head`）的权重都将量化到 FP8_E4M3 精度，使用 per-tensor scale。激活值在每次前向传播中计算其最小值和最大值，以提供动态的 per-tensor scale，从而实现高精度。因此，此模式下的延迟改进有限。

```python
from vllm import LLM

llm = LLM("facebook/opt-125m", quantization="fp8")
# INFO 06-10 17:55:42 model_runner.py:157] Loading model weights took 0.1550 GB
result = llm.generate("Hello, my name is")
print(result[0].outputs[0].text)
```
