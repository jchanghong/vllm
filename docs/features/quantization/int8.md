# INT8 W8A8

vLLM 支持将权重和激活值量化到 INT8，以实现内存节省和推理加速。这种量化方法特别适用于在保持良好性能的同时减小模型体积。

请访问 HF 上[可直接与 vLLM 配合使用的流行 LLM 的 INT8 量化 checkpoint 集合](https://huggingface.co/collections/neuralmagic/int8-llms-for-vllm-668ec32c049dca0369816415)。

!!! note
    INT8 计算支持计算能力 > 7.5 的 NVIDIA GPU（Turing、Ampere、Ada Lovelace、Hopper）。

!!! warning
    **Blackwell GPU 限制**：INT8 不支持计算能力 >= 10.0 的 GPU（例如 RTX 6000 Blackwell）。
    请改用 [FP8 量化](fp8.md)，或在 Hopper/Ada/Ampere 架构上运行。

## 先决条件

要在 vLLM 中使用 INT8 量化，您需要安装 [llm-compressor](https://github.com/vllm-project/llm-compressor/) 库：

```bash
pip install llmcompressor
```

此外，还需安装 `vllm` 和 `lm-evaluation-harness` 用于评估：

```bash
pip install vllm "lm-eval[api]>=0.4.12"
```

## 量化流程

量化过程包括四个主要步骤：

1. 加载模型
2. 准备校准数据
3. 应用量化
4. 在 vLLM 中评估精度

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

### 2. 准备校准数据

将激活值量化到 INT8 时，需要样本数据来估算激活 scale。最好使用与您部署数据高度匹配的校准数据。对于通用的指令调优模型，可以使用诸如 `ultrachat` 之类的数据集：

??? code

    ```python
    from datasets import load_dataset

    NUM_CALIBRATION_SAMPLES = 512
    MAX_SEQUENCE_LENGTH = 2048

    # 加载并预处理数据集
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ```

</details>

### 3. 应用量化

现在，应用量化算法：

??? code

    ```python
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

    # 配置量化算法
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]

    # 应用量化
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # 保存压缩模型：Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token
    SAVE_DIR = MODEL_ID.split("/")[1] + "-W8A8-Dynamic-Per-Token"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    ```

此过程创建了一个 W8A8 模型，其中权重和激活值都量化为 8 位整数。

### 4. 评估精度

量化后，您可以在 vLLM 中加载并运行模型：

```python
from vllm import LLM

llm = LLM("./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token")
```

要评估精度，可以使用 `lm_eval`：

```bash
lm_eval --model vllm \
  --model_args pretrained="./Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Per-Token",add_bos_token=true \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 250 \
  --batch_size 'auto'
```

!!! note
    量化模型可能对 `bos` token 的存在敏感。运行评估时请确保包含 `add_bos_token=True` 参数。

## 最佳实践

- 从 512 个校准数据样本开始（如果精度下降则增加数量）
- 使用 2048 的序列长度作为起始点
- 使用模型训练时所用的聊天模板或指令模板
- 如果您对模型进行了微调，考虑使用一部分训练数据进行校准

## 故障排除与支持

如果您遇到任何问题或有功能请求，请在 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor/issues) GitHub 仓库中提交 issue。
