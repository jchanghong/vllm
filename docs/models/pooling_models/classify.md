# 分类用途

分类涉及预测哪个预定义的类别、类或标签最适合给定的输入。

## 摘要

- 模型用途：(序列) 分类
- 池化任务：`classify`
- 离线 API：
    - `LLM.classify(...)`
    - `LLM.encode(..., pooling_task="classify")`
- 在线 API：
    - [分类 API](classify.md#online-serving) (`/classify`)
    - Pooling API (`/pooling`)

(序列) 分类和 token 分类之间的关键区别在于它们的输出粒度：(序列) 分类为整个输入序列生成单个结果，而 token 分类为序列中的每个单独 token 生成结果。

许多分类模型同时支持 (序列) 分类和 token 分类。有关 token 分类的更多详细信息，请参阅[此页面](token_classify.md)。

只有当一个分类模型输出的 num_labels 等于 1 时，它才能用作评分模型并启用其评分 API，请参阅[此页面](scoring.md)。

## 典型用例

### 分类

分类模型最基本的应用是将输入数据分类到预定义的类别中。

## 支持的模型

### 纯文本模型

| 架构 | 模型 | 示例 HF 模型 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `ErnieForSequenceClassification` | 类 BERT 中文 ERNIE | `Forrest20231206/ernie-3.0-base-zh-cls` | | |
| `GPT2ForSequenceClassification` | GPT2 | `nie3e/sentiment-polish-gpt2-small` | | |
| `Qwen2ForSequenceClassification`<sup>C</sup> | 基于 Qwen2 | `jason9693/Qwen2.5-1.5B-apeach` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup> 等 | 生成式模型 | N/A | \* | \* |

### 多模态模型

!!! note
    有关多模态模型输入的更多信息，请参见[此页面](../supported_models.md#list-of-multimodal-language-models)。

| 架构 | 模型 | 输入 | 示例 HF 模型 | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `Qwen2_5_VLForSequenceClassification`<sup>C</sup> | 基于 Qwen2_5_VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `muziyongshixin/Qwen2.5-VL-7B-for-VideoCls` | | |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup> 等 | 生成式模型 | \* | N/A | \* | \* |

<sup>C</sup> 通过 `--convert classify` 自动转换为分类模型。([详细信息](./README.md#model-conversion))  
\* 功能支持与原模型相同。

如果您的模型不在上述列表中，我们将尝试使用 [as_seq_cls_model][vllm.model_executor.models.adapters.as_seq_cls_model] 自动转换模型。默认情况下，类别概率从最后一个 token 对应的 softmax 化隐藏状态中提取。

### 交叉编码器模型

交叉编码器（又称重排序器）模型是分类模型的一个子集，接受两个提示作为输入，并且输出 num_labels 等于 1。大多数分类模型也可以用作[交叉编码器模型](scoring.md#cross-encoder-models)。有关交叉编码器模型的更多信息，请参见[此页面](scoring.md)。

--8<-- "docs/models/pooling_models/scoring.md:supported-cross-encoder-models"

### 奖励模型

使用 (序列) 分类模型作为奖励模型。更多信息请参见[奖励模型](reward.md)。

--8<-- "docs/models/pooling_models/reward.md:supported-sequence-reward-models"

## 离线推理

### 池化参数

支持以下[池化参数][vllm.PoolingParams]。

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:classify-pooling-params"
```

### `LLM.classify`

[classify][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.classify] 方法为每个提示输出一个概率向量。

```python
from vllm import LLM

llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", runner="pooling")
(output,) = llm.classify("Hello, my name is")

probs = output.outputs.probs
print(f"Class Probabilities: {probs!r} (size={len(probs)})")
```

代码示例请参见：[examples/basic/offline_inference/classify.py](../../../examples/basic/offline_inference/classify.py)

### `LLM.encode`

[encode][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.encode] 方法适用于 vLLM 中的所有池化模型。

为分类模型使用 `LLM.encode` 时设置 `pooling_task="classify"`：

```python
from vllm import LLM

llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="classify")

data = output.outputs.data
print(f"Data: {data!r}")
```

## 在线服务

### 分类 API

在线 `/classify` API 类似于 `LLM.classify`。

#### Completion 参数

支持以下分类 API 参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:completion-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-params"
    ```

支持以下额外参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:completion-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
    ```

#### Chat 参数

对于类似聊天的输入（即如果传递了 `messages`），则支持以下参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:chat-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-params"
    ```

而是支持以下额外参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:chat-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
    ```

#### 示例请求

代码示例：[examples/pooling/classify/classification_online.py](../../../examples/pooling/classify/classification_online.py)

您可以通过传递字符串数组来对多个文本进行分类：

```bash
curl -v "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": [
      "Loved the new café—coffee was great.",
      "This update broke everything. Frustrating."
    ]
  }'
```

??? console "响应"

    ```json
    {
      "id": "classify-7c87cac407b749a6935d8c7ce2a8fba2",
      "object": "list",
      "created": 1745383065,
      "model": "jason9693/Qwen2.5-1.5B-apeach",
      "data": [
        {
          "index": 0,
          "label": "Default",
          "probs": [
            0.565970778465271,
            0.4340292513370514
          ],
          "num_classes": 2
        },
        {
          "index": 1,
          "label": "Spoiled",
          "probs": [
            0.26448777318000793,
            0.7355121970176697
          ],
          "num_classes": 2
        }
      ],
      "usage": {
        "prompt_tokens": 20,
        "total_tokens": 20,
        "completion_tokens": 0,
        "prompt_tokens_details": null
      }
    }
    ```

您也可以直接将字符串传递给 `input` 字段：

```bash
curl -v "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": "Loved the new café—coffee was great."
  }'
```

??? console "响应"

    ```json
    {
      "id": "classify-9bf17f2847b046c7b2d5495f4b4f9682",
      "object": "list",
      "created": 1745383213,
      "model": "jason9693/Qwen2.5-1.5B-apeach",
      "data": [
        {
          "index": 0,
          "label": "Default",
          "probs": [
            0.565970778465271,
            0.4340292513370514
          ],
          "num_classes": 2
        }
      ],
      "usage": {
        "prompt_tokens": 10,
        "total_tokens": 10,
        "completion_tokens": 0,
        "prompt_tokens_details": null
      }
    }
    ```

## 更多示例

更多示例请参见：[examples/pooling/classify](../../../examples/pooling/classify)

## 支持的功能

### 启用/禁用激活

您可以通过 `use_activation` 启用或禁用激活。

### 问题类型（例如 `multi_label_classification`）

您可以通过 Hugging Face 配置中的 `problem_type` 修改 `problem_type`。支持的 problem_type 有：`single_label_classification`、`multi_label_classification` 和 `regression`。

实现与 transformers [ForSequenceClassificationLoss](https://github.com/huggingface/transformers/blob/57bb6db6ee4cfaccc45b8d474dfad5a17811ca60/src/transformers/loss/loss_utils.py#L92) 的对齐。

### 仿射分数校准

仿射分数校准，也称为 [Platt 缩放](https://en.wikipedia.org/wiki/Platt_scaling)（Platt, 1999），是将分类器输出校准为良好校准概率的最广泛使用的方法。

校准遵循以下变换：

`activation((logit - logit_mean) / logit_sigma)`

| 参数 | 默认值 | 描述 |
| --------- | ------- | ----------- |
| `logit_mean` | `None` | 从 logits 减去的均值（居中分数） |
| `logit_sigma` | `None` | 在均值相减后用于缩放 logits 的标准差 |

计算顺序如下：

```python
logits -= logit_mean   # 减去均值（居中分数）
logits /= logit_sigma  # 除以 sigma（缩放）
logits = activation(logits)  # 例如 sigmoid
```

示例配置：

```bash
--pooler-config '{"use_activation": true, "logit_mean": 4.5, "logit_sigma": 1.0}'
```

## 已移除的功能

### 从 PoolingParams 中移除 softmax

我们已从 PoolingParams 中移除 `softmax` 和 `activation`。请改用 `use_activation`，因为我们允许 `classify` 和 `token_classify` 使用任何激活函数。

### 移除 `logit_bias` 和 `logit_scale`

`logit_bias` 和 `logit_scale` 分别是 `logit_mean` 和 `logit_sigma` 的已弃用别名。使用 `logit_scale` 时，会自动转换为 `logit_sigma = 1/logit_scale`。这些已弃用的参数将在 v0.21 中移除。
