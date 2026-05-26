# 单元测试

本页面解释了如何编写单元测试来验证模型的实现。

## 必需的测试

这些测试是让您的 PR 合并到 vLLM 库中所必需的。
如果没有这些测试，您 PR 的 CI 将会失败。

### 模型加载

在 [tests/models/registry.py](../../../tests/models/registry.py) 中为您的模型包含一个 HuggingFace 仓库示例。
这使得一个单元测试能够加载虚拟权重，以确保模型可以在 vLLM 中初始化。

!!! important
    每个部分中的模型列表应按字母顺序维护。

!!! tip
    如果您的模型需要开发版本的 HF Transformers，您可以设置
    `min_transformers_version` 在 CI 中跳过测试，直到模型发布。

## 可选测试

这些测试对于将您的 PR 合并到 vLLM 库中是可选的。
通过这些测试可以更有信心地确认您的实现是正确的，并有助于避免未来的回归问题。

### 模型正确性

这些测试比较 vLLM 模型输出与 [HF Transformers](https://github.com/huggingface/transformers) 的输出。您可以在 [tests/models](../../../tests/models) 的子目录下添加新的测试。

#### 生成式模型

对于[生成式模型](../../models/generative_models.md)，有两个级别的正确性测试，定义在 [tests/models/utils.py](../../../tests/models/utils.py) 中：

- 精确正确性（`check_outputs_equal`）：vLLM 输出的文本应与 HF 输出的文本完全匹配。
- Logprobs 相似度（`check_logprobs_close`）：vLLM 输出的 logprobs 应出现在 HF 输出的 top-k logprobs 中，反之亦然。

#### 池化模型

对于[池化模型](../../models/pooling_models/README.md)，我们只需检查余弦相似度，定义在 [tests/models/utils.py](../../../tests/models/utils.py) 中。

### 多模态处理

#### 通用测试

将您的模型添加到 [tests/models/multimodal/processing/test_common.py](../../../tests/models/multimodal/processing/test_common.py) 可以验证以下输入组合产生相同的输出：

- 文本 + 多模态数据
- Token + 多模态数据
- 文本 + 缓存的多模态数据
- Token + 缓存的多模态数据

#### 模型特定测试

您可以在 [tests/models/multimodal/processing](../../../tests/models/multimodal/processing) 下添加一个新文件，以运行仅适用于您的模型的测试。

例如，如果您的模型的 HF 处理器接受用户指定的关键字参数，您可以验证关键字参数是否正确应用，如 [tests/models/multimodal/processing/test_phi3v.py](../../../tests/models/multimodal/processing/test_phi3v.py) 中所示。
