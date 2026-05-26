# 多模态数据处理

为了在 vLLM 中启用各种优化（如[分块预填充](../configuration/optimization.md#chunked-prefill)和[前缀缓存](../features/automatic_prefix_caching.md)），我们使用 [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor] 根据 HF processor 的输出来提供占位符特征 token（例如 `<image>`）与多模态输入（例如原始输入图像）之间的对应关系。

以下是 [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor] 的主要特性：

## Prompt 更新检测

HF processor 的主要职责之一是用占位符 token 更新 prompt。例如：

- 在字符串开头插入特征占位符 token（例如 `<image><image>...<image>`，数量等于特征大小）。
- 将现有的输入占位符 token（例如单个图像的 `<image>`）替换为特征占位符 token（例如 `<image><image>...<image>`，数量等于特征大小）。

哪些 token 已被更新的信息是找到占位符特征 token 与多模态输入之间对应关系的关键。

在 vLLM 中，此信息通过 [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates] 中的 [PromptUpdate][vllm.multimodal.processing.PromptUpdate] 指定。我们可以通过检查更新后的 token 是否存在来自动检测 HF 是否已更新 prompt。

## 分词后的 prompt 输入

为了支持在单独的进程中进行分词，我们支持将输入 token ID 与多模态数据一起传递。

### 问题

考虑 HF processor 遵循以下主要步骤：

1. 对文本进行分词
2. 处理多模态输入
3. 执行 prompt 更新

我们要求：

- 对于文本 + 多模态输入，应用步骤 1--3。
- 对于分词后 + 多模态输入，仅应用步骤 2--3。

如何在不重写 HF processor 的情况下实现这一点？我们可以尝试在不同输入上多次调用 HF processor：

- 对于文本 + 多模态输入，直接调用 HF processor。
- 对于分词后 + 多模态输入，仅在多模态输入上调用 processor。

虽然 HF processor 原生支持文本 + 多模态输入，但对于分词后 + 多模态输入并非如此：如果输入占位符 token 的数量与多模态输入的数量不对应，将抛出错误。

此外，由于分词后的文本没有经过 HF processor，我们必须自行应用步骤 3 以保持输出 token 和多模态数据之间的一致性。

### 虚拟文本

我们通过要求每个模型通过 [get_dummy_text][vllm.multimodal.processing.BaseDummyInputsBuilder.get_dummy_text] 定义如何基于多模态输入数量生成虚拟文本来解决第一个问题。这使我们能够生成与多模态输入相对应的虚拟文本，并将它们一起输入以获得处理后的多模态数据。

### 自动 prompt 更新

我们通过在 [_apply_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._apply_prompt_updates] 中实现与模型无关的代码来解决第二个问题，该代码根据 [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates] 输出的规范自动使用特征占位符 token 更新 prompt。

### 总结

借助虚拟文本和自动 prompt 更新，我们的多模态 processor 最终可以接受文本 prompt 和 token prompt 以及多模态数据。详细逻辑请参见 [_apply_hf_processor_main][vllm.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_main]。

## Processor 输出缓存

某些 HF processor，例如 Qwen2-VL 的 processor，[非常缓慢](https://github.com/vllm-project/vllm/issues/9238)。为缓解此问题，我们缓存 HF processor 的多模态输出，以避免重复处理相同的多模态输入（例如图像）。

当传入新数据时，我们首先检查哪些项在缓存中，哪些项缺失。缺失的项在一个批次中传入 HF processor 并缓存，然后与缓存中的现有项合并。

由于我们只处理缺失的多模态数据项，输入占位符 token 的数量不再与多模态输入的数量相对应，因此它们不能与文本 prompt 一起传入 HF processor。因此，我们分别处理文本和多模态输入，使用[虚拟文本](#虚拟文本)来避免 HF 错误。由于这跳过了 HF 的 prompt 更新代码，我们在之后应用[自动 prompt 更新](#自动-prompt-更新)以保持输出 token 和多模态数据之间的一致性。
