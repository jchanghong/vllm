# 概要

## 配置

vLLM 配置类的 API 文档。

- [vllm.config.ModelConfig][]
- [vllm.config.CacheConfig][]
- [vllm.config.LoadConfig][]
- [vllm.config.ParallelConfig][]
- [vllm.config.SchedulerConfig][]
- [vllm.config.DeviceConfig][]
- [vllm.config.SpeculativeConfig][]
- [vllm.config.LoRAConfig][]
- [vllm.config.MultiModalConfig][]
- [vllm.config.PoolerConfig][]
- [vllm.config.StructuredOutputsConfig][]
- [vllm.config.ProfilerConfig][]
- [vllm.config.ObservabilityConfig][]
- [vllm.config.KVTransferConfig][]
- [vllm.config.CompilationConfig][]
- [vllm.config.VllmConfig][]

## 离线推理

LLM 类。

- [vllm.LLM][]

LLM API 的提示词模式。

- [vllm.inputs.llm][]

## vLLM 引擎

用于离线和在线推理的引擎类。

- [vllm.LLMEngine][]
- [vllm.AsyncLLMEngine][]

## 推理参数

vLLM API 的推理参数。

- [vllm.SamplingParams][]
- [vllm.PoolingParams][]

## 多模态

vLLM 通过 [vllm.multimodal][] 包为多模态模型提供实验性支持。

多模态输入可以与文本和 token 提示一起传递给[支持的模型](../models/supported_models.md#list-of-multimodal-language-models)，
通过 [vllm.inputs.PromptType][] 中的 `multi_modal_data` 字段。

想要添加自己的多模态模型？请按照[此处](../contributing/model/multimodal.md)的说明操作。

- [vllm.multimodal.MULTIMODAL_REGISTRY][]

### 内部数据结构

- [vllm.multimodal.inputs.PlaceholderRange][]
- [vllm.multimodal.inputs.NestedTensors][]
- [vllm.multimodal.inputs.MultiModalFieldElem][]
- [vllm.multimodal.inputs.MultiModalFieldConfig][]
- [vllm.multimodal.inputs.MultiModalKwargsItem][]
- [vllm.multimodal.inputs.MultiModalKwargsItems][]

### 数据解析

- [vllm.multimodal.parse][]

### 数据处理

- [vllm.multimodal.processing][]

### 注册表

- [vllm.multimodal.registry][]

## 模型开发

- [vllm.model_executor.models.interfaces_base][]
- [vllm.model_executor.models.interfaces][]
- [vllm.model_executor.models.adapters][]
