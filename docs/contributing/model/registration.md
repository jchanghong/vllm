# 注册模型

vLLM 依赖模型注册表来确定如何运行每个模型。
预注册的架构列表可以在[这里](../../models/supported_models.md)找到。

如果您的模型不在此列表中，则必须将其注册到 vLLM。
本页面提供了如何执行此操作的详细说明。

## 内置模型

要将模型直接添加到 vLLM 库中，请先复刻我们的 [GitHub 仓库](https://github.com/vllm-project/vllm)，然后[从源码构建](../../getting_started/installation/gpu.md#build-wheel-from-source)。
这使您能够修改代码库并测试您的模型。

在您实现了模型之后（请参阅[教程](basic.md)），将其放入 [vllm/model_executor/models](../../../vllm/model_executor/models) 目录。
然后，将您的模型类添加到 [vllm/model_executor/models/registry.py](../../../vllm/model_executor/models/registry.py) 中的 `_VLLM_MODELS`，以便在导入 vLLM 时自动注册。
最后，更新我们的[支持的模型列表](../../models/supported_models.md)来推广您的模型！

!!! important
    每个部分中的模型列表应按字母顺序维护。

## 树外模型

您可以[使用插件](../../design/plugin_system.md)加载外部模型，而无需修改 vLLM 代码库。

要注册模型，请使用以下代码：

```python
# 您的插件的入口点
def register():
    from vllm import ModelRegistry
    from your_code import YourModelForCausalLM

    ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)
```

如果您的模型导入了初始化 CUDA 的模块，请考虑延迟导入以避免 `RuntimeError: Cannot re-initialize CUDA in forked subprocess` 等错误：

```python
# 您的插件的入口点
def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "YourModelForCausalLM",
        "your_code:YourModelForCausalLM",
    )
```

!!! important
    如果您的模型是多模态模型，请确保模型类实现了 [SupportsMultiModal][vllm.model_executor.models.interfaces.SupportsMultiModal] 接口。
    更多信息请参阅[此处](multimodal.md)。
