# 量化 (Quantization)

量化通过牺牲模型精度来换取更小的内存占用，使得大型模型能够在更广泛的设备上运行。

!!! tip
    要开始使用量化，请参阅 [LLM Compressor](llm_compressor.md)，这是一个用于优化模型以在 vLLM 上部署的库，支持 FP8、INT8、INT4 和其他量化格式。

以下是 vLLM 支持的量化格式：

- [AutoAWQ](auto_awq.md)
- [BitsAndBytes](bnb.md)
- [GGUF](gguf.md)
- [GPTQModel](gptqmodel.md)
- [Intel Neural Compressor](inc.md)
- [INT4 W4A16](int4.md)
- [INT8 W8A8](int8.md)
- [FP8 W8A8](fp8.md)
- [NVIDIA Model Optimizer](modelopt.md)
- [在线量化 (Online Quantization)](online.md)
- [AMD Quark](quark.md)
- [量化 KV 缓存 (Quantized KV Cache)](quantized_kvcache.md)
- [TorchAO](torchao.md)
- [FP8 ViT 编码器注意力 (FP8 ViT Encoder Attention)](fp8_vit_attn.md)

## 支持的硬件

下表展示了 vLLM 中各种量化实现在不同硬件平台上的兼容性：

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| 实现 (Implementation)      | Volta | Turing | Ampere | Ada | Hopper | AMD GPU | Intel GPU | x86 CPU |
| ------------------------- | ----- | ------ | ------ | --- | ------ | ------- | --------- | ------- |
| AWQ                       | ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      |
| GPTQ                      | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      |
| Marlin (GPTQ/AWQ/FP8/FP4) | ❌    | ✅︎*    | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      |
| INT8 (W8A8)               | ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ✅︎      |
| FP8 (W8A8)                | ❌    | ❌     | ❌     | ✅︎  | ✅︎     | ✅︎      | ❌        | ❌      |
| bitsandbytes              | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      |
| DeepSpeedFP               | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      |
| GGUF                      | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ✅︎      | ❌        | ❌      |

- Volta 指 SM 7.0，Turing 指 SM 7.5，Ampere 指 SM 8.0/8.6，Ada 指 SM 8.9，Hopper 指 SM 9.0。
- ✅︎ 表示该量化方法在指定硬件上受支持。
- ❌ 表示该量化方法在指定硬件上不受支持。
- 所有 Intel Gaudi 量化支持已迁移至 [vLLM-Gaudi](https://github.com/vllm-project/vllm-gaudi)。
- *Turing 不支持 Marlin MXFP4。

!!! note
    关于 Google TPU 上量化支持的信息，请参阅 [TPU-Inference 推荐模型与特性](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/) 文档。

!!! note
    此兼容性表格可能会随着 vLLM 的持续演进及其对不同硬件平台和量化方法的支持扩展而发生变化。

    关于硬件支持和量化方法的最新信息，请参阅 [vllm/model_executor/layers/quantization](../../../vllm/model_executor/layers/quantization) 或咨询 vLLM 开发团队。

## 树外量化插件 (Out-of-Tree Quantization Plugins)

vLLM 支持使用 `@register_quantization_config` 装饰器注册自定义的树外量化方法。这使得你无需修改 vLLM 代码库即可实现和使用自己的量化方案。

### 注册自定义量化方法

要注册自定义量化方法，创建一个继承自 `QuantizationConfig` 的类并使用 `@register_quantization_config` 进行装饰。`get_quant_method` 根据层类型分派到相应的量化方法：

```python
import torch
from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.fused_moe import FusedMoE

@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):
    """自定义量化配置。"""

    def get_name(self) -> str:
        return "my_quant"

    def get_supported_act_dtypes(self) -> list:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # 最低 GPU 计算能力，-1 表示无限制
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        # 在模型目录中搜索的配置文件
        return []

    @classmethod
    def from_config(cls, config: dict) -> "MyQuantConfig":
        # 从模型的量化配置创建配置
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # 根据层类型分派
        # 注意：你只需要实现你关心的方法
        if isinstance(layer, LinearBase):
            return MyQuantLinearMethod()
        elif isinstance(layer, FusedMoE):
            return MyQuantMoEMethod(layer.moe_config)
        return None
```

### 必需的 QuantizationConfig 方法

你的自定义 `QuantizationConfig` 子类必须实现以下抽象方法：

| 方法 (Method) | 描述 (Description) |
| ------ | ----------- |
| `get_name()` | 返回量化方法的名称 |
| `get_supported_act_dtypes()` | 返回支持的激活数据类型列表（例如 `torch.float16`） |
| `get_min_capability()` | 返回最低 GPU 计算能力（例如 Ampere 为 80，-1 为无限制） |
| `get_config_filenames()` | 返回在模型目录中搜索的配置文件名列表 |
| `from_config(config)` | 类方法，从模型的量化配置字典创建配置 |
| `get_quant_method(layer, prefix)` | 返回给定层的量化方法，或返回 `None` 跳过 |

### 实现量化的线性层方法

对于线性层，从 `get_quant_method` 返回一个 `QuantizeMethodBase` 子类。你可以扩展 `UnquantizedLinearMethod` 作为起点：

```python
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

class MyQuantLinearMethod(UnquantizedLinearMethod):
    """线性层的自定义量化方法。"""

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        # 为层创建量化权重
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 在此应用自定义量化逻辑
        ...
```

### 实现量化的 MoE 方法

对于混合专家 (MoE) 模型，从 `get_quant_method` 返回一个 `FusedMoEMethodBase` 子类。你可以使用 `UnquantizedFusedMoEMethod` 跳过 MoE 量化：

```python
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

class MyQuantMoEMethod(FusedMoEMethodBase):
    """MoE 层的自定义量化方法。"""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # 为 MoE 层创建量化权重
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        router: "FusedMoERouter",
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        # 使用量化权重执行 MoE 计算
        ...

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # 返回 MoE 量化配置
        ...
```

有关参考，请参见 `vllm/model_executor/layers/quantization/fp8.py` 中的 `Fp8MoEMethod` 等现有实现。

### 使用插件

注册完成后，你可以在 vLLM 中使用自定义的量化方法：

```python
# 注册你的量化方法（导入包含配置的模块）
import my_quant_plugin

from vllm import LLM

# 使用自定义量化方法
llm = LLM(model="your-model", quantization="my_quant")
```

关于插件系统的更多信息，请参阅[插件系统文档](../../design/plugin_system.md)。
