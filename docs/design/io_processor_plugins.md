# IO 处理器插件

IO 处理器插件是一种功能，允许对池化模型的模型输入和输出进行预处理和后处理。其思路是允许用户向 vLLM 传递自定义输入，该输入被转换成一个或多个模型 prompt 并馈送到模型的 `encode` 方法。此类插件的一个潜在用途是使用 vLLM 生成多模态数据。例如，用户向 vLLM 输入一张图像并得到一张图像作为输出。

当使用 IO 处理器插件执行推理时，prompt 类型由插件定义，最终请求输出也是如此。vLLM 不对输入/输出数据执行任何验证，由插件确保正确的数据被馈送到模型并返回给用户。目前，这些插件仅支持池化模型，可以通过 `LLM` 和 `AsyncLLM` 中的 `encode` 方法触发，或通过在线服务模式下的 `/pooling` 端点触发。

## 编写 IO 处理器插件

IO 处理器插件实现了 [`IOProcessor`][vllm.plugins.io_processors.interface.IOProcessor] 接口：

```python
IOProcessorInput = TypeVar("IOProcessorInput")
IOProcessorOutput = TypeVar("IOProcessorOutput")

class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):
    """引擎 I/O 预处理/后处理的抽象接口。"""

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):
        super().__init__()

        self.vllm_config = vllm_config

    def parse_data(self, data: object) -> IOProcessorInput:
        raise NotImplementedError

    def merge_sampling_params(
        self,
        params: SamplingParams | None = None,
    ) -> SamplingParams:
        return params or SamplingParams()

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        return params or PoolingParams(task="plugin")

    @abstractmethod
    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        raise NotImplementedError

    async def post_process_async(
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        # 我们无法保证输出返回的顺序与
        # 输入 vLLM 时的顺序相同。
        # 在后处理之前按 id 排序
        sorted_output = sorted(
            [(i, item) async for i, item in model_output], key=lambda output: output[0]
        )
        collected_output = [output[1] for output in sorted_output]
        return self.post_process(collected_output, request_id=request_id, **kwargs)
```

`parse_data` 方法用于验证用户数据并将其转换为 `pre_process*` 方法期望的输入。
`merge_sampling_params` 和 `merge_pooling_params` 方法将输入的 `SamplingParams` 或 `PoolingParams`（如果有）与默认值合并。
`pre_process*` 方法接受经过验证的插件输入以生成用于常规推理的 vLLM 模型 prompt。
`post_process*` 方法接受 `PoolingRequestOutput` 对象作为输入并生成自定义的插件输出。

一个使用 PrithviGeospatialMAE 模型生成 geotiff 图像的插件实现示例可在[此处](https://github.com/IBM/terratorch/tree/main/terratorch/vllm/plugins/segmentation)找到。同时，请参考我们的在线（[examples/pooling/plugin/prithvi_geospatial_mae_online.py](../../examples/pooling/plugin/prithvi_geospatial_mae_online.py)）和离线（[examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py](../../examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py)）推理示例。

## 使用 IO 处理器插件

IO 处理器插件在引擎启动时加载，有两种方法指定要加载的插件名称：

1. 通过 vLLM 的 `EngineArgs`：在用于初始化 `AsyncLLM` 的 `EngineArgs` 中设置 `io_processor_plugin` 参数。在离线模式下，也可以通过将 `io_processor_plugin` 参数传递给 `LLM` 来实现；在服务模式下，通过传递 `--io-processor-plugin` 参数来实现。
2. 通过模型 HF 配置：在模型配置（config.json）中添加 `io_processor_plugin` 字段。

顺序也决定了方法的优先级。即，通过 `EngineArgs` 设置插件名称将覆盖模型 HF 配置（config.json）中指定的任何插件名称。
