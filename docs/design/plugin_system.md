# 插件系统

社区经常要求扩展 vLLM 以添加自定义功能。为了方便这一点，vLLM 包含一个插件系统，允许用户在不修改 vLLM 代码库的情况下添加自定义功能。本文档解释了插件在 vLLM 中的工作原理以及如何为 vLLM 创建插件。

## 插件在 vLLM 中的工作原理

插件是用户注册的、由 vLLM 执行的代码。鉴于 vLLM 的架构（参见[架构概述](arch_overview.md)），尤其在使用各种并行技术的分布式推理时，可能涉及多个进程。为了成功启用插件，vLLM 创建的每个进程都需要加载该插件。这是通过 `vllm.plugins` 模块中的 [load_plugins_by_group][vllm.plugins.load_plugins_by_group] 函数完成的。

## vLLM 如何发现插件

vLLM 的插件系统使用标准的 Python `entry_points` 机制。该机制允许开发者在他们的 Python 包中注册函数以供其他包使用。插件示例如下：

??? code

    ```python
    # 在 `setup.py` 文件中
    from setuptools import setup

    setup(name='vllm_add_dummy_model',
        version='0.1',
        packages=['vllm_add_dummy_model'],
        entry_points={
            'vllm.general_plugins':
            ["register_dummy_model = vllm_add_dummy_model:register"]
        })

    # 在 `vllm_add_dummy_model/__init__.py` 文件中
    def register():
        from vllm import ModelRegistry

        if "MyLlava" not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(
                "MyLlava",
                "vllm_add_dummy_model.my_llava:MyLlava",
            )
    ```

有关向包中添加 entry points 的更多信息，请查看[官方文档](https://setuptools.pypa.io/en/latest/userguide/entry_point.html)。

每个插件包含三个部分：

1. **插件组（Plugin group）**：entry point 组的名称。vLLM 使用 entry point 组 `vllm.general_plugins` 来注册通用插件。这是 `setup.py` 文件中 `entry_points` 的键。始终为 vLLM 的通用插件使用 `vllm.general_plugins`。
2. **插件名称（Plugin name）**：插件的名称。这是 `entry_points` 字典中字典部分的值。在上面的示例中，插件名称是 `register_dummy_model`。可以使用 `VLLM_PLUGINS` 环境变量按名称筛选插件。要仅加载特定插件，请将 `VLLM_PLUGINS` 设置为该插件名称。
3. **插件值（Plugin value）**：要在插件系统中注册的函数或模块的完全限定名称。在上面的示例中，插件值是 `vllm_add_dummy_model:register`，它引用了 `vllm_add_dummy_model` 模块中名为 `register` 的函数。

## 支持的插件类型

- **通用插件**（组名 `vllm.general_plugins`）：这些插件的主要用例是将自定义的、非树内模型注册到 vLLM 中。这是通过在插件函数内调用 `ModelRegistry.register_model` 来完成的。有关官方模型插件的示例，请参见 [bart-plugin](https://github.com/vllm-project/bart-plugin)，它增加了对 `BartForConditionalGeneration` 的支持。

- **平台插件**（组名 `vllm.platform_plugins`）：这些插件的主要用例是将自定义的、非树内平台注册到 vLLM 中。当当前环境不支持该平台时，插件函数应返回 `None`；当支持时，返回平台类的完全限定名称。

- **IO 处理器插件**（组名 `vllm.io_processor_plugins`）：这些插件的主要用例是为池化模型注册自定义的模型 prompt 预处理/后处理和模型输出后处理。插件函数返回 IOProcessor 类的完全限定名称。

- **统计日志插件**（组名 `vllm.stat_logger_plugins`）：这些插件的主要用例是将自定义的、非树内的日志器注册到 vLLM 中。entry point 应该是一个继承自 `StatLoggerBase` 的类。

## 编写插件的指南

- **可重入性**：entry point 中指定的函数应该是可重入的，这意味着它可以被多次调用而不会引起问题。这是必要的，因为该函数在某些进程中可能会被多次调用。

### 平台插件指南

1. 创建一个平台插件项目，例如 `vllm_add_dummy_platform`。项目结构应如下所示：

    ```shell
    vllm_add_dummy_platform/
    ├── vllm_add_dummy_platform/
    │   ├── __init__.py
    │   ├── my_dummy_platform.py
    │   ├── my_dummy_worker.py
    │   ├── my_dummy_attention.py
    │   ├── my_dummy_device_communicator.py
    │   ├── my_dummy_custom_ops.py
    ├── setup.py
    ```

2. 在 `setup.py` 文件中，添加以下 entry point：

    ```python
    setup(
        name="vllm_add_dummy_platform",
        ...
        entry_points={
            "vllm.platform_plugins": [
                "my_dummy_platform = vllm_add_dummy_platform:register"
            ]
        },
        ...
    )
    ```

    请确保 `vllm_add_dummy_platform:register` 是一个可调用函数，并返回平台类的完全限定名称。例如：

    ```python
    def register():
        return "vllm_add_dummy_platform.my_dummy_platform.MyDummyPlatform"
    ```

3. 在 `my_dummy_platform.py` 中实现平台类 `MyDummyPlatform`。平台类应继承自 `vllm.platforms.interface.Platform`。请按照接口逐个实现函数。以下是一些必须至少实现的重要函数和属性：

    - `_enum`：该属性是来自 [PlatformEnum][vllm.platforms.interface.PlatformEnum] 的设备枚举值。通常，它应该是 `PlatformEnum.OOT`，表示该平台是非树内的。
    - `device_type`：该属性应返回 pytorch 使用的设备类型。例如，`"cpu"`、`"cuda"` 等。
    - `device_name`：该属性通常设置为与 `device_type` 相同。主要用于日志记录。
    - `check_and_update_config`：该函数在 vLLM 初始化过程的早期被调用。用于插件更新 vLLM 配置。例如，块大小、graph 模式配置等可以在该函数中更新。最重要的是，应该在此函数中设置 **worker_cls**，以让 vLLM 知道为工作进程使用哪个工作器类。
    - `get_attn_backend_cls`：该函数应返回注意力后端类的完全限定名称。
    - `get_device_communicator_cls`：该函数应返回设备通信器类的完全限定名称。

4. 在 `my_dummy_worker.py` 中实现工作器类 `MyDummyWorker`。工作器类应继承自 [WorkerBase][vllm.v1.worker.worker_base.WorkerBase]。请按照接口逐个实现函数。基本上，基类中的所有接口都应该实现，因为它们在 vLLM 中各处被调用。为了确保模型可以执行，需要实现的基本函数包括：

    - `init_device`：该函数用于为工作器设置设备。
    - `initialize_cache`：该函数用于为工作器设置缓存配置。
    - `load_model`：该函数用于将模型权重加载到设备上。
    - `get_kv_cache_spec`：该函数用于生成模型的 KV cache 规范。
    - `determine_available_memory`：该函数用于分析模型的峰值内存使用情况，以确定在不发生 OOM 的情况下有多少内存可用于 KV cache。
    - `initialize_from_config`：该函数用于使用指定的 `kv_cache_config` 分配设备 KV cache。
    - `execute_model`：该函数在每一步被调用以执行模型推理。

    可以实现的其他函数包括：

    - 如果插件想支持休眠模式功能，请实现 `sleep` 和 `wakeup` 函数。
    - 如果插件想支持 graph 模式功能，请实现 `compile_or_warm_up_model` 函数。
    - 如果插件想支持推测解码功能，请实现 `take_draft_token_ids` 函数。
    - 如果插件想支持 lora 功能，请实现 `add_lora`、`remove_lora`、`list_loras` 和 `pin_lora` 函数。
    - 如果插件想支持数据并行功能，请实现 `execute_dummy_batch` 函数。

    有关更多可以实现的函数，请查看工作器基类 [WorkerBase][vllm.v1.worker.worker_base.WorkerBase]。

5. 在 `my_dummy_attention.py` 中实现注意力后端类 `MyDummyAttention`。注意力后端类应继承自 [AttentionBackend][vllm.v1.attention.backend.AttentionBackend]。它用于使用您的设备计算注意力。以 `vllm.v1.attention.backends` 为例，它包含许多注意力后端的实现。

6. 为高性能实现自定义算子。大多数算子可以通过 pytorch 原生实现运行，但性能可能不佳。在这种情况下，您可以为插件实现特定的自定义算子。目前，vLLM 支持以下几种自定义算子：

    - pytorch 算子
      有三种 pytorch 算子：

        - `communicator ops`：设备通信器算子，如 all-reduce、all-gather 等。
          请在 `my_dummy_device_communicator.py` 中实现设备通信器类 `MyDummyDeviceCommunicator`。设备通信器类应继承自 [DeviceCommunicatorBase][vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase]。
        - `common ops`：通用算子，如 matmul、softmax 等。
          请通过注册 OOT 方式实现通用算子。更多详情请参见 [CustomOp][vllm.model_executor.custom_op.CustomOp] 类。
        - `csrc ops`：C++ 算子。这类算子用 C++ 实现，并注册为 torch 自定义算子。
          请遵循 csrc 模块和 `vllm._custom_ops` 来实现您的算子。

    - triton 算子
      Triton 算子目前不支持自定义方式。

7. （可选）实现其他可插拔模块，如 lora、graph 后端、量化、mamba 注意力后端等。

## 兼容性保证

vLLM 保证记录的插件接口（如 `ModelRegistry.register_model`）始终可用于插件注册模型。但是，插件开发者有责任确保其插件与目标 vLLM 版本的兼容性。例如，`"vllm_add_dummy_model.my_llava:MyLlava"` 应与插件目标版本的 vLLM 兼容。

模型/模块的接口在 vLLM 开发过程中可能会发生变化。如果您看到任何弃用日志信息，请将您的插件升级到最新版本。

## 弃用公告

!!! warning "弃用说明"
    - `Platform.get_attn_backend_cls` 中的 `use_v1` 参数已弃用。已在 v0.13.0 中移除。
    - `vllm.attention` 中的 `_Backend` 已弃用。已在 v0.13.0 中移除。请改用 `vllm.v1.attention.backends.registry.register_backend` 向 `AttentionBackendEnum` 添加新的注意力后端。
    - `seed_everything` 平台接口已弃用。已在 v0.16.0 中移除。请改用 `vllm.utils.torch_utils.set_random_seed`。
    - `Platform.validate_request` 中的 `prompt` 已弃用。已在 v0.18.0 中移除。
