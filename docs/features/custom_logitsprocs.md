# 自定义 Logits 处理器

!!! important
    一些 logits 处理器的设计变更仍在进行中，API 可能在不久的将来发生变化。我们希望能尽快稳定这部分 API。

"自定义"logits 处理器由 vLLM 用户编写，在初始化时加载到 vLLM 中，无需修改或重新编译 vLLM 源代码。它与内置 logits 处理器相反。

本文档介绍如何编写、加载和使用自定义 logits 处理器。

## Logits 处理器背景

Logits 处理器调整下一个 token 的概率分布，通常旨在引导模型朝期望的行为方向发展。

在 vLLM 中，logits 处理器以批量粒度运行。在给定的引擎步骤中，logits 处理器接收一个形状为 `(num_requests) x (vocab_size)` 的原始 logits 张量（模型输出）。对于所有启用该 logits 处理器的请求，logits 处理器对 logits 张量的相应行应用变换，同时保持其他行不变。转换后的 logits 张量随后传递给 softmax。

## 创建自定义 Logits 处理器

自定义 logits 处理器必须继承 `vllm.v1.sample.logits_processor.LogitsProcessor` 并（至少）定义以下方法：

* `validate_params(cls, sampling_params: SamplingParams)`：
    * 如果 `SamplingParams` 包含 logits 处理器使用的无效参数（尤其是自定义参数），则引发 `ValueError`。
    * 当请求发送到入口点时，`validate_params()` 将验证 `SamplingParams` 并拒绝包含无效参数的请求。
    * **注意：**实现 `validate_params()` 以防止自定义 logits 处理器的无效参数非常重要。否则，包含无效参数的请求可能在自定义 logits 处理器中导致意外行为。

* `__init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool)`
    * `vllm_config`：引擎配置数据结构
    * `device`：硬件加速器设备信息
    * `is_pin_memory`：指示固定内存是否可用于支持 logits 处理器实现的标志

* `apply(self, logits: torch.Tensor) -> torch.Tensor`：
    * 接收形状为 `(num_requests) x (vocab_size)` 的 logits 张量（`logits`）
    * 以批量粒度应用 logits 处理器变换
    * 返回转换后的形状为 `(num_requests) x (vocab_size)` 的 logits 张量
    * 您可以原地或非原地修改输入的 logits 处理器；原地修改更节省内存

* `is_argmax_invariant(self) -> bool`：
    * 如果 logits 处理器对 argmax 不变（从不改变给定请求的最高 logit 值 token ID），则返回 `True`；如果 logits 处理器可能修改 argmax，则返回 `False`
    * `is_argmax_invariant()` 在启动时评估一次；如果为 `True`，当所有请求都使用贪婪采样时，vLLM 将在给定步骤中跳过应用此 logits 处理器

* `update_state(self, batch_update: Optional["BatchUpdate"]) -> None`：
    * 接收一个 `BatchUpdate` 数据结构，表示当前引擎步骤开始时持久批处理状态的变化
    * 使用 `BatchUpdate` 成员更新 logits 处理器的内部状态
    * **注意：**批处理更新数据结构可能为 `None`，表示批处理组成没有变化。在这种情况下，LogitsProcessor 可能仍希望根据更新后的 `output_token_ids` 列表更新其状态（这些列表可能在添加时已保留）。

### vLLM 引擎如何构建 `BatchUpdate` 数据结构

!!! important
    一些 logits 处理器的设计变更仍在进行中。我们预计将来在实现 logits 处理器时，您不需要考虑批处理状态变化，本节中的信息将变得无关紧要。

Logits 处理器的 `update_state()` 实现应假设模型运行器更新持久批处理状态的以下模型（在此以 `BatchUpdate` 抽象表示）：

1. 识别在当前引擎步骤中完成的请求的索引

2. 识别在当前步骤中引入的新请求

3. 使用 Add 操作将尽可能多的已完成请求替换为新请求，按被替换请求索引的升序从最低索引开始

4. 根据新请求和已完成请求的相对数量：

    1. 如果新请求和已完成请求的数量相同，则进入下一步

    2. *如果新请求多于已完成请求：*使用 Add 操作将未替换已完成请求的剩余新请求扩展到批处理中。为这些新请求分配连续索引，从 `current_max_batch_index + 1` 开始

    3. *如果新请求少于已完成请求：*

        * 对未被新请求替换的已完成请求应用 Remove 操作。这些被移除的请求索引必然大于上一步中被替换的已完成请求的最大索引。Remove 操作可能使批处理处于非连续状态

        * **"压缩"批处理使其连续：**从最低索引的空槽开始（由 Remove 导致），应用单向移动，从批处理中当前最高的非空槽填充空槽。按照空槽目标索引递增和非空槽源索引递减的顺序继续执行额外的单向移动操作，直到批处理连续

        * **收缩批处理：**压缩批处理的副作用是，由 Remove 操作导致的空槽被分组在批处理数组末尾的一个连续块中。因此，压缩后，更新 `BatchUpdate.batch_size` 以反映非空槽的数量

5. 为提高效率重新排序批处理。根据注意力后端的实现和批处理的当前特征，可能应用零个或多个 Swap 移动操作来重新排序批处理

注意：

* Logits 处理器的 `update_state()` 方法必须按以下顺序处理批处理更新操作：移除、添加、移动

* Add 操作的索引参数指的是 *添加发生时的索引*，即在任何 Move 操作之前
    * 示例：如果一个请求在索引 5 处添加，然后与索引 3 交换，`BatchUpdate.added` 中的 Add 操作将与索引 5 关联，而不是 3
    * 换句话说，可以假定 Move 操作在 Adds 和 Removes 之后应用

* 可以假定 Move 操作按照它们在 `BatchUpdate.moved` 中出现的顺序应用

* 如果没有新的/完成的请求且没有批处理重新排序，则 logits 处理器的批处理更新将为 `None`

### 向自定义 Logits 处理器传递自定义参数

与内置 logits 处理器不同，自定义 logits 处理器可能需要未硬编码到 `SamplingParams` 或 vLLM 服务器 REST API 中的配置参数。为解决此问题，自定义 logits 处理器可以利用 vLLM 的[自定义参数](./custom_arguments.md)支持来接收用户配置设置（尽管您也可以自由设计利用 `SamplingParams` 中现有字段的自定义 logits 处理器。）

### 自定义 Logits 处理器实现示例

下面的人为示例实现了一个自定义 logits 处理器，它接收形状为 `(num\_requests) \times (vocab\_size)` 的 logits 张量，并使用 `float(-inf)` 屏蔽除一个 token（`target_token`）之外的所有 token。对于未指定 `target_token` 的任何请求，logits 处理器被禁用。为确定 logits 处理器是否已启用以及保留哪个 token，logits 处理器检查每个请求的 `SamplingParams.extra_args` 中的 `target_token` 自定义参数：

??? code "示例自定义 logits 处理器定义"

    ``` python
    import torch
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor import (BatchUpdate,
                                                LogitsProcessor,
                                                MoveDirectionality)

    class DummyLogitsProcessor(LogitsProcessor):
        """用于支持单元测试和示例的虚拟 logit 处理器"""

        @classmethod
        def validate_params(cls, params: SamplingParams):
            target_token: int | None = params.extra_args and params.extra_args.get(
                "target_token"
            )
            if target_token is not None and not isinstance(target_token, int):
                raise ValueError(f"target_token 值 {target_token} 不是整数类型")

        def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                    is_pin_memory: bool):
            self.req_info: dict[int, int] = {}

        def is_argmax_invariant(self) -> bool:
            """从不影响贪婪采样"""
            return False

        def update_state(self, batch_update: BatchUpdate | None):
            if not batch_update:
                return

            # 处理添加的请求。
            for index, params, _, _ in batch_update.added:
                assert params is not None
                self.validate_params(params)
                if params.extra_args and (target_token :=
                                        params.extra_args.get("target_token")):
                    self.req_info[index] = target_token
                else: 
                    self.req_info.pop(index, None)

            if self.req_info:
                # 处理移除的请求。
                for index in batch_update.removed:
                    self.req_info.pop(index, None)

                # 处理移动的请求，单向移动（a->b）和交换（a<->b）
                for adx, bdx, direct in batch_update.moved:
                    a_val = self.req_info.pop(adx, None)
                    b_val = self.req_info.pop(bdx, None)
                    if a_val is not None:
                        self.req_info[bdx] = a_val
                    if direct == MoveDirectionality.SWAP and b_val is not None:
                        self.req_info[adx] = b_val

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if not self.req_info:
                return logits

            # 在修改前保存目标值
            cols = torch.tensor(
                list(self.req_info.values()), dtype=torch.long, device=logits.device
            )
            rows = torch.tensor(
                list(self.req_info.keys()), dtype=torch.long, device=logits.device
            )
            values_to_keep = logits[rows, cols].clone()

            # 屏蔽除目标 token 外的所有 token
            logits[rows] = float('-inf')
            logits[rows, cols] = values_to_keep

            return logits

    ```

在本文档的其余部分，我们将使用 `DummyLogitsProcessor` 作为自定义 logits 处理器的示例。

`DummyLogitsProcessor.update_state()` 实现维护了一个"稀疏"表示的批量请求，存储在 `self.req_info` 字典中：只有指定了 `target_token` 值的请求才会在字典中有键。`update_state()` 根据针对持久批处理的 Add、Remove 和 Move 操作，调整存储的请求索引和 `target_token` 值（分别为 `self.req_info` 中的键和值）。

### 包装现有的请求级 Logits 处理器

尽管 vLLM 引擎以批量粒度应用 logits 处理器，但一些用户可能希望将 vLLM 与"请求级"logits 处理器实现一起使用——即对单个请求进行操作的处理实现。如果您的 logits 处理器是为 vLLM 版本 0 开发的，情况尤其如此，该版本要求它是符合以下类型注解的 `Callable`（如[此处][vllm.logits_process]所述）：

``` python
RequestLogitsProcessor = Union[

    # (output token ids, logits tensor) -> logits tensor
    Callable[[list[int], Tensor], Tensor],

    # (prompt token ids, output token ids, logits tensor) -> logits tensor
    Callable[[list[int], list[int], Tensor], Tensor],
]
```

虽然请求级 logits 处理器明确*不*受 vLLM 引擎支持，但 vLLM *确实*提供了一种便捷过程来包装现有的 `Callable` 请求级 logits 处理器，并创建与 vLLM 兼容的批量级 logits 处理器。`Callable` 必须符合上述类型注解；如果您的请求级 logits 处理器具有不同的接口，则需要进行修改或实现额外的包装层以符合上述接口规范。

您可以通过继承 `AdapterLogitsProcessor` 来包装请求级 logits 处理器，如下例所示（在此示例中，`DummyPerReqLogitsProcessor` 代表您需要包装的请求级 logits 处理器）：

* 重写 `AdapterLogitsProcessor.validate_params(cls,params)` 以验证请求的采样参数。

* 重写 `AdapterLogitsProcessor.is_argmax_invariant(self)` 以准确反映您的请求级 logits 处理器是否可能影响具有最高 logit 值的 token。

* 重写 `AdapterLogitsProcessor.new_req_logits_processor(self,params)` 以从 `SamplingParams` 实例创建新的请求级 logits 处理器实例：

??? code "包装请求级 Logits 处理器的示例"

    ``` python
    ...

    from vllm.v1.sample.logits_processor import (
        AdapterLogitsProcessor, # 包装器基类
        RequestLogitsProcessor, # 请求级 logitsproc 类型注解
    )

    ...

    # 替代您的请求级 logits 处理器：
    class DummyPerReqLogitsProcessor:
        """请求级 logits 处理器，屏蔽除
        `target_token` 标识的 token ID 之外的所有 logits"""

        def __init__(self, target_token: int) -> None:
            """指定 `target_token`"""
            self.target_token = target_token

        def __call__(
            self,
            output_ids: list[int],
            logits: torch.Tensor,
        ) -> torch.Tensor:
            val_to_keep = logits[self.target_token].item()
            logits[:] = float("-inf")
            logits[self.target_token] = val_to_keep
            return logits

    ...

    # 包装请求级 logits 处理器的示例：
    class WrappedPerReqLogitsProcessor(AdapterLogitsProcessor):
        """包装虚拟请求级 logit 处理器以创建
        批量级 logits 处理器的示例"""

        @classmethod
        def validate_params(cls, params: SamplingParams):
            target_token: Any | None = params.extra_args and params.extra_args.get(
                "target_token"
            )
            if target_token is not None and not isinstance(target_token, int):
                raise ValueError(
                    f"target_token 值 {target_token} 不是整数类型"
                )

        def is_argmax_invariant(self) -> bool:
            return False

        def new_req_logits_processor(
            self,
            params: SamplingParams,
        ) -> Optional[RequestLogitsProcessor]:
            """此方法返回一个新的请求级 logits 处理器，根据与特定请求关联的
            `target_token` 值进行定制。

            如果 logits 处理器不应应用于特定请求，则返回 None。
            要使用 logits 处理器，请求必须有一个整数值的 "target_token" 自定义参数。

            参数：
            params：每个请求的采样参数

            返回：
            `Callable` 请求 logits 处理器，或 None
            """
            target_token: Any | None = params.extra_args and params.extra_args.get(
                "target_token"
            )
            if target_token is None:
                return None
            return DummyPerReqLogitsProcessor(target_token)
    ```

!!! note
    您的 `new_req_logits_processor()` 重写可以返回 `None`，以表示包装的 logits 处理器不应用于所讨论的请求。

一旦您创建了包装请求级 logits 处理器的自定义子类（如 `WrappedPerReqLogitsProcessor`），您可以通过下一节中描述的任何方法将自定义子类传递给 vLLM。

## 在 vLLM 中加载自定义 Logits 处理器的方法

Logits 处理器在初始化时加载。关键的是，已加载的 logits 处理器集合在 vLLM 引擎完成加载后无法修改，并且不能按需为单个请求加载新的 logits 处理器。

本节详细介绍使您的 logits 处理器对 vLLM 可见并触发 vLLM 加载 logits 处理器的不同方法。

### 方法 1：在初始化时将自定义 Logits 处理器的完全限定类名（FQCN）传递给 vLLM

此方法在 vLLM 的离线和在线使用场景中都受支持。自定义 logits 处理器的 FQCN（格式为 `dotted.path.to.module:ClassName`）可以作为参数传递给 `LLM` 和 `AsyncLLM` Python 构造函数，或作为 CLI 参数传递给 `vllm serve`，语法如下：

``` bash
vllm serve ... --logits_processors <logits 处理器 1> <logits 处理器 2> ...
```

对 FQCN 的唯一要求是：

1. Python 的 `importlib.import_module()` 必须能够解析 FQCN 的点分隔路径部分并将其作为模块加载

2. FQCN 的类名部分必须能够从加载的模块中导入

3. FQCN 指向的对象必须是 `LogitsProcessor` 的子类

请参见以下示例：

??? code "在 Python 中将自定义 logits 处理器 FQCN 传递给 `LLM`"

    ``` python
    # 传递 FQCN
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=["your.module.path:DummyLogitsProcessor"],
    )
    ```

??? code "在 Python 中将自定义 logits 处理器 FQCN 传递给 `AsyncLLM`"

    ``` python
    # 传递 FQCN
    engine_args = AsyncEngineArgs(model="facebook/opt-125m",
                                  logits_processors=["your.module.path:DummyLogitsProcessor"])
    async_llm = AsyncLLM.from_engine_args(engine_args)
    ```

??? code "通过 CLI 将自定义 logits 处理器 FQCN 传递给 vLLM 服务器"

    ```bash
    vllm serve facebook/opt-125m --logits_processors your.module.path:DummyLogitsProcessor
    ```

### 方法 2：自动检测 Python 环境中作为入口点安装的自定义 Logits 处理器

[`setuptools`](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) 可以使已安装的包通过称为"入口点"的元数据片段，作为插件提供给其他 Python 程序。

在初始化期间，vLLM 自动扫描 `vllm.logits_processors` 入口点组，并加载它找到的所有已安装的 logits 处理器。

假设您开发了一个包含自定义 logits 处理器的 Python 包。您可以通过为每个 logits 处理器向您的 logits 处理器 Python 包添加唯一入口点，将每个 logits 处理器暴露给 vLLM。下面的示例演示如何在项目的 `pyproject.toml` 文件中添加入口点：

??? code "将自定义 logits 处理器暴露为 Python 入口点"

    ``` toml
    [project.entry-points."vllm.logits_processors"]
    dummy_logits_processor = "your.module.path:DummyLogitsProcessor"
    ```

安装您的包后，每当 vLLM 初始化时，自定义 logits 处理器将自动加载。如果您的 logits 处理器作为入口点暴露，您*不*需要在初始化时显式将自定义 logits 处理器传递给 `LLM` 或 `AsyncLLM` 构造函数或 vLLM 服务器。

!!! note
    vLLM 将*始终*加载通过 `vllm.logits_processors` 分组入口点暴露的*所有* logits 处理器。

### 方法 3（仅限离线）：将 Python 类对象传递给 vLLM 构造函数

您可以将一个或多个自定义 logits 处理器类对象传递给 `LLM` 和 `AsyncLLM` 构造函数。此选项非常灵活，因为 logits 处理器类可以是 (1) 在实例化 `LLM` 或 `AsyncLLM` 的同一 Python 源文件中本地定义，或 (2) 从 Python 包中导入。

??? code "在 Python 中将自定义 logits 处理器类对象传递给 `LLM` 或 `AsyncLLM`"

    ``` python
    # 导入自定义 logits 处理器
    from some.module import DummyLogitsProcessor

    # ...或者...

    # 本地定义自定义 logits 处理器
    from vllm.v1.sample.logits_processor import LogitsProcessor

    class DummyLogitsProcessor(LogitsProcessor):
        # 请参阅上面的 DummyLogitsProcessor 实现
        ...

    # 将类对象传递给 LLM 构造函数
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=[DummyLogitsProcessor],
    )

    # 将类对象传递给 AsyncLLM 构造函数
    engine_args = AsyncEngineArgs(model="facebook/opt-125m",
                                  logits_processors=[DummyLogitsProcessor])
    async_llm = AsyncLLM.from_engine_args(engine_args)
    ```

## 针对请求调用自定义 Logits 处理器

自定义 logits 处理器的设计决定了该 logits 处理器是否必须为给定请求启用/禁用，以及需要提供哪些参数来配置 logits 处理器。

下面的示例展示了用户如何向 `DummyLogitsProcessor` 传递自定义参数（`target_token`），以 (1) 为该特定请求启用 logits 处理器，以及 (2) 控制 logits 处理器的行为。

??? code "vLLM REST API：为请求配置自定义 logits 处理器"

    ``` bash
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            ...
            "vllm_xargs": {"target_token": 67}
        }'
    ```

??? code "OpenAI SDK：为请求配置自定义 logits 处理器"

    ``` python
    batch = await client.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        ...,
        extra_body={
            "vllm_xargs": {
                "target_token": 67
            }
        }
    )
    ```

??? code "离线：为 `LLM` 请求配置自定义 logits 处理器"

    ``` python
    outputs_logitproc = llm.generate("your prompt", 
                                     SamplingParams(...,
                                        extra_args={"target_token": 67}))
    ```

??? code "离线：为 `AsyncLLM` 请求配置自定义 logits 处理器"

    ``` python
    async for out in engine.generate(request_id="your request id",
                                     prompt="your prompt",
                                     sampling_params=SamplingParams(...,
                                        extra_args={"target_token": 67})):

        # 处理异步请求输出
        ...
    ```

## 编写自定义 Logits 处理器的最佳实践

一旦 vLLM 在初始化时加载了 logits 处理器，vLLM 将在每个引擎步骤中对该 logits 处理器调用 `update_state()` 和 `apply()`。这两种方法都作用于当前存在于 vLLM 持久批处理中的所有请求。因此，高效实现这些方法非常重要。

* 鉴于 logits 处理器以批量粒度运行，编写高效的 `apply()` 和 `update_state()` 实现
    * 例如，您可以使用高效的向量化操作来实现 `apply()`，或在 `update_state()` 中更新内部状态向量
    * 但是，如果您认为某个 logits 处理器可能不经常使用，则使用请求状态的"稀疏"表示可能是合适的，即该类可以使用仅存储启用该 logits 处理器的请求元数据的字典
    * **注意：**包装的请求级 logits 处理器不需要实现 `apply()` 和 `update_state()`；默认的 `AdapterLogitsProcessor.update_state()` 实现维护了请求状态的稀疏表示，其中 `new_req_logits_processor()` 返回 `None` 的请求不包含在基类状态字典中。`AdapterLogitsProcessor.apply()` 的默认实现顺序将请求级 logits 处理器应用于输入 logits 的每一行，并组装输出 logits 张量。如果此 `AdapterLogitsProcessor` 默认实现的性能不足，则避免包装请求级 logits 处理器，而是将其重新实现为具有优化的 `apply()` 和 `update_state()` 实现的 `LogitsProcessor` 子类，以批量粒度操作

* 由 logits 处理器作者决定：

    1. **配置 logits 处理器针对该请求行为的每个请求属性。**自定义 logits 处理器的 `update_state()` 重写决定如何将 `SamplingParams` 字段映射到 logits 处理器状态

        * **注意：**对于包装的请求级 logits 处理器，`new_req_logits_processor()` 决定如何使用 `SamplingParams` 字段来初始化请求级 logits 处理器实例。

    2. **在逐个请求的基础上启用或禁用 logits 处理器的条件。**除非您希望自定义 logits 处理器始终对所有请求起作用，否则应编写 logits 处理器，使其能够为给定请求禁用，即通过将参数默认为 `None` 或传入特定的无操作参数值（如 `0.0`）。尝试为禁用 logits 处理器的请求节省计算和内存

        * **注意：**对于包装的逐请求 logits 处理器，默认的 `AdapterLogitsProcessor.update_state()` 实现确保当 `new_req_logits_processor()` 为该请求返回 `None` 时，请求级 logits 处理器被禁用

    3. **在批量级别短路 logits 处理器的条件。**即使您已经定义了在请求级别禁用自定义 logits 处理器的方法，这也可能难以转化为计算节省，例如，如果您的 `update_state()` 和 `apply()` 实现使用对整个持久批处理在单个命令中操作的高效向量化实现。例如，如果只有一个请求禁用了 logits 处理器，您无法跳过 `apply()` 中的整个向量化操作。为了在没有运行中的请求使用自定义 logits 处理器的边缘情况下节省计算，我们建议设计 `apply()`，如果所有请求都禁用了 logits 处理器，则返回未修改的输入张量。同样，考虑在没有请求启用 logits 处理器时是否可以跳过 `update_state()` 中的步骤

        * 此外，在 `update_state()` 中节省计算的一个简单方法是，当 `batch_update` 为 `None` 时提前退出

        * **注意：**对于包装的逐请求 logits 处理器，`AdapterLogitsProcessor` 基类默认实现了上述优化

* 确保 logits 处理器的 `update_state` 方法丢弃已完成请求的信息（即被 Add 替换或受到 Remove 操作的请求）

    * **注意：**对于包装的逐请求 logits 处理器，`AdapterLogitsProcessor` 基类默认处理此问题

* `is_argmax_invariant()` 可以在 logits 处理器具有一致行为时硬编码为 `True` 或 `False`。然而，argmax 不变性也可以编程方式确定（即，如果您的 logits 处理器是用户可定制的，影响了 logits 处理器是否对 argmax 不变）。因此，`is_argmax_invariant()` 不是一个类方法。
