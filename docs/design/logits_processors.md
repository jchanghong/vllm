# Logits 处理器

!!! important
    一些 logits 处理器的设计更改仍在进行中，API 可能在不久的将来发生变化。我们希望尽快稳定这部分 API。

本文档描述了 vLLM 引擎如何与 logits 处理器交互，以及 vLLM 支持用于实现 logits 处理器的编程模型。

## Logits 处理器背景

Logits 处理器调整下一个 token 的概率分布，通常旨在引导模型朝着期望的行为方向发展。

在 vLLM 中，logits 处理器以批次粒度运行。在给定的引擎步骤中，logits 处理器使用一个 `(num_requests) x (vocab_size)` 的原始 logits 张量（模型输出）。对于所有启用 logits 处理器的请求，logits 处理器对 logits 张量的对应行应用变换，同时保持其他行不变。变换后的 logits 张量随后传递给 softmax。

## vLLM 引擎中的 Logits 处理器

vLLM 引擎的持久批处理数据结构维护了一个已加载的 logits 处理器列表。

为了在单个批次上操作，每个 logits 处理器可以维护关于批次中请求的元数据（即每个请求的 logits 处理器特定配置设置）。因此，logits 处理器是有状态的。

在每个引擎步骤中，vLLM 引擎将 (1) 更新每个 logits 处理器的内部状态，以及 (2) 将 logits 处理器应用于模型输出的 logits。

### 更新 Logits 处理器内部状态

在每个引擎步骤开始时，持久批处理可能根据调度器输出添加、丢弃和/或重新排序请求。在持久批处理重新组织后，vLLM 引擎调用每个 logits 处理器的 `update_state()` 方法。这对于确保 logits 处理器的内部状态被重新组织以匹配引擎步骤开始时新的持久批处理状态是必要的。

下面的伪代码显示了 vLLM 持久批处理通知每个 logits 处理器批次状态变化的过程：

??? code "模型运行器更新 Logits 处理器状态"

    ``` python
    # gpu_model_runner.py

    class GPUModelRunner(...):

        ...

        def execute_model(self, scheduler_output, ...):
            self._update_states(scheduler_output)

            ...

        def _update_states(...):

            ...

            # ...更新持久批处理以反映新的/完成的请求以及批次内请求的重新排序...

            ...

            self.input_batch.refresh_metadata()


    # gpu_input_batch.py

    class InputBatch:

        ...

        def refresh_metadata(self):

            ...

            # 更新每个 logits 处理器的状态以反映持久批处理状态
            batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
            for logit_proc in self.logitsprocs.all:
                logit_proc.update_state(batch_update)

            ...


    # vllm/v1/sample/logits_processor/interface.py

    @dataclass(frozen=True)
    class BatchUpdate:
        # 批次状态更改数据结构，传递给 logits 处理器的
        # update_state() 方法

        batch_size: int

        removed: Sequence[RemovedRequest]
        added: Sequence[AddedRequest]
        moved: Sequence[MovedRequest]
    
    ```

### 将 Logits 处理器应用于模型输出的 Logits

更新持久批处理状态后，vLLM 模型运行器执行模型推理以获得 logits。然后，模型运行器对 logits 调用采样器。反过来，采样器操作的一部分是调用 logits 处理器的 `apply()` 方法对模型输出的 logits 进行处理，产生变换后的 logits（`apply()` 方法可以就地或非就地修改 logits，尽管就地修改更节省内存）。此过程显示在下面的伪代码中。

请注意，采样器将通过 `SamplingMetadata.logitsprocs` 访问 logits 处理器。当 vLLM 引擎构造 `SamplingMetadata`（未在下面的代码中显示）时，logits 处理器列表的引用从持久批处理数据结构传递到 `SamplingMetadata`。

??? code "将 logits 处理器应用于模型输出的 logits"

    ``` python
    # gpu_model_runner.py

    class GPUModelRunner(...):

        ...

        def execute_model(self, scheduler_output, ...):
            #（在前一节中讨论过）
            self._update_states(scheduler_output)

            ...

            # ...运行模型推理以获得 logits...

            ...

            # 调用采样器，它将应用 logits 处理器
            sampler_output = self.sampler(logits=logits,
                                          sampling_metadata=sampling_metadata)

            ...


    # sampler.py

    class Sampler(nn.Module):

        ...

        def forward(self, logits, sampling_metadata):

            ...

            # 将非 argmax 不变的 logits 处理器应用于模型输出的 logits
            for processor in (sampling_metadata.logitsprocs.non_argmax_invariant):
                logits = processor.apply(logits)

            sampled = self.sample(logits, sampling_metadata)

            ...

            # ...返回采样器输出数据结构...


        def sample(self, logits, sampling_metadata)

            ...

            # ...如果所有请求都是贪心采样，则提前退出...

            ...

            # 应用 argmax 不变的 logits 处理器
            for processor in sampling_metadata.logitsprocs.argmax_invariant:
                logits = processor.apply(logits)

            ...

            # ...执行采样并返回采样结果...
    ``` 

在采样时，采样器检查持久批处理中的所有请求是否都使用贪心采样。如果是，采样器通过跳过"argmax 不变"的 logits 处理器来节省计算。这里，"argmax"是 logits 张量给定行中具有最高 logit 值的 token ID 的简写（即模型对给定请求加权最高的 token）。

* **argmax 不变的 logits 处理器** 是一种不修改 argmax 的 logits 处理器（例如 Min-P）。例如，掩码掉最低概率 token 的 logits 处理器不会改变哪个 token ID 具有最大 logit。贪心采样总是选择最高 logit 值的 token ID，因此从概念上讲，对于贪心采样请求，可以跳过 argmax 不变的 logits 处理器。

* **非 argmax 不变的 logits 处理器** 是一种可能修改 argmax 的 logits 处理器。例如，在特定步数后掩码除 EOS 之外所有 token 以强制解码终止的 logits 处理器，可能最终会掩码最大 logit 值的 token，从而改变 argmax。从概念上讲，对于贪心采样请求，不能跳过这些 logits 处理器。

vLLM 的 logits 处理器抽象要求引擎以批次粒度应用 logits 处理器；因此，在实践中，只有当整个批次使用贪心采样时，才能跳过 argmax 不变的 logits 处理器。

## Logits 处理器编程模型

前面的部分提到了 vLLM logits 处理器必须支持的接口。本节全面介绍了为与 vLLM 引擎兼容而实现 logits 处理器的编程模型，包括 `LogitsProcessor` 基类及其接口方法，以及用于表示持久批处理状态变化的 `BatchUpdate` 数据结构，两者如下所示：

??? code "`LogitsProcessor` 基类和 `BatchUpdate` 数据结构"

    ``` python
    from abc import ABC, abstractmethod
    from collections.abc import Sequence
    from dataclasses import dataclass
    from enum import Enum, auto
    from typing import TYPE_CHECKING

    import torch

    from vllm import SamplingParams

    if TYPE_CHECKING:
        from vllm.config import VllmConfig


    class MoveDirectionality(Enum):
        # 批次内的单向 i1->i2 请求移动
        UNIDIRECTIONAL = auto()
        # 批次内的双向 i1<->i2 请求交换
        SWAP = auto()


    # 添加到批次中的新请求的（索引、参数、prompt token ID、输出 token ID）元组。
    AddedRequest = tuple[int, SamplingParams, list[int], list[int]]

    # 表示批次中请求的单向移动或双向交换的（索引 1、索引 2、方向性）元组
    MovedRequest = tuple[int, int, MoveDirectionality]

    # 任何已移除请求的批次索引。
    RemovedRequest = int


    @dataclass(frozen=True)
    class BatchUpdate:
        """用于 logits 处理器的持久批处理状态更改信息"""
        batch_size: int  # 批次中当前的请求数

        # 添加到、从批次中移除以及在持久批处理中移动的请求的元数据。
        #
        # 关键假设：`output_tok_ids` 列表（它是 `added` 中每个元组的一个元素）
        # 是对请求运行中输出 token 列表的引用；通过此引用，logits 处理器
        # 始终可以看到最新的生成输出 token 列表
        removed: Sequence[RemovedRequest]
        moved: Sequence[MovedRequest]
        added: Sequence[AddedRequest]


    class LogitsProcessor(ABC):

        @abstractmethod
        def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                    is_pin_memory: bool) -> None:
            raise NotImplementedError

        @abstractmethod
        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @abstractmethod
        def is_argmax_invariant(self) -> bool:
            """如果 logits 处理器对贪心采样中的
            argmax 计算没有影响，则为 True。
            注意：对于给定的 LogitsProcessor 子类的不同实例，
            可能有相同或不同的值，取决于子类实现。
            """
            raise NotImplementedError

        @abstractmethod
        def update_state(
            self,
            batch_update: "BatchUpdate" | None,
        ) -> None:
            """当有新的输出 token 时调用，在每次前向传播之前。

            参数：
                batch_update 非 None 当且仅当批次组成发生了变化。
            """
            raise NotImplementedError

        @classmethod
        def validate_params(cls, sampling_params: SamplingParams):
            """为此 logits 处理器验证采样参数。

            对无效参数抛出 ValueError。
            """
            return None

    ```

vLLM logits 处理器必须继承 `LogitsProcessor` 并至少定义以下方法：

* `__init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool)`
    * `vllm_config`：引擎配置数据结构
    * `device`：硬件加速器设备信息
    * `is_pin_memory`：指示是否可以使用固定内存以支持 logits 处理器实现的标志

* `apply(self, logits: torch.Tensor) -> torch.Tensor`：
    * 消费一个 `(num_requests) x (vocab_size)` 的 logits 张量（`logits`）
    * 以批次粒度应用 logits 处理器变换
    * 返回一个变换后的 `(num_requests) x (vocab_size)` 的 logits 张量
    * 您可以就地或非就地修改输入的 logits；就地修改更节省内存

* `is_argmax_invariant(self) -> bool`：
    * 如果 logits 处理器是 argmax 不变的（从不会改变给定请求的最高 logit 值的 token ID），返回 `True`，如果 logits 处理器可能修改 argmax，返回 `False`
    * `is_argmax_invariant()` 在启动时评估一次；如果为 `True`，vLLM 将在所有请求使用贪心采样的步骤中跳过应用此 logits 处理器

* `update_state(self, batch_update: "BatchUpdate" | None) -> None`：
    * 消费一个表示当前引擎步骤开始时持久批处理状态变化的 `BatchUpdate` 数据结构
    * 使用 `BatchUpdate` 成员更新 logits 处理器的内部状态
    * **注意：** 批次更新数据结构可能为 `None`，表示批次组成没有变化。在这种情况下，LogitsProcessor 可能仍希望根据更新的 `output_token_ids` 列表（可能在添加时已保留）来更新其状态。

* `validate_params(cls, sampling_params: SamplingParams)`：
    * 如果 `SamplingParams` 有 logits 处理器使用的无效参数（特别是自定义参数），抛出 `ValueError`。
    * 当请求发送到入口点时，`validate_params()` 将验证 `SamplingParams` 并拒绝带有无效参数的请求。

### `BatchUpdate` 数据结构

`BatchUpdate` 抽象将持久批处理建模为一个请求列表，支持以下更改批次状态的操作（注意，以下操作的顺序反映了它们应在 `update_state()` 中处理的顺序）：

* **移除：** 移除（无替换）索引 `i` 处的请求

    * 在 `Batchupdate.removed` 中由 `int`（表示 `i`）表示

    * 索引移除对批次的影响：

        ``` text
        批次：[A,B,C]
        移除 @ i:  1

        =>

        新批次：[A,x,C] # 丢弃 B 并留下一个空槽
        ```

* **添加：** 在索引 `i` 处添加（或替换现有请求）一个新请求。如果替换了一个请求，其关联状态应被丢弃。

    * 在 `Batchupdate.added` 中表示为一个元组

        ``` text
        (index, new request SamplingParams, prompt token ids, output token ids)
        ```

    * `prompt token ids` 和 `output token ids` 分别是请求的 prompt token ID 列表和输出 token ID 列表的引用。注意，输出 token ID 列表随着每个引擎步骤增长，这种增长对 logits 处理器是可见的，因为输出 token ID 是通过引用传递的。**这对于考虑已生成 token 的 LogitsProcessor 很重要**。

    * 特定 logits 处理器子类的实现决定了添加的请求元组中的字段如何被消化为内部表示。例如，不利用 prompt 或输出 token ID 的 logits 处理器可能只需要使用 `index` 和 `SamplingParams` 并丢弃其他元组字段。

    * 如果索引 `i` 当前持有一个请求，则发生替换：

        ``` text
        批次：[A,B,C]
        要添加的新请求 @ i: D @ 1

        =>

        新批次：[A,D,C] # 添加 D，丢弃 B
        ```

    * 如果索引 `i` 当前不持有请求（因为 `i` 超出当前批次大小范围）：

        ``` text
        批次：[A,B,C]
        要添加的新请求 @ i: D @ 3

        =>

        新批次：[A,B,C,D] # 添加 D，扩展批次
        ```

* **移动：** 将索引 `s` 处的请求移动到索引 `d`，或交换索引 `s` 和 `d` 处的请求

    * 在 `Batchupdate.moved` 中表示为一个元组

        ``` text
        (s, d, UNIDIRECTIONAL 或 SWAP)
        ```

    * 如果 Move 指定为 `UNIDIRECTIONAL`：

        * 索引 `s` 处的请求被移动到索引 `d`；索引 `s` 变为空槽

            ``` text
            批次：[A,x,C,D]
            单向 Move s -> d:  3 -> 1

            =>

            新批次：[A,D,C,x] # 将 D 移动到 1，在 3 处留下空槽
            ```

        * 如果索引 `d` 已有一个请求，它被替换并丢弃

            ``` text
            批次：[A,B,C,D]
            单向 Move s -> d:  3 -> 1

            =>

            新批次：[A,D,C,x] # 将 D 移动到 1，丢弃 B，在 3 处留下空槽
            ```

    * 如果 Move 指定为 `SWAP`，则 `s` 和 `d` 处的请求交换索引

        ``` text
        批次：[A,B,C,D]
        交换 Move s <-> d:  3 <-> 1

        =>

        新批次：[A,D,C,B] # 交换 B 和 D
        ```

此外，`BatchUpdate` 数据结构包含引擎步骤开始时持久批处理大小的表示（`batch_size`）。

### vLLM 引擎如何构建 `BatchUpdate` 数据结构

Logits 处理器的 `update_state()` 实现应假定模型运行器更新持久批处理的以下模型（以 `BatchUpdate` 抽象表示）：

1. 识别在当前引擎步骤中完成的请求的索引

2. 识别在当前步骤中引入的新请求

3. 使用 Add 操作尽可能多地用新请求替换已完成的请求，按被替换请求索引递增的顺序，从最低索引开始

4. 根据新请求和已完成请求的相对数量：

    1. 如果新请求和已完成请求的数量相同，进入下一步

    2. *如果新请求多于已完成请求：* 对未替换已完成请求的剩余新请求应用 Add 操作以扩展批次。为这些新请求分配连续的索引，从 `current_max_batch_index + 1` 开始

    3. *如果新请求少于已完成请求：*

        * 对未替换为新请求的已完成请求应用 Remove 操作。这些被移除的请求索引必然大于上一步中被替换的已完成请求的最大索引。Removes 可能使批次处于非连续状态

        * **"压缩"批次使其连续：** 从最低索引的空槽开始（由 Remove 引起），应用从批次中当前最高非空槽到该空槽的单向 Move。继续按递增的空槽目标索引和递减的非空槽源索引的顺序应用额外的单向 Move 操作，直到批次连续

        * **收缩批次：** 压缩批次的副作用是，由 Remove 操作产生的空槽被分组在批次数组末尾的一个连续块中。因此，压缩后，更新 `BatchUpdate.batch_size` 以反映非空槽的数量

5. 为提升效率重新排序批次。根据注意力后端实现和当前批次特征，可以应用零个或多个 Swap Move 操作来重新排序批次

注意：

* Logits 处理器 `update_state()` 方法必须按以下顺序处理批次更新操作：removes、adds、moves

* Add 操作的索引参数指的是 *Add 发生时* 的索引，即在任何 Move 操作之前
    * 示例：如果一个请求在索引 5 处被 Add，然后与索引 3 交换，`BatchUpdate.added` 中的 Add 操作将与索引 5（而不是 3）关联
    * 换句话说，可以假定 Move 操作是在 Adds 和 Removes 之后应用的

* 可以假定 Move 操作按照它们在 `BatchUpdate.moved` 中出现的顺序应用

* 如果没有新的/完成的请求并且没有批次重新排序，则 logits 处理器的批次更新将为 `None`

#### 示例：新请求少于已完成请求的批次更新

以下示例建模了一个引擎步骤，其中引入了 1 个新请求，消除了 2 个已完成请求，另外注意力后端执行了一次交换以优化批次排序。

``` text
批次状态（引擎步骤开始时）：[A,B,C,D]
批次大小：4

新请求：E

已完成请求：A、C

处理步骤（使用 BatchUpdate 抽象）：

1. 在索引 0 处添加 E

[E,B,C,D] # 丢弃 A
批次大小：4

2. 在索引 2 处移除

[E,B,x,D] # 丢弃 C，索引 2 为空槽
批次大小：4

3. 使用单向 Move 3 -> 2 压缩批次并收缩批次

[E,B,D] x # 空槽现在在批次之外
批次大小：3

4. 注意力后端优化：使用 Swap 0 <-> 1 重新排序批次

[B,E,D]
批次大小：3

```

结果 `BatchUpdate` 数据结构将如下所示：

``` text
BatchUpdate 实例
* added: [(0,E 的 SamplingParams,E 的 prompt token 引用,E 的 output token 引用)]
* removed: [2] # 请求 C 被无替换移除
* moved: [(3,2,UNIDIRECTIONAL),(0,1,SWAP)]
```

#### 示例：新请求多于已完成请求的批次更新

以下示例建模了一个引擎步骤，其中引入了 2 个新请求，消除了 1 个已完成请求，另外注意力后端执行了一次交换以优化批次排序。

``` text
批次状态（引擎步骤开始时）：[A,B,C,D]
批次大小：4

新请求：E,F

已完成请求：C

处理步骤（使用 BatchUpdate 抽象）：

1. 在索引 2 处添加 E

[A,B,E,D] # 丢弃 C
批次大小：4

2. 在索引 4 处添加 F（当前最大批次索引 + 1）

[A,B,E,D,F] # 扩展批次 1
批次大小：5

4. 注意力后端优化：使用 Swap 0 <-> 1 重新排序批次

[B,A,E,D,F]
批次大小：5

```

注意，跳过批次压缩，因为 Remove 操作没有留下空槽。

结果 `BatchUpdate` 数据结构将如下所示：

``` text
BatchUpdate 实例
* added: [(2,E 的 SamplingParams,E 的 prompt token 引用,E 的 output token 引用),(4,F 的 SamplingParams,F 的 prompt token 引用,F 的 output token 引用)]
* removed: [] # 没有请求被无替换移除
* moved: [(0,1,SWAP)]
```

## 如何向 vLLM 引入新的 Logits 处理器

### 编写内置 Logits 处理器的最佳实践

* 鉴于 logits 处理器以批次粒度运行，编写高效的 `apply()` 和 `update_state()` 实现
    * 例如，您可以使用高效的向量化操作来实现 `apply()` 或在 `update_state()` 中更新内部状态向量
    * 然而，如果您认为某个 logits 处理器可能不经常使用，使用请求状态的"稀疏"表示可能是合适的，即该类可以使用一个字典来表示请求配置，该字典仅存储启用该 logits 处理器的请求的元数据

* 由 logits 处理器作者决定：

    1. **配置 logits 处理器对该请求行为的每个请求属性。** 例如，如果您正在为 vLLM 编写一个新的内置 logits 处理器，您可能需要向 `SamplingParams` 和 vLLM REST API 添加额外字段，也可能不需要。

    2. **在每个请求基础上启用或禁用 logits 处理器的条件。** 除非您的意图是让内置 logits 处理器始终作用于所有请求，否则您应该以允许为给定请求禁用 logits 处理器的方式编写您的 logits 处理器，即通过将参数默认为 `None` 或通过传入特定的无操作参数值（如 `0.0`）。尽量节省禁用 logits 处理器的请求的计算和内存。

    3. **在批次级别短路 logits 处理器的条件。** 即使您定义了在请求级别禁用内置 logits 处理器的方法，将其转化为计算节省也可能很困难，例如，如果您的 `update_state()` 和 `apply()` 实现使用在单个命令中操作整个持久批处理的高效向量化实现。例如，仅仅因为一个请求禁用了 logits 处理器，您就不能跳过 `apply()` 中的整个向量化操作。为了在边缘情况下（当没有运行的请求使用内置 logits 处理器时）节省计算，我们建议设计 `apply()` 在禁用所有请求的 logits 处理器时返回未修改的输入张量。同样，考虑如果没有请求启用 logits 处理器，`update_state()` 中的步骤是否可以跳过。

        * 此外，在 `update_state()` 中节省计算的一个简单方法是在 `batch_update` 为 `None` 时提前退出。

* 确保 logits 处理器的 `update_state` 方法丢弃有关已完成请求的信息（即被 Add 替换或遭受 Remove 的请求）。

* `is_argmax_invariant()` 可以硬编码为 `True` 或 `False`，如果 logits 处理器具有一致的行为。然而，argmax 不变性也可以通过编程方式确定（例如，如果您的 logits 处理器在某些方面是用户可定制的，这会影响 logits 处理器是否是 argmax 不变的）。因此，`is_argmax_invariant()` 不是一个类方法。

### 内置 Logits 处理器

内置的 logits 处理器在 vLLM 引擎启动时始终加载。请参见 `vllm/v1/sample/logits_processor/builtin.py` 中现有的 vLLM 内置 logits 处理器，了解如何编写新的内置 vLLM logits 处理器的示例。如果某个 logits 处理器可能对广大用户有用，那么提交 PR 将其作为内置处理器引入是合理的。vLLM 目前使用以下基于上述编程模型的内置 logits 处理器：

* Min-P
* Logit bias
* Min-tokens

请查阅这些 logits 处理器实现，以获取编写内置 logits 处理器的指导。

此外，以下类似 logits 处理器的功能被硬编码到采样器中，尚未使用上述编程模型。它们中的大多数将被重构以使用上述 logits 处理器编程模型。

* Allowed token IDs
* Bad words
* Repetition penalty
* Frequency penalty
* Presence penalty
* Temperature
* Top-K
* Top-P

### 自定义 Logits 处理器

vLLM 可以通过[用户提供的自定义 logits 处理器](../features/custom_logitsprocs.md)进行增强。
