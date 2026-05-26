# vLLM IR：函数式中间表示

## 动机

vLLM IR 是一种**函数式中间表示（IR）**，填补了底层 `torch` 算子与 vLLM 层（如 `RMSNorm` 和量化算子）之间的空白。通过将算子**语义**与**实现**和**分派**分离，vLLM IR 同时简化了编译和内核注册与分派。它作为 torch FX 表示中的一种**方言**运行，可以与"常规"的 torch 算子和自定义 torch 算子/内核完全互操作，并支持从之前的 `CustomOp` 方法逐步迁移。

关键设计原则：

- **即时-编译一致性**：在即时和编译模式下行为一致（除微小数值差异外）
- **简单、透明且强大的内核选择**：良好的可见性和控制性，便于调试
- **约定优于配置**：注册算子和实现几乎无需样板代码
- **可扩展性**：算子和实现可以在任何位置注册，无论是树内还是树外
- **互操作性**：与"常规"的 torch 算子和自定义 torch 算子/内核完全兼容，减少开发摩擦并支持逐步迁移

清晰的语义/实现分离支持统一且可扩展的分派机制，允许每个平台有多个内核和强大的内核选择。这种分离还促进了更清晰的测试和基准测试，消除了传统方法所需的大量样板代码。

通过将内核选择延迟到编译过程的后期，编译器可以在更高级别的表示上运行，具有以下主要优势：

- 融合/变换传递中的模式匹配每个算子只需要一个简单的模式
- OOT 编译器后端可以从更高级别的表示进行降级（进行中）
- 编译器可以对可用实现进行自动调优（未来功能）

## 快速概览

### 声明 IR 操作

IR 操作使用 `@register_op` 装饰器声明，并附带定义算子语义的原生 PyTorch 实现：

```python
# vllm/ir/ops/layernorm.py
from torch import Tensor
from vllm.ir import register_op

@register_op
def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None) -> Tensor:
    """加权均方根层归一化"""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(orig_dtype)
    if weight is not None:
        x = x * weight
    return x
```

原生实现有三个目的：

1. **语义定义**：指定操作的精确语义，包括形状和步长
2. **默认实现**：当没有其他（更好的）实现可用时使用
3. **测试参考**：其他实现必须匹配这些语义

### 注册实现

内核实现使用 IR 算子对象上的 `register_impl` 装饰器注册：

```python
# vllm/kernels/vllm_c.py
from vllm import ir

rms_norm_no_var = lambda x, weight, epsilon, variance_size=None: variance_size is None

@ir.ops.rms_norm.register_impl("vllm_c", supports_args=rms_norm_no_var, supported=current_platform.is_cuda_alike())
def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None) -> Tensor:
    output = torch.empty_like(x)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output
```

实现可以指定：

- `supported`：指示该实现是否可用的静态布尔值
- `supports_args`：检查实现是否支持特定参数的函数
- `inplace`：此实现是否为输出重用输入内存

### 在模型中使用 IR 操作

IR 操作在模型代码中直接导入和调用：

```python
# vllm/model_executor/layers/layernorm.py
from vllm import ir

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: Tensor, residual: Tensor | None = None):
        if residual is None:
            return ir.ops.rms_norm(x, self.weight, self.variance_epsilon)

        # 使用 maybe_inplace 重载允许实现为输出重用输入内存
        # （在此调用后使用 x 或 residual 是未定义行为）
        return ir.ops.fused_add_rms_norm.maybe_inplace(
            x, residual, self.weight, self.variance_epsilon
        )
```

### 配置内核选择

内核选择通过配置中的优先级列表控制。优先级列表指定考虑实现的顺序，选择第一个受支持的实施。这包括静态支持检查（`supported=...`）和动态参数支持检查（`supports_args=...`）。

#### 命令行配置

使用 `--ir-op-priority.<op_name>=<provider1>,<provider2>,...`：

```bash
# CUDA：为 rms_norm 使用 vllm_c 实现
vllm serve meta-llama/Llama-3.2-1B \
  --ir-op-priority.rms_norm=vllm_c

# ROCm：首先尝试 aiter，回退到 vllm_c，然后是 native
vllm serve meta-llama/Llama-3.2-1B \
  --ir-op-priority.rms_norm=aiter,vllm_c,native

# 配置多个操作
vllm serve meta-llama/Llama-3.2-1B \
  --ir-op-priority.rms_norm=vllm_c \
  --ir-op-priority.fused_add_rms_norm=vllm_c
```

#### Python 配置

```python
from vllm import LLM
from vllm.config import VllmConfig, KernelConfig

llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    vllm_config=VllmConfig(
        kernel_config=KernelConfig(
            ir_op_priority={
                "rms_norm": ["vllm_c", "native"],
                "fused_add_rms_norm": ["vllm_c", "native"],
            }
        )
    )
)
```

#### 平台默认值

每个平台提供自动应用的默认优先级列表：

```python
# CUDA/XPU/ROCm 平台默认值（使用 Inductor 编译时）
{
  "rms_norm": ["native"],  # 默认使用原生 torch
  "fused_add_rms_norm": ["native"],
}

# CUDA 平台默认值（即时或仅 Dynamo）
{
  "rms_norm": ["vllm_c", "native"],
  "fused_add_rms_norm": ["vllm_c", "native"],
}

# ROCm 平台默认值（未来 - 目前与 CUDA 相同）
{
    "rms_norm": ["aiter", "vllm_c", "native"],
    "fused_add_rms_norm": ["aiter", "vllm_c", "native"],
}

# XPU 平台默认值（即时或仅 Dynamo）
{
    "rms_norm": ["xpu_kernels", "native"],
    "fused_add_rms_norm": ["xpu_kernels", "native"],
}
```

用户指定的优先级会预置到平台默认值之前，因此您只需指定顺序异常的实现，其他实现会自动附加。

## 编译管道

vLLM IR 大量定制了基于 `torch.compile` 的编译过程，允许自定义编译传递在高层次 IR 上操作，同时最终仍产生高效的底层代码。编译管道由多个阶段组成：

### 1. Dynamo 追踪

当 `torch.compile` 追踪模型的前向传播时，vLLM IR 操作在 `vllm_ir` torch 库中显示为自定义操作。这些操作对 Dynamo 是不透明的，意味着它们直接出现在 FX 图中而无需分解：

```python
# Python 代码（epsilon=1e-5）
x1 = ir.ops.rms_norm(x, weight, epsilon)
x2, residual_out = ir.ops.fused_add_rms_norm.maybe_inplace(x1, residual, weight, epsilon)

# Dynamo 追踪后的 FX 图
x1 = torch.ops.vllm_ir.rms_norm.default(x, weight, 1e-5); x = None
out = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(x1, residual, weight, 1e-5); x1 = residual = None
x2 = out[0]
residual_out = out[1]
```

### 2. AOTAutograd 和函数化

AOTAutograd 对图进行函数化，将所有可变操作转换为函数式等价操作。对于具有 `maybe_inplace` 重载的 vLLM IR 操作，我们在 AOTAutograd 之前使用前梯度自定义传递钩子手动处理，将它们转换为函数式 `default` 重载。

```python
# 函数化后
x1 = torch.ops.vllm_ir.rms_norm.default(x, weight, 1e-5); x = None
out = torch.ops.vllm_ir.fused_add_rms_norm.default(x1, residual, weight, 1e-5); x1 = residual = None
x2 = out[0]
residual_out = out[1]
```

该传递还追踪哪些输入被"捐赠"（传递给 `maybe_inplace`），将这些信息存储在 vLLM 的 `PassContext` 中，用于后续的克隆消除。

### 3. IR 融合和变换传递

函数化后，自定义的 vLLM 传递在包含高层 IR 操作的函数式 FX 图上进行操作。这些传递可以执行融合、为序列并行分发操作以及其他变换：

```python
# 示例：序列并行（参见 SequenceParallelismPass）
# SP 传递前

all_reduce = torch.ops.vllm.all_reduce(x, "tp:0")
rms_norm = torch.ops.vllm_ir.rms_norm(all_reduce, weight, 1e-5)

# SP 传递后
reduce_scatter = torch.ops.vllm.reduce_scatter(x, "tp:0")
rms_norm = torch.ops.vllm_ir.rms_norm(all_reduce, weight, 1e-5)
all_gather = torch.ops.vllm.all_gather(x, "tp:0")
```

融合传递受益于高层表示：它们不需要匹配底层 PyTorch 操作、单独处理不同的内核实现，或处理自定义内核的函数化。

### 4. IR 降级

降级传递（`VllmIRLoweringPass`）将每个 vLLM IR 操作替换为其选定的实现。实现的选择基于优先级列表和支持谓词，使用图中元数据中的**伪张量**代替操作参数：

```python
# 实现选择，在即时分派和编译降级中相同
def dispatch(*args) -> IrOpImpl:
  for provider in priority_list:  # 例如 ["vllm_c", "native"]
    impl = ir_op.impls[provider]
    if not impl.supported:
      continue
    if impl.supports_args and not impl.supports_args(*args):
      continue
    return impl

# make_fx 使用 torch.fx.symbolic_trace
impl_graph = make_fx(selected_impl.impl_fn)
# 用 impl_graph 的节点替换 IR 算子节点
match.replace_by_example(selected_impl.impl_fn, node.args)
```

例如，使用 `vllm_c` 实现降级 `rms_norm`：

```python
# 降级前（IR 算子）
rms_norm = torch.ops.vllm_ir.rms_norm.default(x, weight, 1e-5)

# 降级后（追踪的 vllm_c 实现）
# 注意：降级目前不进行函数化，这将来可能会改变。
empty =  torch.ops.aten.empty.memory_format(x.shape, ...)
rms_norm = torch.ops._C.rms_norm(empty, x, weight, 1e-5)
```

当降级一个会修改输入的实现（`inplace=True`）时，降级传递会插入克隆以保持函数式语义：

```python
# fused_add_rms_norm 的 vllm_c 实现会修改其前两个参数
# 为安全起见降级时添加克隆
clone_default = torch.ops.aten.clone.default(x)
clone_default_1 = torch.ops.aten.clone.default(residual)
fused_add_rms_norm = torch.ops._C.fused_add_rms_norm.default(clone_default, clone_default_1, weight, 1e-5)
```

### 5. 克隆清理

降级后，克隆消除传递（`UnsafeCloneEliminationPass`）会移除降级过程中引入的不必要的克隆。此传递对于在使用 `maybe_inplace` 的就地内核时实现零复制行为至关重要。如果以下条件之一成立，该传递会移除克隆：

- 克隆的输入是在图中创建的，且不在图中再次使用
- 克隆的输入是图参数，并被标记为已捐赠

```python
# 清理后（已捐赠的输入，无后续使用）
fused_add_rms_norm = torch.ops._C.fused_add_rms_norm.default(x, residual, weight, 1e-5)
```

就地函数化（追踪已捐赠输入）和克隆清理的结合使编译器能够安全地使用就地内核，而无需添加冗余复制或增加内存使用。

### 6. Inductor 优化和代码生成

IR 降级和清理后，图中仅包含标准 PyTorch 操作和平台特定的自定义算子。然后 Inductor 执行其标准代码生成：

- **Inductor 降级和逐点融合**：融合逐元素操作、规约等。
- **内存规划**：确定缓冲区分配和重用
- **内核生成**：为融合操作生成 Triton 或 C++ 代码
- **自动调优**：选择最佳内核配置

### 管道总结

```text
模型前向传播
    ↓
[Dynamo 追踪] → 包含 vllm_ir.* 算子的 FX 图
    ↓
[前梯度：就地函数化] → maybe_inplace → default，追踪已捐赠输入
    ↓
[AOTAutograd] → 函数化
    ↓
[后梯度：IR 融合传递] → 融合高层 IR 算子（例如 rms_norm + quant）
    ↓
[后梯度：IR 降级] → vllm_ir.* 算子 → 实现算子（如需则带克隆）
    ↓
[后梯度：克隆清理] → 使用已捐赠输入信息移除不必要的克隆
    ↓
[Inductor] → 模式匹配、融合、内存规划、代码生成
    ↓
编译后的代码
```

## 核心 vLLM IR 概念

### 操作声明

操作使用 `@register_op` 装饰器声明，该装饰器创建一个 `IrOp` 对象：

```python
@register_op(
    name=None,           # 操作名称（默认为函数名）
    activations=None,    # 激活参数列表（默认为以 'x' 开头的参数）
    allow_inplace=False, # 是否创建 maybe_inplace 重载
)
def op_name(...):
    ...
```

**参数：**

- `activations`：被视为"激活"的参数名列表（通常由 `maybe_inplace` 消耗）。默认为以 `x` 开头的参数。
- `allow_inplace`：创建 `maybe_inplace` 重载以实现内存高效执行（见下文）。

### `maybe_inplace` 重载

`maybe_inplace` 重载是 LLM 推理中内存效率的关键特性。它表明调用者在操作后不需要保留激活输入，允许就地实现为输出重用输入内存。

#### 语义和用法

```python
# 标准用法：输入被保留
out, res_out = ir.ops.fused_add_rms_norm(x, residual, weight, epsilon)
# x 和 residual 保持不变，out 和 res_out 是新张量

# maybe_inplace：输入可能被修改
out, res_out = ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, epsilon)
# x 和 residual 可能被修改（之后使用它们是未定义行为）
# out 和 res_out 可能与 x 和 residual 共享内存
```

在将激活输入传递给 `maybe_inplace` 后使用它是**未定义行为**：

```python
# 错误：在捐赠 x 后使用它
out, res_out = ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, epsilon)
result = out + x  # 错误：x 已被捐赠！
```

如果您需要保留输入，要么使用默认重载，要么手动克隆：

```python
# 选项 1：使用默认重载
out, res_out = ir.ops.fused_add_rms_norm(x, residual, weight, epsilon)
result = out + x  # 正确：x 被保留

# 选项 2：在 maybe_inplace 前克隆
out, res_out = ir.ops.fused_add_rms_norm.maybe_inplace(x.clone(), residual, weight, epsilon)
result = out + x  # 正确：x 被保留，克隆被捐赠
```

#### 编译行为

在编译期间，就地函数化传递验证已捐赠的输入不再被使用，并将 `maybe_inplace` 转换为函数式的 `default` 重载：

```python
# 就地函数化传递（前梯度）
for node in graph.nodes:
    if node.target == torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace:
        # 检查激活输入在此节点后是否被使用
        for activation_arg in activation_inputs:
            for user in activation_arg.users:
                if user appears after node:
                    raise ValueError(f"输入 {activation_arg} 已被捐赠但再次使用")

        # 转换为默认重载
        node.target = torch.ops.vllm_ir.fused_add_rms_norm.default

        # 追踪已捐赠的图输入，用于后续的克隆消除
        for i, arg in enumerate(node.args):
            if arg.op == "placeholder" and i in activation_indices:
                pass_context.donated_input_ids.add(node_to_idx[arg])
```

然后，克隆消除传递使用已捐赠的输入信息，在降级就地内核时消除不必要的复制。

#### 即时模式行为

在即时模式下（无 `torch.compile`），`maybe_inplace` 实现了**最大内存高效**执行，允许 IR 操作直接分派到就地实现：

```python
# maybe_inplace 的即时分派逻辑
impl: IrOpImpl = ir_op.dispatch(*args)
return impl.impl_fn(*args)

# default 的即时分派逻辑：
impl: IrOpImpl = ir_op.dispatch(*args)
if impl.inplace:
  args = [
    arg.clone() if i in ir_op.activations else arg
    for i, arg in enumerate(args)
  ]
return impl.impl_fn(*args)
```

模型代码中的 `maybe_inplace` 与就地内核实现的结合，在即时和编译模式下均提供了最佳的内存效率，且两种情况下语义相同。

#### 内存节省示例

考虑一个带残差连接的 transformer 层：

```python
# 没有 maybe_inplace（每层 2 次分配）
hidden_states = self.attention(input)
normed, residual = ir.ops.fused_add_rms_norm(hidden_states, input, weight, eps)
# 内存：input（保留）、hidden_states（保留）、normed（新）、residual（新）

# 使用 maybe_inplace（使用就地内核时每层 0 次分配）
hidden_states = self.attention(input)
normed, residual = ir.ops.fused_add_rms_norm.maybe_inplace(hidden_states, input, weight, eps)
# 内存：normed（重用 hidden_states）、residual（重用 input）
```

### 实现注册

实现使用 `register_impl` 方法注册：

```python
@ir.ops.op_name.register_impl(
    provider="provider_name",  # 唯一标识符（例如 "vllm_c"、"aiter"、"triton"）
    supported=True,            # 静态可用性检查
    supports_args=None,        # 动态参数支持检查
)
def impl_fn(...):
    ...
```

**提供者命名约定：**

- `native`：保留用于原生 torch 实现（使用 `@register_op` 声明）
- `vllm_c`：通过 `torch.ops._C` 的 C++/CUDA 内核
- `aiter`：AMD AITER 库
- `xpu_kernels`：在 `vllm-xpu-kernels` 中实现的 SYCL/SYCLTLA 内核
- `triton_*`：Triton 内核
- 其他实现使用平台/库名称

**支持检查：**

- `supported`：静态布尔值，在导入时检查一次（例如 `HAS_TRITON`、`is_cuda_alike()`）
- `supports_args`：函数 `(*args, **kwargs) -> bool`，检查参数兼容性
    - 编译期间使用**伪张量**调用，实现零成本检查
    - 即时模式分派期间使用**真实张量**调用
    - 不应检查批量大小或基于值添加守卫

支持谓词示例：

```python
def aiter_rms_norm_supports(x, weight, epsilon, variance_size=None):
    # 检查 dtype（正确：不依赖于批量大小）
    if x.dtype not in [torch.float16, torch.bfloat16]:
        return False
    # 检查可选参数（正确：静态检查）
    if variance_size is not None:
        return False
    return True

@ir.ops.rms_norm.register_impl("aiter", supports_args=aiter_rms_norm_supports)
def rms_norm(...):
    ...
```

当设置 `VLLM_BATCH_INVARIANT=1` 时，批处理不变内核会自动被选择。

### 即时模式 vs 编译模式

vLLM IR 操作在即时和编译模式下行为相同：

**即时模式：**

- 基于优先级列表直接分派到实现
- 使用真实张量参数检查支持情况
- 最小开销（如果需要，可以进一步优化）

**编译模式：**

- IR 算子作为 `torch.ops.vllm_ir.*` 自定义算子出现在 FX 图中
- 降级使用伪张量选择实现
- 完全集成 Inductor 优化

这种一致性使得：

- 可以自信地在即时模式下进行原型开发
- 通过禁用编译进行调试
- 从即时执行逐步迁移到编译执行

## 其他主题

### 树外实现

外部平台可以在不修改 vLLM 的情况下注册实现：

```python
# 在外部包中
from vllm import ir

@ir.ops.rms_norm.register_impl("my_platform", supported=is_my_platform())
def rms_norm(x, weight, epsilon, variance_size=None):
    return my_platform.rms_norm(x, weight, epsilon)
```

然后配置优先级以使用您的实现：

```python
class MyPlatform(Platform):
  def get_default_ir_op_priority(self):
    return IrOpPriorityConfig(rms_norm=['my_platform', 'native'])

# 用户仍可以相同方式覆盖优先级
llm = LLM(ir_op_priority=IrOpPriorityConfig(rms_norm=['custom_oot_kernel']))
```

### 调试和可观测性

!!! note
    请让我们知道如何为您的用例改进可观测性！

启用调试日志以查看内核选择：

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve ...
```

这将记录：

- 为每个操作选择了哪些实现
- 实现被拒绝的原因（不支持、参数不受支持）
- 编译缓存命中/未命中
- IR 降级统计信息

在编译图中检查选定的实现：

```python
# 编译后，检查降级传递
lowering_pass = backend.lowering_pass
print(lowering_pass.selected_impls)
# 输出：{'rms_norm': {'node_123': 'vllm_c', 'node_456': 'vllm_c'}}
```

## 从 CustomOp 迁移

vLLM IR 设计为与 `CustomOp` 共存并逐步取代它：

1. **算子声明**：将 `CustomOp` 类的 `PluggableLayer` 转换，并将 `forward_native` 移动为 `@register_op` 函数
2. **实现注册**：使用 `@ir.ops.op_name.register_impl` 代替重写方法
3. **层使用**：将 `self.op(...)` 替换为 `ir.ops.op_name(...)`
4. **配置**：将 `--compilation-config.custom-ops` 迁移到 `--ir-op-priority`

迁移可以逐步进行，一次一个操作。

## 另请参阅

- [torch.compile 集成](torch_compile.md) — 通用编译基础设施
- [融合](fusions.md) — vLLM 中的自定义融合和变换传递
- [自定义操作](custom_op.md) — 旧版自定义算子系统
