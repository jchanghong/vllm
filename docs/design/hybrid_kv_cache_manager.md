# 混合 KV 缓存管理器

!!! warning
    本文档基于 commit [458e74](https://github.com/vllm-project/vllm/commit/458e74eb907f96069e6d8a4f3c9f457001fef2ea) 编写。此功能仍处于早期阶段，可能会有变化。

## 什么是混合模型？

许多最近的"混合"LLM 在一个模型中组合了多种注意力类型。例如：

1. 滑动窗口注意力（sw）+ 全注意力（full）：gpt-oss、Gemma 2/3、Ministral、cohere 等。
2. Mamba + full：Bamba、Jamba、Minimax 等。
3. 局部分块注意力 + full：Llama4

为了高效地服务这些模型，我们的 [KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager] 必须：

1. 为不同的层类型分配不同的槽位，例如：
    - 全注意力层：为**所有** token 预留槽位。
    - 滑动窗口层：仅为最近的 **`sliding_window_size`** 个 token 预留槽位。
2. 支持特定于层的前缀缓存规则，例如：
    - 全注意力：缓存命中的前缀要求**所有** token 保留在 KV 缓存中。
    - 滑动窗口：缓存命中的前缀仅要求最近的 **`sliding_window_size`** 个 token 保留在 KV 缓存中。

## 定义

1. **kv hidden size**：为单个层存储一个 token 的 KV 缓存所需的字节数。
2. **block**：为 KV 缓存预留的内存被划分为多个具有相同页面大小的*块*（如下所定义）。
3. **block size**：一个块中包含的 token 数量。
4. **page size**：一个块的物理内存大小，定义为：

    $$
    \text{num_layers} \times \text{block_size} \times \text{kv_hidden_size}
    $$

    `num_layers` 不表示模型中层的总数。具体数字取决于本文档中的上下文。

    !!! note
        这与代码中的 `KVCacheSpec.page_size_bytes` 不同，后者定义为：

        $$
        \text{block_size} \times \text{kv_hidden_size}
        $$

## 分配

### 高层思路

我们对所有层类型使用单个内存池。内存池被划分为多个具有相同页面大小的块。[KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager] 根据其注意力类型为不同层分配不同数量的块。

核心挑战是确保每种层类型使用相同的**页面大小**。对于仅使用全注意力的模型，页面大小很简单，定义为：

$$
\text{page_size} = \text{block_size} \times \text{num_hidden_layers} \times \text{kv_hidden_size}
$$

然而，在混合模型中，`num_hidden_layers` 因注意力类型而异，这通常会产生不匹配的页面大小。下面的案例展示了我们如何统一它们。

### 案例 1：玩具模型

让我们从一个玩具示例开始：一个模型有 1 个全注意力层和 3 个滑动窗口注意力层。所有层具有相同的 `kv_hidden_size`。

我们让每个块为一个层保存 `block_size` 个 token，因此：

$$
\text{page_size} = \text{kv_hidden_size} \times \text{block_size}
$$

[KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager] 为每个层分配不同数量的块。

这个案例只是一个玩具示例。对于真实模型，请参考以下案例。

### 案例 2：相同的 `kv_hidden_size` 和规则的模式

当模型有更多层时，例如 20 个滑动窗口注意力层和 10 个具有相同 `kv_hidden_size` 的全注意力层。为每个层调用一次分配器（30 次调用）是可以的，但效率不高。作为解决方案，我们将需要相同数量块的层分组，以减少调用次数。

这种分组是可行的，因为不同类型的层之间通常存在一个良好的比例。例如：

- Gemma-2：1 sw : 1 full
- Llama 4：3 local : 1 full

我们的示例可以视为 2 sw : 1 full。我们可以分配块，就好像模型中有 2 个 sw 和 1 个 full，并将结果重复 10 次，为 30 个层生成 `block_ids`。页面大小变为：

$$
10 \times \text{kv_hidden_size} \times \text{block_size}
$$

假设 `block_size` 为 16，滑动窗口大小为 32，请求长度为 112，那么对于上述示例模型，我们需要分配 11 个块（0-6 给 full，7-8 给 sw 组 1，9-10 给 sw 组 2）。

![分配结果](../assets/design/hybrid_kv_cache_manager/basic_grouping_example.png)

这里，"/" 表示不需要块（滑动窗口层不需要早期 token 的槽位）。

请参见下面的正式定义。这些层被分为多个 *KV 缓存组*，以便满足：

1. **每组内部具有相同的注意力类型**：每个组只包含具有相同注意力类型的层，因此对于给定的请求需要相同数量的块。这使得同一组中的层可以共享相同的块 ID，而不会浪费内存。
2. **各组之间具有相同的页面大小**：因为我们的内存池只有一个页面大小。

我们的示例模型被分为 3 个 KV 缓存组：

- 组 0：10 个全注意力层（full.0 - full.9）
- 组 1：10 个滑动窗口注意力层（sw.0 - sw.9）
- 组 2：10 个滑动窗口注意力层（sw.10 - sw.19）

显然，它满足规则 1。对于规则 2，所有 3 个组都有

$$
10 \times \text{kv_hidden_size} \times \text{block_size}
$$

作为它们的页面大小。

### 案例 3：相同的 `kv_hidden_size` 且无规则模式

不幸的是，并非所有模型都有如此良好的比例，案例 2 中的方法会产生太多的小组。例如，Gemma-3-27b 有 52 个滑动窗口注意力层和 10 个全注意力层。使用案例 2 的约束条件，将有 26 个滑动窗口组和 5 个全注意力组，每个组包含 2 个层。分配仍然效率低下。为了减少 KV 缓存组的数量，我们使用所有注意力类型中最小的层数来分组。例如，Gemma-3-27b 中 min(52, 10)=10 层每组的层数。那么分组结果为：

- 组 0：10 个全注意力层（full.0 - full.9）
- 组 1：10 个滑动窗口注意力层（sw.0 - sw.9）
- 组 2：10 个滑动窗口注意力层（sw.10 - sw.19）
- ...
- 组 6：10 个滑动窗口注意力层（sw.40 - sw.49）
- 组 7：2 个滑动窗口注意力层（sw.50 - sw.51）和 8 个填充层

如果在出现新模型时此启发式方法导致不良结果（例如 20 full + 30 sw，组大小应为 10 而不是 20），我们将更新此算法。

这种情况发生在 Gemma-3 系列模型中，以及案例 2 中带有 eagle 推测解码的模型（引入了一个全注意力层）。该方案有一些内存浪费，并不完美。请报告任何填充开销变得不可接受的情况，以便我们改进算法。

### 案例 4：不同的 `kv_hidden_size`（主要是混合 mamba 模型）

一些架构（例如 Bamba、Jamba、Minimax）将标准注意力层与 Mamba 层交错，其中每个 Mamba 层的每个 token 的状态大小可能远大于注意力层的 `kv_hidden_size`。因为我们只支持所有组之间的单一页面大小，我们必须调和这些不同的隐藏大小。

当前的算法是：

1. 增加注意力层的 `block_size`，直到
    $$
    \text{block_size} \times \text{kv_hidden_size}_{\text{att}} \ge \text{state_size}_{\text{mamba}}
    $$
2. 将每层的 mamba 状态填充到
    $$
    \text{block_size} \times \text{kv_hidden_size}_{\text{att}}
    $$
3. 应用案例 3 中的分组策略。

!!! note
    这可能导致注意力层的 `block_size` 超过 400，这太大了。另一种填充策略是增加 `block_size`，直到

    $$
    \text{block_size} \times \text{kv_hidden_size}_{\text{att}} \times \text{num_attn_layers} \ge \text{state_size}_{\text{mamba}}
    $$

    此填充策略仍在进行中。

### 案例 5：KV 共享

KV 共享指的是一个层使用另一个层的 KV 缓存，例如 gemma-3n。在这些模型中，[KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager] 忽略所有具有 KV 共享的层，仅为需要 KV 缓存的层分配 KV 缓存，并在模型运行器中进行一些修补，将分配结果应用于 KV 共享层。

## 前缀缓存

为简单起见，我们在本节中假设 `block_size=1`。

### 高层思路

块池使用类似于 `tuple(block_hash, group_id) -> block` 的字典来捕获完整的块。这意味着不同组的相同 token 被独立缓存和驱逐。

当新请求到来时，我们检查组的前缀缓存命中，然后返回这些组的交集作为请求的缓存前缀。请参见下文，了解检查一个组的缓存命中及执行交集的详细算法。

### 案例 0：仅全注意力模型

对于全注意力层，为请求中的所有 token 分配块。关于底层设计的详细信息，请参见[前缀缓存](prefix_caching.md)。

要找到请求的最长缓存命中前缀，我们从左（第一个块）到右（最后一个块）枚举，检查块是否被缓存，当缓存未命中时退出。例如，在下面的示例中，我们将返回前 7 个 token（0-6）作为缓存命中前缀（蓝色块已缓存）：

![全注意力的前缀缓存](../assets/design/hybrid_kv_cache_manager/full_attn.png)

### 案例 1：仅滑动窗口注意力模型

对于滑动窗口注意力层，一种朴素的内存分配实现是分配 `sliding_window_size` 个块，并以轮询方式填充这些块。但这种朴素实现与前缀缓存不兼容，因此我们没有选择这种设计。在 vLLM 中，我们为不同的 token 分配不同的块，并释放滑动窗口之外的块。

对于新请求，缓存命中前缀仅要求最后 `sliding_window_size - 1` 个 token 被缓存。假设 `sliding_window_size = 4` 且 `block_size = 1`，请求是一个 15 token 的提示（蓝色块已缓存）：

![滑动窗口注意力的前缀缓存](../assets/design/hybrid_kv_cache_manager/sw_attn.png)

有 3 种可能的缓存命中前缀：

- 缓存命中长度 5，使用 [2, 3, 4] → [5, 6, …, 14] 计算预填充
- 缓存命中长度 6，使用 [3, 4, 5] → [6, 7, …, 14] 计算预填充
- 缓存命中长度 14，使用 [11, 12, 13] → [14] 计算预填充（最高效）

我们可以从右到左检查缓存命中，并在找到匹配时提前退出。这与全注意力相反，全注意力从左到右检查并在匹配失败时提前退出。一个潜在的缺点（与全注意力相比）是，当没有匹配时，我们最终会遍历整个 token 列表，这通常是常见情况。这可能导致不可忽视的开销，但对于 full + swa 来说可以接受，如下所述。

### 案例 2：滑动窗口注意力 + 全注意力模型

第一个问题是如何找到缓存命中前缀。我们需要通过以下方式"求交"全局和滑动窗口注意力层的缓存命中：

1. 获取全注意力的最长缓存命中（从左到右扫描）
2. 获取在该长度内的滑动窗口注意力的最长缓存命中。通过从右到左从全注意力的缓存命中长度开始检查缓存命中来实现。

可以确保滑动窗口注意力层的缓存命中结果也是全注意力层的缓存命中。这比找出每个组的所有可能前缀并求交集更高效，因为如果没有任何缓存命中，我们的方法可以提前退出。

该算法适用于恰好有两种注意力类型的模型：全注意力 + X，其中 X 可以是任意高效的注意力算法，如滑动窗口、llama 4 局部注意力和 mamba。它不支持没有全注意力层的模型，以及具有超过 2 种注意力类型的模型。在撰写本文档时，这对大多数混合模型来说已经足够。

第二个问题是缓存驱逐策略。目前，我们对所有 KV 缓存组使用一个 LRU 队列。当块被释放时（因为请求完成或块超出滑动窗口），它们被添加到 LRU 队列。

### 案例 3：mamba 模型

Mamba 模型的前缀缓存支持正在进行中。一旦实现，可以通过案例 2 中的全注意力 + X 算法支持具有 mamba 层 + 全注意力层的模型。

## 实现

### 概览

![混合 KV 缓存管理器概览](../assets/design/hybrid_kv_cache_manager/overview.png)

`KVCacheManager` 分为 3 层：

- **[KVCacheManager][vllm.v1.core.kv_cache_manager.KVCacheManager]**：调度器和 KV 缓存管理系统之间的接口。
- **[KVCacheCoordinator][vllm.v1.core.kv_cache_coordinator.KVCacheCoordinator]**：协调每个组的 SingleTypeKVCacheManager 以生成请求的分配结果。根据模型的配置，选择以下协调器之一：
    - **[KVCacheCoordinatorNoPrefixCache][vllm.v1.core.kv_cache_coordinator.KVCacheCoordinatorNoPrefixCache]**：当前缀缓存禁用时使用。
    - **[UnitaryKVCacheCoordinator][vllm.v1.core.kv_cache_coordinator.UnitaryKVCacheCoordinator]**：如果仅有一个 KV 缓存组。前缀缓存逻辑被简化，因为不需要求交集。
    - **[HybridKVCacheCoordinator][vllm.v1.core.kv_cache_coordinator.HybridKVCacheCoordinator]**：处理恰好两个 KV 缓存组（必须包括一个全注意力组加上另一个高效注意力组）。其他情况尚未实现。您可以禁用前缀缓存以使用 KVCacheCoordinatorNoPrefixCache。
- **[SingleTypeKVCacheManager][vllm.v1.core.single_type_kv_cache_manager.SingleTypeKVCacheManager]**：每个实例管理一个 KV 缓存组的分配和前缀缓存，实现特定于注意力类型的逻辑（例如全注意力、滑动窗口、Mamba）。

上图中的蓝色框显示了具有 10 个全注意力层和 20 个滑动窗口注意力层的情况，因此：

- 使用 `HybridKVCacheCoordinator`
- 为 3 个 `KVCacheGroup` 使用 1 个 `FullAttentionManager` 和 2 个 `SlidingWindowManager`

### 内存布局

对于一个具有 n 个 `KVCacheGroup`（每个包含 m 个层）的模型，我们分配 m 个缓冲区。每个缓冲区由 n 个层共享（每个组一个）。

下图适用于一个具有 10 个全注意力层（full.0 - full.9）和 20 个滑动窗口注意力层（sw.0-sw.19）的模型。它遵循"分配"部分的"案例 2"，分为 3 个组：

- 组 0：10 个全注意力层（full.0 - full.9）
- 组 1：10 个滑动窗口注意力层（sw.0 - sw.9）
- 组 2：10 个滑动窗口注意力层（sw.10 - sw.19）

对于一个请求，我们分配 11 个块，`block_id` 0-6 给组 0，7-8 给组 1，9-10 给组 2。

以此示例，物理内存被划分为 10 个缓冲区（`KVCacheTensor` 0 - `KVCacheTensor` 9）。每个缓冲区由 3 个层共享（例如，`KVCacheTensor` 0 由组 0 的 full.0、组 1 的 sw.0 和组 2 的 sw.10 共享），并被划分为大小为 `block_size * kv_hidden_size` 的片段。这 3 个注意力层的 KV 缓存根据分配的 `block_ids` 保存到缓冲区的不同片段中：

![示例内存布局](../assets/design/hybrid_kv_cache_manager/memory_layout.png)

!!! note
    一个逻辑"块"映射到物理内存的 10 个缓冲区中的 10 个片段。
