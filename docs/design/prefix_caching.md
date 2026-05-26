# 自动前缀缓存

KV 缓存块的前缀缓存是 LLM 推理中一种流行的优化技术，用于避免冗余的 prompt 计算。核心思想很简单——我们缓存已处理请求的 KV 缓存块，并在新请求到来时，如果其前缀与之前的请求相同，则重用这些块。由于前缀缓存几乎是一种免费的优化，且不会改变模型输出，它已被许多公共端点（例如 OpenAI、Anthropic 等）和大多数开源 LLM 推理框架（例如 SGLang）广泛使用。

虽然实现前缀缓存的方法有很多，但 vLLM 选择了一种基于哈希的方法。具体来说，我们根据块中的 token 和块之前的 token 前缀对每个 KV 缓存块进行哈希：

```text
                    Block 1                  Block 2                  Block 3
         [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

在上面的示例中，第一个块中的 KV 缓存可以由 token "A gentle breeze stirred"唯一标识。第三个块可以由块中的 token "laughed in the distance"以及前缀 token "A gentle breeze stirred the leaves as children"唯一标识。因此，我们可以构建 `hash(tuple[components])` 的块哈希，其中组件包括：

* 父哈希值：父哈希块的哈希值。
* 块 token：此块中的 token 元组。包含确切 token 的原因是为了减少潜在的哈希值冲突。
* 额外哈希：使此块唯一所需的其他值，例如 LoRA ID、多模态输入哈希（请参见下面的示例）以及在多租户环境中隔离缓存的缓存盐值。

!!! note "注意 1"
    我们只缓存完整的块。

!!! note "注意 2"
    在之前的版本中，哈希键不能保证无冲突。从 v0.11 开始，默认哈希算法为 `sha256`，解决了冲突风险。

    对于 `vllm serve`，您可以通过 `--prefix-caching-hash-algo` 控制哈希算法：
    - `sha256`（默认）：使用 Python 的 `pickle` 进行序列化。哈希在不同 Python 或 vLLM 版本之间可能不可重现。
    - `sha256_cbor`：使用 `cbor2` 进行序列化，提供可重现、跨语言兼容的哈希。建议用于跨环境确定性缓存。
    - `xxhash`：使用 Pickle 序列化与 xxHash（128 位）进行更快、非加密哈希。需要可选的 `xxhash` 包。重要提示：使用不被视为加密安全的哈希算法理论上会增加哈希冲突的风险，这可能导致未定义的行为，甚至在多租户环境中泄露私有信息。即使冲突仍然非常不可能，在开启之前，考虑您对安全风险的容忍度与性能收益非常重要。
    - `xxhash_cbor` 结合了规范的 CBOR 序列化和 xxHash，用于可重现的哈希。需要可选的 `xxhash` 包。

**一个多模态输入哈希的示例**
在此示例中，我们说明了前缀缓存如何与多模态输入（例如图像）一起工作。假设我们有一个包含以下消息的请求：

```text
messages = [
    {"role": "user",
     "content": [
         {"type": "text",
          "text": "What's in this image?"
         },
         {"type": "image_url",
          "image_url": {"url": image_url},
         },
    ]},
]
```

它将变成以下 prompt：

```text
Prompt:
    <s>[INST]What's in this image?\n[IMG][/INST]

Tokenized prompt:
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, 10, 4]

Prompt with placeholders (<P>):
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <P>, <P>, ..., <P>, 4]
```

正如我们所见，在分词之后，`[IMG]` 将被替换为一系列占位符 token，这些占位符将在预填充期间被图像嵌入替换。前缀缓存支持这种情况的挑战在于，我们需要区分图像和占位符。为了解决这个问题，我们对前端图像处理器生成的图像哈希进行编码。例如，上述 prompt 中块的哈希将是（假设块大小为 16，并且有 41 个占位符 token）：

```text
Block 0
    Parent hash: None
    Token IDs: 1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <p>, ..., <p>
    Extra hash: <image hash>
Block 1
    Parent hash: Block 0 hash
    Token IDs: <p>, ..., <p>
    Extra hash: <image hash>
Block 2
    Parent hash: Block 1 hash
    Token IDs: <p>, ..., <p>
    Extra hash: <image hash>
Block 3
    Parent hash: Block 2 hash
    Token IDs: <p>, ..., <p>, 4
    Extra hash: <image hash>
```

在本文档的其余部分，我们首先介绍 vLLM v1 中用于前缀缓存的数据结构，然后是主要 KV 缓存操作符（如分配、追加、释放、驱逐）的前缀缓存工作流程。最后，我们使用一个示例来说明端到端的前缀缓存工作流程。

**缓存隔离用于安全**
为了提高共享环境中的隐私性，vLLM 支持通过可选的每个请求盐值来隔离前缀缓存重用。通过在请求中包含 `cache_salt`，此值被注入到第一个块的哈希中，确保只有具有相同盐值的请求才能重用缓存的 KV 块。这可以防止时序攻击，即攻击者可以通过观察延迟差异来推断缓存内容。这在提供保护的同时不影响性能。

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Here is a document with details about the world series: ..."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ],
  "cache_salt": "your-cache-salt"
}
```

通过这种设置，缓存共享仅限于显式同意通用盐值的用户或请求，从而在信任组内启用缓存重用，同时隔离其他组。

## 数据结构

vLLM v1 中的前缀缓存在 KV 缓存管理器中实现。基本构建块是"Block"数据类（简化）：

```python
class KVCacheBlock:
    # 块 ID（不可变）
    block_id: int
    # 块哈希（当块满时分配，
    # 当块被驱逐时重置）。
    block_hash: BlockHash
    # 当前使用此块的请求数。
    ref_cnt: int

    # 用于空闲队列的双向链表指针。
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```

有两个设计要点需要强调：

1. 在初始化 KV 缓存管理器时，我们分配所有 KVCacheBlock 作为一个块池。这避免了 Python 对象创建的开销，并且可以轻松地随时跟踪所有块。
2. 我们直接在 KVCacheBlock 中引入双向链表指针，以便我们可以直接构造空闲队列。这给我们带来两个好处：
    1. 我们可以以 O(1) 复杂度将中间元素移动到尾部。
    2. 我们可以避免引入另一个 Python 队列（例如 `deque`），后者对元素有包装器。

因此，当 KV 缓存管理器初始化时，我们将拥有以下组件：

![组件概览](../assets/design/prefix_caching/overview.png)

* 块池：KVCacheBlock 的列表。
* 空闲块队列：仅存储头部和尾部块的指针，用于操作。
* 缓存块：从哈希键到块 ID 的映射。
* 请求块：从请求 ID 到分配的块 ID 的映射。

## 操作

### 块分配

**新请求：** 调度器使用 KV 缓存块分配来调度新请求的工作流程：

1. 调度器调用 `kv_cache_manager.get_computed_blocks()` 以获取已计算完成的块序列。这是通过哈希请求中的 prompt token 并查找缓存块来完成的。
2. 调度器调用 `kv_cache_manager.allocate_slots()`。它执行以下步骤：
    1. 计算所需的新块数量，如果没有足够的块可分配则返回。
    2. "触及"已计算的块。它增加计算块的引用计数，如果该块未被其他请求使用，则从空闲队列中移除它。这是为了避免这些计算块被驱逐。请参见下一节中的示例进行说明。
    3. 通过从空闲队列头部弹出块来分配新块。如果头部块是缓存块，这也会"驱逐"该块，以便其他请求从此无法再重用它。
    4. 如果分配的块已满 token，我们立即将其添加到缓存块中，以便同一批次中的其他请求可以重用它。

**运行中的请求：** 调度器使用 KV 缓存块分配来调度运行中的请求的工作流程：

1. 调度器调用 `kv_cache_manager.allocate_slots()`。它执行以下步骤：
    1. 计算所需的新块数量，如果没有足够的块可分配则返回。
    2. 通过从空闲队列头部弹出块来分配新块。如果头部块是缓存块，这也会"驱逐"该块，以便其他请求从此无法再重用它。
    3. 将 token ID 附加到现有块以及新块的槽位中。如果块已满，我们将其添加到缓存块中进行缓存。

**重复块**
假设块大小为 4，您发送一个 prompt 为 ABCDEF 且解码长度为 3 的请求（请求 1）：

```text
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
  Tokens: [A, B, C, D, E, F, G]
  Block Table: [0 (ABCD), 1 (EFG)]
  Cache Blocks: 0
Time 1:
  Tokens: [A, B, C, D, E, F, G, H]
  Block Table: [0 (ABCD), 1 (EFGH)]
  Cache Blocks: 0, 1
Time 2:
  Tokens: [A, B, C, D, E, F, G, H, I]
  Block Table: [0 (ABCD), 1 (EFGH), 2 (I)]
  Cache Blocks: 0, 1
```

现在块 0 和块 1 被缓存，我们再次发送相同的请求（请求 2）并使用贪心采样，以便它将产生与请求 1 完全相同的输出：

```text
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
  Tokens: [A, B, C, D, E, F, G]
  Block Table: [0 (ABCD), 3 (EFG)]
  Cache Blocks: 0, 1
Time 1:
  Tokens: [A, B, C, D, E, F, G, H]
  Block Table: [0 (ABCD), 3 (EFGH)]
  Cache Blocks: 0, 1, 3
```

可以看出，块 3 是一个新的完整块并被缓存。然而，它与块 1 是冗余的，意味着我们两次缓存了相同的块。在 v0 中，当检测到块 3 是重复的时，我们释放块 3 并让请求 2 改用块 1，因此在时间 1 时其块表变为 `[0, 1]`。然而，vLLM v1 中的块表是仅追加的，意味着将块表从 `[0, 3]` 更改为 `[0, 1]` 是不允许的。因此，哈希键 E-H 将有重复的块。这种重复将在请求释放时被消除。

### 释放

当请求完成时，如果没有其他请求正在使用这些块（引用计数 = 0），我们释放其所有块。在此示例中，我们释放请求 1 以及与其关联的块 2、3、4、8。我们可以看到，释放的块以*相反*的顺序添加到空闲队列的尾部。这是因为请求的最后一个块必须哈希更多的 token，因此不太可能被其他请求重用。因此，它应该首先被驱逐。

![请求释放后的空闲队列](../assets/design/prefix_caching/free.png)

### 驱逐（LRU）

当空闲队列的头部块（最近最少使用的块）被缓存时，我们必须驱逐该块以防止它被其他请求使用。具体来说，驱逐涉及以下步骤：

1. 从空闲队列头部弹出块。这是要被驱逐的 LRU 块。
2. 从缓存块中移除块 ID。
3. 移除块哈希。

## 示例

在此示例中，我们假设块大小为 4（每个块可以缓存 4 个 token），并且 KV 缓存管理器中总共有 10 个块。

**时间 1：缓存为空，新请求到来。** 我们分配 4 个块。其中 3 个已满并缓存。第四个块部分满，有 3/4 个 token。

![示例时间 1](../assets/design/prefix_caching/example-time-1.png)

**时间 2：请求 0 使块 3 变满，并请求一个新块以继续解码。** 我们缓存块 3 并分配块 4。

![示例时间 2](../assets/design/prefix_caching/example-time-3.png)

**时间 3：请求 1 到来，带有 14 个 prompt token，其中前 10 个 token 与请求 0 相同。** 我们可以看到只有前 2 个块（8 个 token）命中了缓存，因为第 3 个块只匹配了 4 个 token 中的 2 个。

![示例时间 3](../assets/design/prefix_caching/example-time-4.png)

**时间 4：请求 0 完成并释放。** 块 2、3 和 4 以相反顺序添加到空闲队列中（但块 2 和 3 仍然被缓存）。块 0 和 1 没有被添加到空闲队列，因为它们正被请求 1 使用。

![示例时间 4](../assets/design/prefix_caching/example-time-5.png)

**时间 5：请求 1 完成并释放。**

![示例时间 5](../assets/design/prefix_caching/example-time-6.png)

**时间 6：请求 2 到来，带有 29 个 prompt token，其中前 12 个 token 与请求 0 相同。** 注意，即使空闲队列中的块顺序是 `7 - 8 - 9 - 4 - 3 - 2 - 6 - 5 - 1 - 0`，缓存命中块（即 0、1、2）在分配前被触摸并从队列中移除，因此空闲队列变为 `7 - 8 - 9 - 4 - 3 - 6 - 5`。因此，分配的块是 0（已缓存）、1（已缓存）、2（已缓存）、7、8、9、4、3（已驱逐）。

![示例时间 6](../assets/design/prefix_caching/example-time-7.png)
