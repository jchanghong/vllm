# Paged Attention

!!! warning
    这是一份基于 [vLLM 原始论文](https://arxiv.org/abs/2309.06180) 的历史文档。
    它不再描述当前 vLLM 中使用的代码。

目前，vLLM 使用其自己实现的多头查询注意力内核（`csrc/attention/attention_kernels.cu`）。
该内核设计为与 vLLM 的分页 KV 缓存兼容，其中键和值缓存存储在不同的块中（注意，此块概念不同于 GPU 线程块。因此在后续文档中，我将 vLLM 分页注意力块称为"块"，而将 GPU 线程块称为"线程块"）。

为了实现高性能，该内核依赖于专门设计的内存布局和访问方法，特别是当线程从全局内存读取数据到共享内存时。本文档的目的是逐步提供内核实现的高层解释，帮助那些希望了解 vLLM 多头查询注意力内核的人。阅读完本文档后，用户可能会更好地理解，并更容易跟随实际的实现。

请注意，本文档可能不涵盖所有细节，例如如何计算相应数据的正确索引或点乘实现。然而，在阅读本文档并熟悉高层逻辑流程后，您将更容易阅读实际代码并理解细节。

## 输入

内核函数接受一系列参数，供当前线程执行其分配的工作。三个最重要的参数是输入指针 `q`、`k_cache` 和 `v_cache`，它们指向需要被读取和处理的全局内存上的查询、键和值数据。输出指针 `out` 指向全局内存中应写入结果的位置。这四个指针实际上指向多维数组，但每个线程只访问分配给它的那部分数据。此处为了简洁，省略了所有其他运行时参数。

```cpp
template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS, int PARTITION_SIZE = 0>
__device__ void paged_attention_kernel(
    ... // 其他参数。
    const scalar_t* __restrict__ out,       // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
    ... // 其他参数。
)
```

函数签名上方还有一系列在编译期间确定的模板参数。`scalar_t` 表示查询、键和值数据元素的数据类型，例如 FP16。`HEAD_SIZE` 表示每个头中的元素数量。`BLOCK_SIZE` 表示每个块中的 token 数量。`NUM_THREADS` 表示每个线程块中的线程数。`PARTITION_SIZE` 表示张量并行 GPU 的数量（为简单起见，我们假设此为 0，即未启用张量并行）。

有了这些参数，我们需要执行一系列准备工作。这包括计算当前头索引、块索引和其他必要变量。然而，目前我们可以忽略这些准备工作，直接进入实际计算。一旦我们掌握了整个流程，理解它们会更容易。

## 概念

在深入计算流程之前，我想先描述一些后续部分需要的概念。不过，您可以跳过本节，如果在后面遇到任何令人困惑的术语再回来查阅。

- **序列**：序列代表一个客户端请求。例如，`q` 指向的数据形状为 `[num_seqs, num_heads, head_size]`。这意味着 `q` 指向总共 `num_seqs` 个查询序列数据。由于此内核是单查询注意力内核，每个序列只有一个查询 token。因此，`num_seqs` 等于批次中处理的总 token 数。
- **上下文**：上下文由序列已生成的 token 组成。例如，`["What", "is", "your"]` 是上下文 token，输入查询 token 是 `"name"`。模型可能会生成 token `"?"`。
- **Vec**：Vec 是一起获取和计算的元素列表。对于查询和键数据，vec 大小（`VEC_SIZE`）被确定为每个线程组每次可以获取和计算 16 字节的数据。对于值数据，vec 大小（`V_VEC_SIZE`）被确定为每个线程每次可以获取和计算 16 字节的数据。例如，如果 `scalar_t` 是 FP16（2 字节）且 `THREAD_GROUP_SIZE` 为 2，则 `VEC_SIZE` 为 4，而 `V_VEC_SIZE` 为 8。
- **线程组**：线程组是一小组线程（`THREAD_GROUP_SIZE`），每次获取和计算一个查询 token 和一个键 token。每个线程只处理 token 数据的一部分。一个线程组处理的总元素数称为 `x`。例如，如果线程组包含 2 个线程且头大小为 8，则线程 0 处理索引为 0、2、4、6 的查询和键元素，而线程 1 处理索引为 1、3、5、7 的元素。
- **块**：vLLM 中的键和值缓存数据被分割成块。每个块在一个头中存储固定数量（`BLOCK_SIZE`）的 token 数据。每个块可能只包含整个上下文 token 的一部分。例如，如果块大小为 16 且头大小为 128，则对于一个头，一个块可以存储 16 * 128 = 2048 个元素。
- **Warp**：Warp 是一组 32 个线程（`WARP_SIZE`），在流多处理器（SM）上同时执行。在此内核中，每个 warp 每次处理一个查询 token 与一个完整块的键 token 之间的计算（它可能通过多次迭代处理多个块）。例如，如果上下文有 4 个 warp 和 6 个块，分配方式可能是 warp 0 处理第 0、4 块，warp 1 处理第 1、5 块，warp 2 处理第 2 块，warp 3 处理第 3 块。
- **线程块**：线程块是一组可以访问相同共享内存的线程（`NUM_THREADS`）。每个线程块包含多个 warp（`NUM_WARPS`），在此内核中，每个线程块处理一个查询 token 与整个上下文的键 token 之间的计算。
- **网格**：网格是线程块的集合，定义了集合的形状。在此内核中，形状为 `(num_heads, num_seqs, max_num_partitions)`。因此，每个线程块只处理一个头、一个序列和一个分区的计算。

## 查询

本节将介绍查询数据如何在内存中存储以及每个线程如何获取数据。如上所述，每个线程组获取一个查询 token 的数据，而每个线程本身只处理一个查询 token 数据的一部分。在每个 warp 内，每个线程组将获取相同的查询 token 数据，但将其与不同的键 token 数据相乘。

```cpp
const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
```

![query](../assets/design/paged_attention/query.png)

每个线程定义自己的 `q_ptr`，指向全局内存上分配的查询 token 数据。例如，如果 `VEC_SIZE` 为 4 且 `HEAD_SIZE` 为 128，则 `q_ptr` 指向包含总共 128 个元素的数据，分为 128 / 4 = 32 个 vec。

![q_vecs](../assets/design/paged_attention/q_vecs.png)

```cpp
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
```

接下来，我们需要将 `q_ptr` 指向的全局内存数据读取到共享内存作为 `q_vecs`。需要注意的是，每个 vecs 被分配给不同的行。例如，如果 `THREAD_GROUP_SIZE` 为 2，线程 0 将处理第 0 行的 vecs，而线程 1 处理第 1 行的 vecs。通过这种方式读取查询数据，相邻的线程（如线程 0 和线程 1）可以读取相邻的内存，实现内存合并以提高性能。

## 键

与"查询"部分类似，本节介绍键的内存布局和分配。虽然每个线程组在一次内核运行中只处理一个查询 token，但可能在多次迭代中处理多个键 token。同时，每个 warp 将在多次迭代中处理多个键 token 块，确保整个线程组在内核运行后处理完所有上下文 token。在此上下文中，"处理"指的是执行查询数据与键数据之间的点乘。

```cpp
const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                    + kv_head_idx * kv_head_stride
                    + physical_block_offset * x;
```

与 `q_ptr` 不同，每个线程中的 `k_ptr` 在不同的迭代中将指向不同的键 token。如上所示，`k_ptr` 根据 `k_cache` 在分配的块、分配的头和分配的 token 处指向键 token 数据。

![key](../assets/design/paged_attention/key.png)

上图说明了键数据的内存布局。假设 `BLOCK_SIZE` 为 16，`HEAD_SIZE` 为 128，`x` 为 8，`THREAD_GROUP_SIZE` 为 2，总共有 4 个 warp。每个矩形代表一个头中一个键 token 的所有元素，将由一个线程组处理。左半部分显示 warp 0 的总共 16 个键 token 数据块，而右半部分代表其他 warp 或迭代的剩余键 token 数据。在每个矩形内部，总共有 32 个 vec（一个 token 的 128 个元素），将由 2 个线程（一个线程组）分别处理。

![k_vecs](../assets/design/paged_attention/k_vecs.png)

```cpp
K_vec k_vecs[NUM_VECS_PER_THREAD]
```

接下来，我们需要从 `k_ptr` 读取键 token 数据，并将其作为 `k_vecs` 存储在寄存器内存中。我们对 `k_vecs` 使用寄存器内存，因为它只会被一个线程访问一次，而 `q_vecs` 会被多个线程多次访问。每个 `k_vecs` 将包含多个向量用于后续计算。每个 vec 将在每个内部迭代中设置。vec 的分配允许 warp 中的相邻线程一起读取相邻内存，这再次促进了内存合并。例如，线程 0 将读取 vec 0，而线程 1 将读取 vec 1。在下一次内部循环中，线程 0 将读取 vec 2，而线程 1 将读取 vec 3，依此类推。

您可能仍然对整个流程有些困惑。不用担心，请继续阅读下一节"QK"。它将更清晰、更高层次地说明查询和键的计算流程。

## QK

如下面的伪代码所示，在整个 for 循环块之前，我们获取一个 token 的查询数据并将其存储在 `q_vecs` 中。然后，在外层 for 循环中，我们遍历指向不同 token 的不同 `k_ptr`，并在内层 for 循环中准备 `k_vecs`。最后，我们执行 `q_vecs` 与每个 `k_vecs` 之间的点乘。

```cpp
q_vecs = ...
for ... {
    k_ptr = ...
    for ... {
        k_vecs[i] = ...
    }
    ...
    float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
}
```

如前所述，每个线程每次只获取部分查询和键 token 数据。然而，在 `Qk_dot<>::dot` 中会发生跨线程组的规约。因此，此处返回的 `qk` 不仅仅是部分查询和键 token 点乘的结果，而实际上是整个查询和键 token 数据之间的完整结果。

例如，如果 `HEAD_SIZE` 的值为 128，`THREAD_GROUP_SIZE` 为 2，则每个线程的 `k_vecs` 将包含总共 64 个元素。然而，返回的 `qk` 实际上是 128 个查询元素与 128 个键元素点乘的结果。如果您想了解更多关于点乘和规约的细节，可以参考 `Qk_dot<>::dot` 的实现。但为简洁起见，本文档不涉及此内容。

## Softmax

接下来，我们需要计算所有 `qk` 的归一化 softmax，如上所示，其中每个 $x$ 代表一个 `qk`。为此，我们必须获得所有 `qk` 的 `qk_max`（$m(x)$）和 `exp_sum`（$\ell(x)$）的规约值。规约应跨整个线程块执行，包括查询 token 与所有上下文键 token 之间的结果。

$$
\begin{gather*}
m(x):=\max _i \quad x_i \\ \quad f(x):=\left[\begin{array}{lll}e^{x_1-m(x)} & \ldots & e^{x_B-m(x)}\end{array}\right]\\ \quad \ell(x):=\sum_i f(x)_i \\
\quad \operatorname{softmax}(x):=\frac{f(x)}{\ell(x)}
\end{gather*}
$$

### `qk_max` 和 `logits`

就在我们获得 `qk` 结果之后，我们可以使用 `qk` 设置临时的 `logits` 结果（最终 `logits` 应存储归一化的 softmax 结果）。同时，我们可以比较并收集当前线程组计算的所有 `qk` 的 `qk_max`。

```cpp
if (thread_group_offset == 0) {
    const bool mask = token_idx >= context_len;
    logits[token_idx - start_token_idx] = mask ? 0.f : qk;
    qk_max = mask ? qk_max : fmaxf(qk_max, qk);
}
```

请注意，此处的 `logits` 位于共享内存上，因此每个线程组将为自己分配的上下文 token 设置字段。总的来说，logits 的大小应为上下文 token 的数量。

```cpp
for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}

if (lane == 0) {
    red_smem[warp_idx] = qk_max;
}
```

然后我们需要获取跨每个 warp 的规约 `qk_max`。主要思想是让 warp 中的线程相互通信，并获得最终的 `qk` 最大值。

```cpp
for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}
qk_max = VLLM_SHFL_SYNC(qk_max, 0);
```

最后，通过比较此线程块中所有 warp 的 `qk_max`，我们可以获得整个线程块的规约 `qk_max`。然后我们需要将最终结果广播到每个线程。

### `exp_sum`

与 `qk_max` 类似，我们也需要从整个线程块获取规约的求和值。

```cpp
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
}
...
exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);
```

首先，累加每个线程组的所有 exp 值，同时将 `logits` 的每个条目从 `qk` 转换为 `exp(qk - qk_max)`。请注意，此处的 `qk_max` 已经是整个线程块的最大 `qk`。然后，我们可以像对 `qk_max` 一样，对整个线程块进行 `exp_sum` 的规约。

```cpp
const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
}
```

最后，通过规约后的 `qk_max` 和 `exp_sum`，我们可以获得最终归一化的 softmax 结果作为 `logits`。此 `logits` 变量将用于后续步骤中与值数据的点乘。现在，它应存储了所有分配的上下文 token 的 `qk` 的归一化 softmax 结果。

## 值

![value](../assets/design/paged_attention/value.png)

![logits_vec](../assets/design/paged_attention/logits_vec.png)

![v_vec](../assets/design/paged_attention/v_vec.png)

现在我们需要检索值数据并执行与 `logits` 的点乘。与查询和键不同，值数据没有线程组的概念。如图所示，与键 token 内存布局不同，同一列的元素对应同一个值 token。对于一个值数据块，有 `HEAD_SIZE` 行和 `BLOCK_SIZE` 列，被分割成多个 `v_vec`。

每个线程每次总是从相同的 `V_VEC_SIZE` 个 token 中获取 `V_VEC_SIZE` 个元素。因此，单个线程通过多次内部迭代从不同行和相同列检索多个 `v_vec`。对于每个 `v_vec`，它需要与对应的 `logits_vec`（也是来自 `logits` 的 `V_VEC_SIZE` 个元素）进行点乘。总的来说，通过多次内部迭代，每个 warp 将处理一个值 token 块。通过多次外部迭代，处理完整个上下文的值 token。

```cpp
float accs[NUM_ROWS_PER_THREAD];
for ... { // 遍历不同块。
    logits_vec = ...
    for ... { // 遍历不同行。
        v_vec = ...
        ...
        accs[i] += dot(logits_vec, v_vec);
    }
}
```

如上伪代码所示，在外层循环中，与 `k_ptr` 类似，`logits_vec` 遍历不同的块并从 `logits` 读取 `V_VEC_SIZE` 个元素。在内层循环中，每个线程从相同的 token 中读取 `V_VEC_SIZE` 个元素作为 `v_vec` 并执行点乘。需要注意的是，在每个内部迭代中，线程为相同的 token 获取不同头位置的元素。点乘结果累加在 `accs` 中。因此，`accs` 的每个条目映射到分配给当前线程的一个头位置。

例如，如果 `BLOCK_SIZE` 为 16 且 `V_VEC_SIZE` 为 8，每个线程每次从 8 个 token 中获取 8 个值元素。每个元素来自相同头位置的不同 token。如果 `HEAD_SIZE` 为 128 且 `WARP_SIZE` 为 32，则对于每个内部循环，一个 warp 需要获取 `WARP_SIZE * V_VEC_SIZE = 256` 个元素。这意味着一个 warp 需要总共 128 * 16 / 256 = 8 次内部迭代来处理整个值 token 块。每个线程中的每个 `accs` 包含从 8 个不同头位置累加的 8 个元素。对于线程 0，`accs` 变量将有 8 个元素，分别是值头的第 0、32...224 个元素，从所有分配的 8 个 token 中累加得到。

## LV

现在，我们需要对每个 warp 内的 `accs` 执行规约。此过程允许每个线程累加一个块中所有 token 在分配的头位置上的 `accs`。

```cpp
for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
        acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
}
```

接下来，我们对所有 warp 的 `accs` 执行规约，使得每个线程拥有所有上下文 token 在分配的头位置上的 `accs` 累加值。请注意，每个线程中的每个 `accs` 只存储整个头的一部分元素在所有上下文 token 上的累加值。然而，总的来说，输出的所有结果都已计算出来，只是存储在不同的线程寄存器内存中。

??? code

    ```cpp
    float* out_smem = reinterpret_cast<float*>(shared_mem);
    for (int i = NUM_WARPS; i > 1; i /= 2) {
        // 上部的 warp 写入共享内存。
        ...
        float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
            ...
            dst[row_idx] = accs[i];
        }

        // 下部的 warp 更新输出。
        const float* src = &out_smem[warp_idx * HEAD_SIZE];
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
            ...
            accs[i] += src[row_idx];
        }

        // 写入 accs。
    }
    ```

## 输出

现在，我们可以将所有计算结果从本地寄存器内存写入最终的输出全局内存。

```cpp
scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                + head_idx * max_num_partitions * HEAD_SIZE
                + partition_idx * HEAD_SIZE;
```

首先，我们需要定义 `out_ptr` 变量，它指向分配的序列和分配的头起始地址。

```cpp
for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
    if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
    }
}
```

最后，我们需要遍历不同分配的头位置，并根据 `out_ptr` 写出相应的累加结果。

## 引用

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```
