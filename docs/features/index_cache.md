# IndexCache

IndexCache 通过跨层缓存和复用 top-k 索引，减少了 DeepSeek-V3.2 (DSA) 模型中的冗余 top-k 计算。

## 背景

DeepSeek-V3.2 使用深度稀疏注意力（DSA）机制，其中每层都会计算 top-k token 的选择。对于具有许多层的深层模型，此计算可能非常昂贵。IndexCache 允许通过复用来自先前层的索引来跳过冗余的 top-k 计算。

参见：[IndexCache 论文](https://arxiv.org/abs/2603.12201)

## 使用方法

### CLI

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' ...
```

### 配置参考

| 参数 | 类型 | 默认值 | 描述 |
|----------------------|------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `use_index_cache` | bool | false | 启用 IndexCache。必须设置为 true 才能使用此功能 |
| `index_topk_freq` | int | 1 | 计算 top-k 的频率（以层为单位）。1 = 每层都计算（禁用），4 = 每 4 层计算 1 次 |
| `index_topk_pattern` | str | null | 逐层 F/S 模式。如果设置，将覆盖 index_topk_freq。每个字符映射到一个 DSA 层：F = 完整计算，S = 共享复用 |

### 配置示例

**使用 `index_topk_freq`**（每 N 层计算一次）：

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' ...
```

**使用 `index_topk_pattern`**（显式逐层控制）：

```bash
# 61 层的自定义模式：F = 计算，S = 复用
vllm serve deepseek-ai/DeepSeek-V3.2 \
    --hf-overrides '{"use_index_cache": true, "index_topk_pattern": "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSF"}'
```

## 工作原理

1. 启用 IndexCache 后，标记为 `"F"`（完整计算）的层会计算并存储 top-k 索引
2. 后续标记为 `"S"`（共享复用）的层从上一层接收缓存的索引，而不是重新计算
3. 缓存的索引通过层堆栈传递，从而减少总计算量

## 要求

- DeepSeek-V3.2 或兼容的 DSA 模型
- 通过 `--hf-overrides` 设置 `use_index_cache: true`
