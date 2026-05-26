# 生成式评分

`/generative_scoring` 端点使用 CausalLM 模型（例如 Llama、Qwen、Mistral）来计算指定令牌 ID 作为下一个令牌出现的概率。每个条目（文档）与查询连接起来形成一个提示，模型预测在提示之后每个标签令牌作为下一个令牌的可能性。这使你能够根据查询对条目进行评分——例如，询问"这是法国首都吗？"并根据模型回答"是"的可能性对每个城市进行评分。

当服务器以生成式模型（任务 `"generate"`）启动时，此端点自动可用。它独立于基于池化的[评分 API](../../models/pooling_models/scoring.md#score-api)，后者使用交叉编码器、双编码器或后期交互模型。

**要求：**

- `label_token_ids` 参数是**必需的**，并且必须包含**至少 1 个令牌 ID**。
- 当提供 2 个标签令牌时，分数等于 `P(label_token_ids[0]) / (P(label_token_ids[0]) + P(label_token_ids[1]))`（两个标签上的 softmax）。
- 当提供更多标签时，分数是第一个标签令牌在所有标签令牌上的 softmax 归一化概率。

## 工作原理

1. **提示构建**：对于每个条目，构建 `prompt = query + item`（如果 `item_first=true` 则为 `item + query`）
2. **前向传递**：在每个提示上运行模型以获取下一个令牌的 logits
3. **概率提取**：提取指定 `label_token_ids` 的 logprobs
4. **Softmax 归一化**：仅对标签令牌应用 softmax（当 `apply_softmax=true` 时）
5. **分数**：返回第一个标签令牌的归一化概率

## 查找令牌 ID

要查找你的标签的令牌 ID，请使用分词器：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
no_id = tokenizer.encode("No", add_special_tokens=False)[0]
print(f"Yes: {yes_id}, No: {no_id}")
```

## 示例

```bash
curl -X POST http://localhost:8000/generative_scoring \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "query": "Is this city the capital of France?",
    "items": ["Paris", "London", "Berlin"],
    "label_token_ids": [9454, 2753]
  }'
```

这里，每个条目都被追加到查询中，形成诸如 `"Is this city the capital of France? Paris"`、`"... London"` 等提示。然后模型预测下一个令牌，分数反映"Yes"（令牌 9454）与"No"（令牌 2753）的概率。

??? console "响应"

    ```json
    {
      "id": "generative-scoring-abc123",
      "object": "list",
      "created": 1234567890,
      "model": "Qwen/Qwen3-0.6B",
      "data": [
        {"index": 0, "object": "score", "score": 0.95},
        {"index": 1, "object": "score", "score": 0.12},
        {"index": 2, "object": "score", "score": 0.08}
      ],
      "usage": {"prompt_tokens": 45, "total_tokens": 48, "completion_tokens": 3}
    }
    ```
