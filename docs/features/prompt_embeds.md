# 提示嵌入输入

本页介绍如何将提示嵌入输入传递给 vLLM。

## 什么是提示嵌入？

大语言模型传统的数据处理流程是从文本到 token ID（通过分词器），然后从 token ID 到提示嵌入。对于传统的仅解码器模型（例如 meta-llama/Llama-3.1-8B-Instruct），将 token ID 转换为提示嵌入这一步是通过从学习的嵌入矩阵中查找完成的，但模型并非仅限于处理与其 token 词汇表对应的嵌入。

## 离线推理

要输入多模态数据，请遵循 [vllm.inputs.EmbedsPrompt][] 中的以下模式：

- `prompt_embeds`：一个 torch 张量，表示一系列提示/token 嵌入。其形状为 (sequence_length, hidden_size)，其中 sequence_length 是 token 嵌入的数量，hidden_size 是模型的隐藏大小（嵌入大小）。

### Hugging Face Transformers 输入

您可以将 Hugging Face Transformers 模型中的提示嵌入传递给提示嵌入字典的 `'prompt_embeds'` 字段，如下例所示：

[examples/features/prompt_embed/prompt_embed_offline.py](../../examples/features/prompt_embed/prompt_embed_offline.py)

## 在线服务

我们的 OpenAI 兼容服务器通过 [Completions API](https://platform.openai.com/docs/api-reference/completions) 和 [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) 接受提示嵌入输入。两者都通过 `vllm serve` 中的 `--enable-prompt-embeds` 标志启用。

### Completions API

提示嵌入输入通过 JSON 请求体中的 `'prompt_embeds'` 键添加。

当单个请求中同时提供 `'prompt_embeds'` 和 `'prompt'` 输入时，提示嵌入总是首先返回。

提示嵌入以 base64 编码的 torch 张量形式传入。

Completions 端点 **不会** 对 `prompt_embeds` 应用聊天模板。如果模型假定使用某种聊天模板，则由调用者负责为完整且已应用模板的提示生成嵌入：先应用聊天模板，然后对生成的 token ID 进行嵌入。模型通常需要的任何内容（系统提示、角色标记、生成提示等）都必须已包含在嵌入的 token 中。

### Chat Completions API

提示嵌入可以作为聊天消息中的内容部分包含在内，与文本交错排列：

```json
{
  "messages": [
    {
      "role": "system",
      "content": [
        {"type": "text", "text": "You are a helpful assistant."},
        {"type": "prompt_embeds", "data": "<base64_encoded_tensor>"}
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "prompt_embeds", "data": "<base64_encoded_tensor>"},
        {"type": "text", "text": "Summarize the above."}
      ]
    }
  ]
}
```

每个 `prompt_embeds` 内容部分包含一个 `data` 字段，其值为一个形状为 `(num_tokens, hidden_size)` 的 base64 编码 `torch.Tensor`。多个 `prompt_embeds` 部分可以出现在任何消息中，并且可以位于相对于文本部分的任何位置。服务器在渲染聊天模板时将每个部分扩展为正确数量的占位符 token，然后在对应位置将预计算的嵌入拼接到模型的输入中。

与 Completions API 不同，`prompt_embeds` 内容部分应仅编码 **内容本身**，而不是已应用模板的对话。服务器会在请求时将聊天模板包裹在嵌入内容周围，就像处理纯文本 `content` 字符串一样。如果在此处嵌入完整的已应用模板对话，将导致模板被重复应用，从而产生错误的模型输入。

!!! warning
    如果传递的嵌入形状不正确，vLLM 引擎可能会崩溃。
    仅对受信任的用户启用此标志！

### 通过 OpenAI 客户端的 Transformers 输入

首先，启动 OpenAI 兼容服务器：

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct --runner generate \
  --max-model-len 4096 --enable-prompt-embeds
```

然后，您可以使用如下 OpenAI 客户端：

[examples/features/prompt_embed/prompt_embed_inference_with_openai_client.py](../../examples/features/prompt_embed/prompt_embed_inference_with_openai_client.py)
