# 使用分块处理的长文本嵌入

此目录包含使用 vLLM 的**分块处理**功能来处理超出模型最大上下文长度的长文本嵌入的示例。

## 🚀 快速开始

### 启动服务器

使用提供的脚本启动启用了分块处理的 vLLM 服务器：

```bash
# 基本用法（支持长达约 300 万令牌的超长文本）
./service.sh

# 使用不同模型的自定义配置
MODEL_NAME="jinaai/jina-embeddings-v3" \
MAX_EMBED_LEN=1048576 \
./service.sh

# 用于超长文档
MODEL_NAME="intfloat/multilingual-e5-large" \
MAX_EMBED_LEN=3072000 \
./service.sh
```

### 测试长文本嵌入

运行全面的测试客户端：

```bash
python client.py
```

## 📁 文件

| 文件 | 描述 |
| ---- | ----------- |
| `service.sh` | 启用分块处理的服务器启动脚本 |
| `client.py` | 用于长文本嵌入的全面测试客户端 |

## ⚙️ 配置

### 服务器配置

分块处理的关键参数在 `--pooler-config` 中：

```json
{
  "pooling_type": "auto",
  "use_activation": true,
  "enable_chunked_processing": true,
  "max_embed_len": 3072000
}
```

!!! note
    `pooling_type` 设置模型自身的池化策略，用于处理每个块内的数据。当输入超过模型的原生最大长度时，跨块聚合自动使用 MEAN 策略。

#### 分块处理行为

当输入超过模型的原生最大长度时，分块处理使用 **MEAN 聚合** 进行跨块组合：

| 组件 | 行为 | 描述 |
| --------- | -------- | ----------- |
| **块内** | 模型原生池化 | 使用模型配置的池化策略 |
| **跨块聚合** | 始终使用 MEAN | 基于块令牌数的加权平均 |
| **性能** | 最优 | 处理所有块以实现完整的语义覆盖 |

### 环境变量

| 变量 | 默认值 | 描述 |
| -------- | ------- | ----------- |
| `MODEL_NAME` | `intfloat/multilingual-e5-large` | 要使用的嵌入模型（支持多种模型） |
| `PORT` | `31090` | 服务器端口 |
| `GPU_COUNT` | `1` | 使用的 GPU 数量 |
| `MAX_EMBED_LEN` | `3072000` | 最大嵌入输入长度（支持超长文档） |
| `POOLING_TYPE` | `auto` | 模型原生池化类型：`auto`、`MEAN`、`CLS`、`LAST`（仅影响块内池化，不影响跨块聚合） |
| `API_KEY` | `EMPTY` | 用于身份验证的 API 密钥 |

## 🔧 工作原理

1. **增强的输入验证**：`max_embed_len` 允许接受超过 `max_model_len` 的输入，无需环境变量
2. **智能分块**：根据 `max_position_embeddings` 对文本进行拆分，以保持语义完整性
3. **统一处理**：所有块使用其配置的池化策略通过模型单独处理
4. **MEAN 聚合**：当输入超过模型原生长度时，使用基于令牌数的加权平均合并所有块的结果
5. **一致的输出**：最终嵌入保持与标准处理相同的维度

### 输入长度处理

- **在 max_embed_len 范围内**：接受并处理输入（最多 300 万+ 令牌）
- **超过 max_position_embeddings**：自动触发分块处理
- **超过 max_embed_len**：拒绝输入并显示清晰的错误消息
- **无需环境变量**：无需 `VLLM_ALLOW_LONG_MAX_MODEL_LEN` 即可工作

### 超长文本支持

使用 `MAX_EMBED_LEN=3072000`，您可以处理：

- **学术论文**：带有参考文献的完整研究论文
- **法律文档**：完整的合同和法律文本  
- **书籍**：完整的章节或小型书籍
- **代码仓库**：大型代码库和文档

## 📊 性能特征

### 分块处理性能

| 方面 | 行为 | 性能 |
| ------ | -------- | ----------- |
| **块处理** | 所有块使用原生池化处理 | 与输入长度一致 |
| **跨块聚合** | MEAN 加权平均 | 最小开销 |
| **内存使用** | 与块数成正比 | 中等，可扩展 |
| **语义质量** | 完整的文本覆盖 | 对长文档最优 |

## 🧪 测试用例

测试客户端演示：

- ✅ **短文本**：正常处理（基线）
- ✅ **中等文本**：单块处理
- ✅ **长文本**：多块处理与聚合
- ✅ **超长文本**：多块处理
- ✅ **极长文本**：文档级处理（10 万+ 令牌）
- ✅ **批处理**：单次请求中的混合长度输入
- ✅ **一致性**：跨运行的可重现结果

## 🐛 故障排除

### 常见问题

1. **未启用分块处理**：

   ```log
   ValueError: This model's maximum position embeddings length is 4096 tokens...
   ```

   **解决方案**：确保 pooler 配置中 `enable_chunked_processing: true`

2. **输入超过 max_embed_len**：

   ```log
   ValueError: This model's maximum embedding input length is 3072000 tokens...
   ```

   **解决方案**：增加 pooler 配置中的 `max_embed_len` 或减少输入长度

3. **内存错误**：
  
   ```log
   RuntimeError: CUDA out of memory
   ```
  
   **解决方案**：通过调整模型的 `max_position_embeddings` 来减小块大小，或使用更少的 GPU

4. **处理缓慢**：
   **预期情况**：长文本因多次推理调用而需要更多时间

### 调试信息

服务器日志显示分块处理活动：

```log
INFO: Input length 150000 exceeds max_position_embeddings 4096, will use chunked processing
INFO: Split input of 150000 tokens into 37 chunks (max_chunk_size: 4096)
```

## 🤝 贡献

要将分块处理支持扩展到其他嵌入模型：

1. 检查模型与池化架构的兼容性
2. 使用不同文本长度进行测试
3. 与单块处理相比，验证嵌入质量
4. 提交包含测试用例和文档更新的 PR

## 🆕 增强功能

### max_embed_len 参数

新的 `max_embed_len` 参数提供：

- **简化配置**：无需 `VLLM_ALLOW_LONG_MAX_MODEL_LEN` 环境变量
- **灵活的输入验证**：接受超过 `max_model_len` 但不超过 `max_embed_len` 的输入
- **超长支持**：处理包含数百万令牌的文档
- **清晰的错误消息**：在输入超出限制时提供更好的反馈
- **向后兼容**：现有配置继续有效
