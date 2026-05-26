# EPD 正确性测试

此测试验证 EPD（编码器-预填充-解码）分离式架构是否能产生与基线单个实例相同的输出。

## 测试内容

- **基线**: 单个 vLLM 实例服务多模态模型
- **EPD (1E+1PD)**: 1 个编码器 + 1 个预填充-解码实例
- **基线 (1P+1D)**: 1 个预填充 + 1 个解码实例
- **EPD (1E+1P+1D)**: 1 个编码器 + 1 个预填充 + 1 个解码实例

该测试确保分离式编码产生与基线**完全相同**的输出。

请注意，当前的 PD 分离式设置可能与单个实例产生略微不同的结果。因此，我们需要 1P+1D 的结果作为 1E+1P+1D 的基线。

有关 EPD 功能的详细说明，请参阅[分离式编码器功能文档](../../../../docs/features/disagg_encoder.md)。

## 文件

- `run_epd_correctness_test.sh` - 主测试脚本（启动所有实例并运行测试）
- `test_epd_correctness.py` - Python 测试脚本（比较输出）

## 使用方法

### 多模态提示（默认）

```bash
cd vllm
./tests/v1/ec_connector/integration/run_epd_correctness_test.sh
```

此测试使用实际的多模态（图像）提示运行。

### 纯文本提示

```bash
cd vllm
USE_MM_PROMPTS=0 ./tests/v1/ec_connector/integration/run_epd_correctness_test.sh
```

此测试使用纯文本提示快速运行，以验证设置是否正常工作。

### 自定义配置

```bash
# 使用特定的 GPU
GPU_E=0 GPU_PD=1 GPU_P=1 GPU_D=2 bash ./tests/v1/ec_connector/integration/run_epd_correctness_test.sh

# 使用特定的端口
ENDPOINT_PORT=10001 bash ./tests/v1/ec_connector/integration/run_epd_correctness_test.sh

# 使用特定的模型
MODEL="Qwen/Qwen2.5-VL-3B-Instruct" bash ./tests/v1/ec_connector/integration/run_epd_correctness_test.sh

# 使用特定的存储路径
EC_SHARED_STORAGE_PATH="/tmp/my_ec_cache" bash ./tests/v1/ec_connector/integration/run_epd_correctness_test.sh
```

## 工作原理

### 步骤 1：基线

1. 在 GPU 上启动单个 vLLM 实例
2. 运行测试提示（多模态或纯文本）
3. 将输出保存到 `.vllm_epd_baseline.txt`
4. 关闭实例

### 步骤 2：EPD (1E + 1PD)

1. 清除编码器缓存存储
2. 启动实例和代理
3. 运行相同的测试提示
4. 断言输出与基线完全匹配
5. 关闭实例

### 步骤 3：EPD (1E + 1P + 1D)

1. 清除编码器缓存存储
2. 启动实例和代理
3. 运行相同的测试提示
4. 断言输出与基线完全匹配
5. 关闭实例

## 测试场景

### 多模态提示（--use_mm_prompts）

测试编码器缓存传输：

- 单张图像查询
- 单个请求中的多张图像
- 混合图像和文本
- 带有详细问题的图像

### 纯文本提示（默认）

快速合理性检查：

- 简单的文本查询
- 纯文本解释
- 验证代理路由是否正常工作

## 预期行为

### ✅ 测试通过条件

- 所有分离式输出与基线输出完全匹配
- 实例启动期间无错误
- 编码器缓存正确保存和加载
- 代理正确路由请求

### ❌ 测试失败条件

- 基线和分离式输出之间存在差异
- 服务器启动失败
- 未找到编码器缓存（应回退到本地执行）
- 代理路由错误

## 备注

- 测试使用确定性生成（`temperature=0.0`，`seed=42`）
- 编码器缓存应能够精确复现输出
- 测试完成后清理所有实例和缓存文件
- 可以安全地多次运行（幂等）
- 我们使用 NixlConnector 设置 PD 分离部分。有关 EPD 的详细信息，请阅读 `examples/disaggregated/disaggregated_encoder/README.md`

## 要求

- 多个 GPU（1E+1P+1D 需要 3 个，1E+1PD 需要 2 个，基线需要 1 个）
    - 1E+1P+1D 现在可以通过将 E 和 P 分配在同一 GPU 上以 2 个 GPU 运行。
- 多模态模型（例如，Qwen2.5-VL-3B-Instruct）
- 互联网访问（用于访问 vllm 测试图像）

## 调试

### 检查日志

日志和基线输出默认保存在 `/tmp/` 中。
可以通过更改环境变量进行自定义。

### 检查编码器缓存

```bash
# 验证缓存文件是否已创建
ls -la $EC_SHARED_STORAGE_PATH/

# 应看到以 mm_hash 命名的目录
# 每个目录包含 encoder_cache.safetensors
```

### 手动测试

运行各个组件：

```bash
# 仅基线
python test_epd_correctness.py \
    --service_url http://localhost:8000 \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --mode baseline \
    --baseline_file test_output.txt \
    --use_mm_prompts

# 仅分离式（需要基线输出文件！）
python test_epd_correctness.py \
    --service_url http://localhost:8000 \
    --model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --mode disagg \
    --baseline_file test_output.txt \
    --use_mm_prompts
```
