# vLLM 注意力基准测试套件

为 vLLM 注意力和 MLA 后端提供快速、灵活的基准测试，并带有扩展的批量规格语法。

## 快速开始

```bash
cd benchmarks/attention_benchmarks

# 运行预配置的基准测试
python benchmark.py --config configs/mla_decode.yaml
python benchmark.py --config configs/mla_mixed_batch.yaml
python benchmark.py --config configs/speculative_decode.yaml
python benchmark.py --config configs/standard_attention.yaml
python benchmark.py --config configs/reorder_threshold.yaml

# 或运行自定义基准测试
python benchmark.py \
    --backends flash flashinfer \
    --batch-specs "q2k" "8q1s1k" "2q2k_32q1s1k" \
    --output-csv results.csv
```

## 简化的批量规格语法

使用查询长度和序列长度来简洁表达工作负载：

```python
"q2k"              # 2048 token 预填充 (q_len=2048, seq_len=2048)
"q1s1k"            # 解码：1 个 token，1K 序列
"8q1s1k"           # 8 个解码请求
"q4s1k"            # 4-token 扩展（例如推测解码）
"2q2k_32q1s1k"     # 混合：2 个预填充 + 32 个解码
"16q4s1k"          # 16 个推测解码（每个 4 个 token）
```

### 语法规则

```text
格式: (<count>?) q<q_len>(k?) (s<seq_len>(k?))?

- count:   相同请求的数量（可选，默认=1）
- q_len:   查询长度（新 token 的数量）
- seq_len: 总序列长度（可选，预填充时默认为 q_len）
- 'k':     将值乘以 1024

混合批次：使用 _ 组合（例如 "2q2k_32q1s1k"）
```

**注意**：解码、预填充和推测解码只是不同的查询长度，不需要特殊语法！

## 预配置的基准测试

该套件包含多个预配置的 YAML 基准测试配置：

### MLA 解码基准测试

测试不同批量大小和序列长度下各种 MLA 后端的纯解码性能。

```bash
python benchmark.py --config configs/mla_decode.yaml
```

### MLA 混合批次基准测试

测试混合预填充和解码批次下的分块预填充性能。

```bash
python benchmark.py --config configs/mla_mixed_batch.yaml
```

### 推测解码基准测试

测试推测解码场景（K-token 验证）和 reorder_batch_threshold 优化。

```bash
python benchmark.py --config configs/speculative_decode.yaml
```

### 标准注意力基准测试

测试标准注意力后端（Flash/Triton/FlashInfer）的纯预填充、解码和混合批次。

```bash
python benchmark.py --config configs/standard_attention.yaml
```

### 重排序阈值研究

**问题**：在多大的查询长度下，预填充管线的速度会超过解码管线？

测试 9 种批量大小下从 1 到 1024 的查询长度，以找到交叉点。使用 `decode_vs_prefill` 模式比较每个查询长度的两条管线。

```bash
python benchmark.py --config configs/reorder_threshold.yaml
```

---

## 通用基准测试

`benchmark.py` 脚本处理**所有**后端，包括标准注意力和 MLA。

### 标准注意力（Flash/Triton/FlashInfer）

```bash
python benchmark.py \
    --backends flash triton flashinfer \
    --batch-specs "q2k" "8q1s1k" "2q2k_32q1s1k" \
    --num-layers 10 \
    --repeats 5 \
    --output-csv results.csv
```

### MLA 后端

```bash
# 比较所有 MLA 后端
python benchmark.py \
    --backends cutlass_mla flashinfer_mla flashattn_mla flashmla \
    --batch-specs "64q1s1k" "64q1s4k" \
    --output-csv mla_results.csv
```

### 参数扫描

使用 `--sweep-param` 和 `--sweep-values` 从命令行运行参数扫描：

#### CUTLASS MLA num-splits 优化

**问题**：CUTLASS MLA 的最优 `num_kv_splits` 是多少？

```bash
python benchmark.py \
    --backend cutlass_mla \
    --batch-specs "64q1s1k" "64q1s4k" "64q1s16k" \
    --sweep-param num_kv_splits \
    --sweep-values 1 2 4 8 16 \
    --output-json optimal_splits.json
```

#### 重排序批次阈值优化

**问题**：推测解码的最优 `reorder_batch_threshold` 是多少？

```bash
python benchmark.py \
    --backend flashmla \
    --batch-specs "q4s1k" "q8s2k" \
    --sweep-param reorder_batch_threshold \
    --sweep-values 1 4 16 64 256 512 \
    --output-csv threshold_sweep.csv
```

### 所有命令行选项

```text
--config CONFIG                     # YAML 配置文件的路径（覆盖其他参数）
--backends BACKEND [BACKEND ...]    # flash, triton, flashinfer, cutlass_mla,
                                    # flashinfer_mla, flashattn_mla, flashmla
--backend BACKEND                   # 单个后端（替代 --backends）
--batch-specs SPEC [SPEC ...]       # 使用扩展语法的批量规格

# 模型配置
--num-layers N                      # 层数
--head-dim N                        # 注意力头维度
--num-q-heads N                     # 查询头数
--num-kv-heads N                    # KV 头数
--block-size N                      # 块大小

# 基准测试设置
--device DEVICE                     # 设备（默认：cuda:0）
--repeats N                         # 重复次数
--warmup-iters N                    # 预热的迭代次数
--profile-memory                    # 分析内存使用情况

# 参数扫描
--sweep-param PARAM                 # 要扫描的参数名称（例如 num_kv_splits、
                                    # reorder_batch_threshold）
--sweep-values N [N ...]            # 要扫描的参数值

# 输出
--output-csv FILE                   # 保存为 CSV
--output-json FILE                  # 保存为 JSON
```

## 硬件要求

| 后端 | 硬件 |
| ------- | -------- |
| Flash/Triton/FlashInfer | 任何 CUDA GPU |
| CUTLASS MLA | Blackwell (SM100+) |
| FlashAttn MLA | Hopper (SM90+) |
| FlashMLA | Hopper (SM90+) |
| FlashInfer-MLA | 任何 CUDA GPU |

## 直接使用 MLA Runner

所有 MLA 后端都可通过 `mla_runner.run_mla_benchmark()` 使用：

```python
from mla_runner import run_mla_benchmark
from common import BenchmarkConfig

config = BenchmarkConfig(
    backend="cutlass_mla",
    batch_spec="64q1s4k",
    num_layers=10,
    head_dim=576,
    num_q_heads=128,
    num_kv_heads=1,
    block_size=128,
    device="cuda:0",
    repeats=5,
    warmup_iters=3,
)

# CUTLASS MLA 使用特定的 num_kv_splits
result = run_mla_benchmark("cutlass_mla", config, num_kv_splits=4)
print(f"时间: {result.mean_time:.6f}s")

# FlashInfer-MLA
result = run_mla_benchmark("flashinfer_mla", config)

# FlashAttn MLA (Hopper SM90+)
result = run_mla_benchmark("flashattn_mla", config, reorder_batch_threshold=64)

# FlashMLA (Hopper SM90+)
result = run_mla_benchmark("flashmla", config, reorder_batch_threshold=64)
```

## Python API

```python
from batch_spec import parse_batch_spec, format_batch_spec, get_batch_stats
from common import BenchmarkConfig, BenchmarkResult, ResultsFormatter

# 解析批量规格
requests = parse_batch_spec("2q2k_q4s1k_32q1s1k")
print(format_batch_spec(requests))
# "2 prefill (2x2k), 1 extend (1xq4kv1k), 32 decode (32x1k)"

# 获取批量统计信息
stats = get_batch_stats(requests)
print(f"总 token 数: {stats['total_tokens']}")
print(f"解码数: {stats['num_decode']}, 预填充数: {stats['num_prefill']}")

# 格式化结果
formatter = ResultsFormatter()
formatter.save_csv(results, "output.csv")
formatter.save_json(results, "output.json")
```

## 小贴士

**1. 预热很重要** - 使用 `--warmup-iters 10` 以获得稳定结果

**2. 多次重复** - 使用 `--repeats 20` 以降低方差

**3. 保存结果** - 始终使用 `--output-csv` 或 `--output-json`

**4. 逐步测试** - 从 `--num-layers 1 --repeats 1` 开始

**5. 扩展语法** - 利用推测解码、分块预填充模式

**6. 参数扫描** - 使用 `--sweep-param` 和 `--sweep-values` 找到最优值
