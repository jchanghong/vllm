# 使用多轮对话进行 KV 缓存卸载基准测试

`benchmark_serving_multi_turn.py` 的（pip）依赖项在 `requirements.txt` 中。

首先启动您的模型服务

```bash
export MODEL_PATH=/models/meta-llama/Meta-Llama-3.1-8B-Instruct/

vllm serve $MODEL_PATH --served-model-name Llama
```

变量 `MODEL_PATH` 应指向模型文件的路径（例如从 huggingface 下载的模型）。

## 合成多轮对话

下载以下文本文件（用于生成合成对话）

```bash
wget https://www.gutenberg.org/ebooks/1184.txt.utf-8
mv 1184.txt.utf-8 pg1184.txt
```

文件名 `pg1184.txt` 在 `generate_multi_turn.json` 中使用（参见 `"text_files"`）。

但您也可以根据需要使用其他文本文件（不要求使用此特定文件）。

然后运行基准测试脚本

```bash
export MODEL_PATH=/models/meta-llama/Meta-Llama-3.1-8B-Instruct/

python benchmark_serving_multi_turn.py --model $MODEL_PATH --served-model-name Llama \
--input-file generate_multi_turn.json --num-clients 2 --max-active-conversations 6
```

您可以编辑 `generate_multi_turn.json` 文件来更改对话参数（对话轮数等）。

如果成功，您将看到以下输出

```bash
----------------------------------------------------------------------------------------------------
Statistics summary:
runtime_sec = 215.810
requests_per_sec = 0.769
----------------------------------------------------------------------------------------------------
                   count     mean     std      min      25%      50%      75%      90%      99%      max
ttft_ms            166.0    78.22   67.63    45.91    59.94    62.26    64.43    69.66   353.18   567.54
tpot_ms            166.0    25.37    0.57    24.40    25.07    25.31    25.50    25.84    27.50    28.05
latency_ms         166.0  2591.07  326.90  1998.53  2341.62  2573.01  2860.10  3003.50  3268.46  3862.94
input_num_turns    166.0     7.43    4.57     1.00     3.00     7.00    11.00    13.00    17.00    17.00
input_num_tokens   166.0  2006.20  893.56   522.00  1247.75  2019.00  2718.00  3233.00  3736.45  3899.00
output_num_tokens  166.0   100.01   11.80    80.00    91.00    99.00   109.75   116.00   120.00   120.00
output_num_chunks  166.0    99.01   11.80    79.00    90.00    98.00   108.75   115.00   119.00   119.00
----------------------------------------------------------------------------------------------------
```

如果使用 `--warmup-step` 运行，摘要还将包含 `warmup_runtime_sec`
和 `total_runtime_incl_warmup_sec`（而 `runtime_sec` 继续反映仅基准测试的运行时间，以便报告的吞吐量保持可比性）。

### 用于合成对话生成的 JSON 配置文件

输入标志 `--input-file` 用于确定基准测试的输入对话。<br/>
当输入是一个包含 `"filetype": "generate_conversations"` 字段的 JSON 文件时，工具将生成合成的多轮（问答）对话。

`generate_multi_turn.json` 是一个示例文件。

该文件必须包含 `prompt_input` 和 `prompt_output` 部分。

`prompt_input` 部分必须包含 `num_turns`、`prefix_num_tokens` 和 `num_tokens`：

* `num_turns` - 对话中的总轮数（包括用户和助手）。<br/>
最终值将始终四舍五入为偶数，以便每轮用户问题都有回复。
* `prefix_num_tokens` - 仅在对话中**第一个用户轮次**开始时添加的 token（每个对话唯一）。
* `num_tokens` - 每个**用户**消息（一轮）的总 token 长度。

`prompt_output` 部分必须包含 `num_tokens`：

* `num_tokens` - 每个**助手**消息（一轮）的总 token 长度。

### 合成对话生成的随机分布

在创建输入 JSON 文件（如 `generate_multi_turn.json`）时，<br/>
每个数字字段（如 `num_turns` 或 `num_tokens`）都需要一个分布。<br/>
该分布决定了如何为该字段随机采样值。

以下是可用的分布。

**注意：** `max` 字段（用于 lognormal、zipf 和 poisson）可用于将采样值限制在上限。</br>
可用于确保每个请求中的总 token 数不超过 `--max-model-len`。

#### constant

```json
{
    "distribution": "constant",
    "value": 500
}
```

* `value` - 固定的整数值（始终返回相同的数字）。

#### uniform

```json
{
    "distribution": "uniform",
    "min": 12,
    "max": 18
}
```

* `min` - 最小值（包含）。
* `max` - 最大值（包含），应大于或等于 min。

#### lognormal

```json
{
    "distribution": "lognormal",
    "average": 1000,
    "max": 5000
}
```

您可以通过两种方式之一参数化对数正态分布：

使用平均值和可选的中位数比：

* `average` - 分布的目标平均值。
* `median_ratio` - 中位数与平均值的比率；控制偏度。必须在 (0, 1) 范围内。

使用底层正态分布的参数：

* `mean` - 底层正态分布的均值。
* `sigma` - 底层正态分布的标准差。

#### zipf

```json
{
    "distribution": "zipf",
    "alpha": 1.2,
    "max": 100
}
```

* `alpha` - 偏度参数 (> 1)。值越大，越倾向于较小的整数。

#### poisson

```json
{
    "distribution": "poisson",
    "alpha": 10,
    "max": 50
}
```

* `alpha` - 期望值 (λ)。也是分布的方差。

## ShareGPT 对话

要使用 ShareGPT 数据运行，请下载以下 ShareGPT 数据集：
`https://huggingface.co/datasets/philschmid/sharegpt-raw/blob/main/sharegpt_20230401_clean_lang_split.json`

使用 `convert_sharegpt_to_openai.py` 脚本将数据集转换为 `benchmark_serving_multi_turn.py` 支持的格式

```bash
python convert_sharegpt_to_openai.py sharegpt_20230401_clean_lang_split.json sharegpt_conv_128.json --seed=99 --max-items=128
```

该脚本将 ShareGPT 数据集转换为具有标准用户/助手角色的数据集。

标志 `--max-items=128` 用于从原始数据集中采样 128 个对话（根据需要更改）。

将输出的 JSON 文件 `sharegpt_conv_128.json` 用作 `benchmark_serving_multi_turn.py` 的 `--input-file`。
