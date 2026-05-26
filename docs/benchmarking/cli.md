# 基准测试 CLI

本指南将指导你使用 vLLM 支持的各种数据集运行基准测试。

这是一份持续性文档，随着新功能和数据集的推出而更新。

!!! tip
    本页描述的基准测试主要用于评估特定的 vLLM 功能以及回归测试。

    对于生产环境 vLLM 服务器的基准测试，我们推荐使用 [GuideLLM](https://github.com/vllm-project/guidellm)，这是一个成熟的性能基准测试框架，提供实时进度更新和自动报告生成。它在数据集加载、请求格式和工作负载模式方面也比 `vllm bench serve` 更灵活。

## 数据集概述

<style>
th {
  min-width: 0 !important;
}
</style>

| 数据集 | 在线 | 离线 | 数据路径 |
| ------- | ------ | ------- | --------- |
| ShareGPT | ✅ | ✅ | `wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json` |
| ShareGPT4V (图像) | ✅ | ✅ | `wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json`<br>注意，图像需要单独下载。例如，下载 COCO 2017 训练图像：<br>`wget http://images.cocodataset.org/zips/train2017.zip` |
| ShareGPT4Video (视频) | ✅ | ✅ | `git clone https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video` |
| BurstGPT | ✅ | ✅ | `wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv` |
| Sonnet (已弃用) | ✅ | ✅ | 本地文件：`benchmarks/sonnet.txt` |
| Random | ✅ | ✅ | `synthetic` |
| RandomMultiModal (图像/视频) | ✅ | ✅ | `synthetic` |
| RandomForReranking | ✅ | ✅ | `synthetic` |
| Prefix Repetition | ✅ | ✅ | `synthetic` |
| HuggingFace-VisionArena | ✅ | ✅ | `lmarena-ai/VisionArena-Chat` |
| HuggingFace-MMVU | ✅ | ✅ | `yale-nlp/MMVU` |
| HuggingFace-InstructCoder | ✅ | ✅ | `likaixin/InstructCoder` |
| HuggingFace-AIMO | ✅ | ✅ | `AI-MO/aimo-validation-aime`, `AI-MO/NuminaMath-1.5`, `AI-MO/NuminaMath-CoT` |
| HuggingFace-Other | ✅ | ✅ | `lmms-lab/LLaVA-OneVision-Data`, `Aeala/ShareGPT_Vicuna_unfiltered` |
| HuggingFace-MTBench | ✅ | ✅ | `philschmid/mt-bench` |
| HuggingFace-HumanEval | ✅ | ✅ | `openai/openai_humaneval` |
| HuggingFace-GSM8K | ✅ | ✅ | `openai/gsm8k` |
| HuggingFace-Blazedit | ✅ | ✅ | `vdaita/edit_5k_char`, `vdaita/edit_10k_char` |
| HuggingFace-ASR | ✅ | ✅ | `openslr/librispeech_asr`, `facebook/voxpopuli`,  `LIUM/tedlium`, `edinburghcstr/ami`,        `speechcolab/gigaspeech`,        `kensho/spgispeech` |
| Spec Bench | ✅ | ✅ | `wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl` |
| SPEED-Bench | ✅ | ✅ | `curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \| python3 -` |
| Custom | ✅ | ✅ | 本地文件：`data.jsonl` |
| Custom Audio | ✅ | ✅ | 本地文件：`audio_data.jsonl` |
| Custom Image | ✅ | ✅ | 本地文件：`image_data.jsonl` |

图例：

- ✅ - 支持
- 🟡 - 部分支持
- 🚧 - 即将支持

!!! note
    HuggingFace 数据集的 `dataset-name` 应设置为 `hf`。
    对于本地 `dataset-path`，请将 `hf-name` 设置为其 HuggingFace ID，例如：

    ```bash
    --dataset-path /datasets/VisionArena-Chat/ --hf-name lmarena-ai/VisionArena-Chat
    ```

## 示例

### 🚀 在线基准测试

<details class="admonition abstract" markdown="1">
<summary>展开</summary>

首先启动模型服务：

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B
```

然后运行基准测试脚本：

```bash
# 下载数据集
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <你的数据路径>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

如果成功，你将看到以下输出：

```text
============ 服务基准测试结果 ============
成功请求数：                     10
基准测试持续时间（秒）：                  5.78
输入 token 总数：                      1369
生成 token 总数：                  2212
请求吞吐量（请求/秒）：              1.73
输出 token 吞吐量（token/秒）：         382.89
总 token 吞吐量（token/秒）：          619.85
---------------首 token 时间----------------
平均 TTFT（毫秒）：                          71.54
中位数 TTFT（毫秒）：                        73.88
P99 TTFT（毫秒）：                           79.49
-----每输出 token 时间（排除首 token）------
平均 TPOT（毫秒）：                          7.91
中位数 TPOT（毫秒）：                        7.96
P99 TPOT（毫秒）：                           8.03
---------------token 间延迟----------------
平均 ITL（毫秒）：                           7.74
中位数 ITL（毫秒）：                         7.70
P99 ITL（毫秒）：                            8.39
==================================================
```

#### 结果可视化

`--plot-timeline` 和 `--plot-dataset-stats` 可分别用于生成请求完成时间线和数据集提示及输出 token 统计图，这对调试目的或深入分析非常有用。

```bash
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --endpoint /v1/completions \
    --dataset-name sharegpt \
    --dataset-path <你的数据路径>/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 100 \
    --plot-timeline \
    --timeline-itl-thresholds 2,5 \
    --plot-dataset-stats \
    --save-result
```

##### 交互式时间线

生成的时间线是一个交互式可视化的 HTML 文件，可在大多数浏览器中渲染。要自定义 ITL 颜色阈值，可以使用 `--timeline-itl-thresholds` 标志（默认值：25ms, 50ms）

示例输出：

<iframe src="../assets/contributing/vllm_bench_serve_timeline.html" width="100%" height="600" frameborder="0"></iframe>

##### 数据集统计

生成的图表显示输入提示和输出 token 的分布。

示例输出：![数据集统计](../assets/contributing/vllm_bench_serve_dataset_stats.png)

#### 自定义数据集

如果你想要基准测试的数据集在 vLLM 中尚不支持，你仍然可以使用 `CustomDataset` 对其进行基准测试。推理时，使用选项 `--dataset-name custom`。你的数据需要采用 `.jsonl` 格式，并且每个条目需要包含 "prompt" 字段，例如 data.jsonl：

```json
{"prompt": "What is the capital of India?"}
{"prompt": "What is the capital of Iran?"}
{"prompt": "What is the capital of China?"}
```

```bash
# 启动服务器
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

```bash
# 运行基准测试脚本
vllm bench serve --port 9001 --save-result --save-detailed \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --endpoint /v1/completions \
  --dataset-name custom \
  --dataset-path <你的数据 jsonl 路径> \
  --custom-skip-chat-template \
  --num-prompts 80 \
  --max-concurrency 1 \
  --temperature=0.3 \
  --top-p=0.75 \
  --result-dir "./log/"
```

如果你的数据已经包含聊天模板，可以使用 `--custom-skip-chat-template` 跳过应用聊天模板。

#### 自定义音频数据集

如果你想要基准测试的音频数据集在 vLLM 中尚不支持，你可以使用 `CustomAudioDataset` 对其进行基准测试。推理时，使用选项 `--dataset-name custom_audio`。你的数据需要采用 `.jsonl` 格式，并且每个条目需要包含 "prompt" 和 "audio" 字段，例如 `audio_data.jsonl`：

```json
{"prompt": "What does this audio say?", "audio": "/path/to/audio_1.wav"}
{"prompt": "Transcribe the audio.", "audio": "/path/to/audio_2.wav"}
```

- **支持的模型：** `CustomAudioDataset` 类支持两种类型的音频模型：ASR 模型（例如 Whisper），不需要 "prompt" 字段；以及多模态音频-文本聊天模型（例如 Qwen2-Audio）。由于这些模型类型在推理时需要不同的参数，我们提供两个示例。

- **示例 1：Whisper**

Whisper 是专用的 ASR 编码器-解码器模型，因此它使用 `--backend openai-audio` 和 `--endpoint /v1/audio/transcriptions`。

```bash
# 启动服务器
vllm serve openai/whisper-tiny
```

```bash
vllm bench serve \
  --model openai/whisper-tiny \
  --backend openai-audio \
  --endpoint /v1/audio/transcriptions \
  --dataset-name custom_audio \
  --dataset-path audio_data.jsonl \
  --no-oversample \
  --custom-output-len 256 \
  --save-result \
  --save-detailed \
  --result-filename whisper_bench.json
```

- **示例 2：Qwen2-Audio**

Qwen2-Audio 是一个多模态聊天模型，可以执行 ASR 和语音分析，因此它使用 `--backend openai-chat` 和 `--endpoint /v1/chat/completions`。它还需要 `--enable-multimodal-chat` 来启用多模态聊天转换。

```bash
vllm bench serve \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name custom_audio \
  --dataset-path audio_data.jsonl \
  --no-oversample \
  --custom-output-len 256 \
  --enable-multimodal-chat \
  --save-result \
  --save-detailed \
  --result-filename qwen_bench.json
```

#### 自定义图像数据集

如果你想要基准测试的图像数据集在 vLLM 中尚不支持，你可以使用 `CustomImageDataset` 对其进行基准测试。推理时，使用选项 `--dataset-name custom_image`。你的数据需要采用 `.jsonl` 格式，并且每个条目需要包含 "prompt" 和 "image_files" 字段，例如 `image_data.jsonl`：

```json
{"prompt": "How many animals are present in the given image?", "image_files": ["/path/to/image/folder/horsepony.jpg"]}
{"prompt": "What colour is the bird shown in the image?", "image_files": ["/path/to/image/folder/flycatcher.jpeg"]}
```

```bash
# 此处需要具有视觉能力的模型
vllm serve Qwen/Qwen2-VL-7B-Instruct
```

```bash
# 运行基准测试脚本
vllm bench serve--save-result --save-detailed \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name custom_image \
  --dataset-path <你的图像数据 jsonl 路径> \
  --allowed-local-media-path /path/to/image/folder
```

注意，对于多模态输入，我们需要使用 `openai-chat` 后端和 `/v1/chat/completions` 端点。

#### 视觉语言模型的 VisionArena 基准测试

```bash
# 此处需要具有视觉能力的模型
vllm serve Qwen/Qwen2-VL-7B-Instruct
```

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --hf-split train \
  --num-prompts 1000
```

#### 使用推测解码的 InstructCoder 基准测试

``` bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-config $'{"method": "ngram",
    "num_speculative_tokens": 5, "prompt_lookup_max": 5,
    "prompt_lookup_min": 2}'
```

``` bash
vllm bench serve \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name hf \
    --dataset-path likaixin/InstructCoder \
    --num-prompts 2048
```

#### 使用推测解码的 Spec Bench 基准测试

``` bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-config $'{"method": "ngram",
    "num_speculative_tokens": 5, "prompt_lookup_max": 5,
    "prompt_lookup_min": 2}'
```

[SpecBench 数据集](https://github.com/hemingkx/Spec-Bench)

运行所有类别：

``` bash
# 使用以下命令下载数据集：
# wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl

vllm bench serve \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name spec_bench \
    --dataset-path "<你的下载路径>/data/spec_bench/question.jsonl" \
    --num-prompts -1
```

可用类别包括 `[writing, roleplay, reasoning, math, coding, extraction, stem, humanities, translation, summarization, qa, math_reasoning, rag]`。

仅运行特定类别，例如 "summarization"：

``` bash
vllm bench serve \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name spec_bench \
    --dataset-path "<你的下载路径>/data/spec_bench/question.jsonl" \
    --num-prompts -1
    --spec-bench-category "summarization"
```

#### 使用推测解码的 SPEED-Bench 基准测试

[SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) 是一个统一且多样化的推测解码数据集，支持使用 Qualitative 分割进行接受率和长度测量，以及使用 5 种输入序列长度配置（1k、2k、8k、16k、32k）的 Throughput 分割进行吞吐量测量。

!!! note
    该数据集受 [NVIDIA 评估数据集许可协议](https://huggingface.co/datasets/nvidia/SPEED-Bench/blob/main/License.pdf) 管辖。用户选择的每个数据集，用户有责任检查数据集许可证是否适合预期用途。`prepare.py` 脚本会自动从所有源数据集获取数据。

首先，使用以下一行命令将数据集下载到文件夹：

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \| python3 -
```

该命令还支持以下参数：

- `--config`：仅下载数据集的一个子集：`qualitative`、`throughput_1k`、`throughput_2k`、`throughput_8k`、`throughput_16k` 和 `throughput_32k`。默认情况下，将下载所有子集。
- `--output_dir`：下载到指定的文件夹。默认情况下，将下载到当前目录。

启动一个带有推测解码的服务器：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --speculative-config $'{"method": "eagle3",
    "num_speculative_tokens": 3,
    "model": "nvidia/Llama-3.3-70B-Instruct-Eagle3"}'
```

运行 Qualitative 分割中的所有类别：

```bash
vllm bench serve \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dataset-name speed_bench \
    --dataset-path "<你的下载路径>/data/speed_bench" \
    --num-prompts -1
```

可用类别包括 `[writing, roleplay, reasoning, math, coding, stem, humanities, multilingual, summarization, qa, rag]`。

仅运行特定类别，例如 "multilingual"：

```bash
vllm bench serve \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dataset-name speed_bench \
    --dataset-path "<你的下载路径>/data/speed_bench" \
    --num-prompts -1
    --speed-bench-category "multilingual"
```

运行 Throughput 分割（2k ISL）中的所有类别：

```bash
vllm bench serve \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dataset-name speed_bench \
    --speed-bench-dataset-subset throughput_2k
    --dataset-path "<你的下载路径>/data/speed_bench/" \
    --num-prompts -1
```

可用类别包括 `[high_entropy, mixed, low_entropy]`，其中高熵数据包含非结构化数据（如创意写作），而低熵数据包含更结构化的数据（如编码），更多细节请参见数据集卡片。

#### 其他 HuggingFace 数据集示例

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct
```

`lmms-lab/LLaVA-OneVision-Data`：

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path lmms-lab/LLaVA-OneVision-Data \
  --hf-split train \
  --hf-subset "chart2text(cauldron)" \
  --num-prompts 10
```

`Aeala/ShareGPT_Vicuna_unfiltered`：

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
  --hf-split train \
  --num-prompts 10
```

`AI-MO/aimo-validation-aime`：

``` bash
vllm bench serve \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path AI-MO/aimo-validation-aime \
    --num-prompts 10 \
    --seed 42
```

`philschmid/mt-bench`：

``` bash
vllm bench serve \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num-prompts 80
```

`openai/openai_humaneval`：

``` bash
vllm bench serve \
    --model NousResearch/Hermes-3-Llama-3.1-8B \
    --dataset-name hf \
    --dataset-path openai/openai_humaneval \
    --num-prompts 80
```

`openai/gsm8k`：

``` bash
vllm bench serve \
    --model NousResearch/Hermes-3-Llama-3.1-8B \
    --dataset-name hf \
    --dataset-path openai/gsm8k \
    --num-prompts 80
```

`vdaita/edit_5k_char` 或 `vdaita/edit_10k_char`：

``` bash
vllm bench serve \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path vdaita/edit_5k_char \
    --num-prompts 90 \
    --blazedit-min-distance 0.01 \
    --blazedit-max-distance 0.99
```

`openslr/librispeech_asr`、`facebook/voxpopuli`、`LIUM/tedlium`、`edinburghcstr/ami`、`speechcolab/gigaspeech`、`kensho/spgispeech`

```bash
vllm bench serve \
    --model openai/whisper-large-v3-turbo \
    --backend openai-audio \
    --dataset-name hf \
    --dataset-path facebook/voxpopuli --hf-subset en --hf-split test --no-stream --trust-remote-code \
    --num-prompts 99999999 \
    --no-oversample \
    --endpoint /v1/audio/transcriptions \
    --ready-check-timeout-sec 600 \
    --save-result \
    --max-concurrency 512
```

#### 使用采样参数运行

当使用 OpenAI 兼容的后端（如 `vllm`）时，可以指定可选的采样参数。客户端命令示例：

```bash
vllm bench serve \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <你的数据路径>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --top-k 10 \
  --top-p 0.9 \
  --temperature 0.5 \
  --num-prompts 10
```

#### 使用递增请求率运行

基准测试工具还支持在基准测试运行期间递增请求率。这对于压力测试服务器或在给定延迟预算下找到其可处理的最大吞吐量非常有用。

支持两种递增策略：

- `linear`：请求率从起始值线性增加到结束值。
- `exponential`：请求率呈指数增加。

以下参数可用于控制递增：

- `--ramp-up-strategy`：使用的递增策略（`linear` 或 `exponential`）。
- `--ramp-up-start-rps`：基准测试开始时的请求率。
- `--ramp-up-end-rps`：基准测试结束时的请求率。

#### 负载模式配置

vLLM 的基准测试服务脚本通过三个关键参数提供复杂的负载模式模拟能力，这些参数控制请求生成和并发行为：

##### 负载模式控制参数

- `--request-rate`：控制目标请求生成速率（每秒请求数）。设置为 `inf` 可进行最大吞吐量测试，设置为有限值可进行可控负载模拟。
- `--burstiness`：使用 Gamma 分布控制流量变异性（范围：> 0）。较低的值产生突发流量，较高的值产生均匀流量。
- `--max-concurrency`：限制并发的未完成请求。如果未提供此参数，则并发无限制。设置一个值以模拟背压。

这些参数协同工作，通过精心选择的默认值创建逼真的负载模式。`--request-rate` 参数默认值为 `inf`（无限），会立即发送所有请求以进行最大吞吐量测试。当设置为有限值时，它使用泊松过程（默认 `--burstiness=1.0`）或 Gamma 分布来实现逼真的请求计时。`--burstiness` 参数仅在 `--request-rate` 不是无限时生效 - 值为 1.0 时产生自然泊松流量，较低的值（0.1-0.5）产生突发模式，较高的值（2.0-5.0）产生均匀间隔。`--max-concurrency` 参数默认为 `None`（无限制），但可以设置以模拟负载均衡器或 API 网关限制并发连接的真实场景。当组合使用时，这些参数允许你模拟从无限制压力测试（`--request-rate=inf`）到具有逼真到达模式和资源限制的生产场景的一切情况。

`--burstiness` 参数在数学上使用 Gamma 分布控制请求到达模式，其中：

- 形状参数：`burstiness` 值
- 变异系数 (CV)：$\frac{1}{\sqrt{burstiness}}$
- 流量特征：
    - `burstiness = 0.1`：高突发流量 (CV ≈ 3.16) - 压力测试
    - `burstiness = 1.0`：自然泊松流量 (CV = 1.0) - 逼真模拟
    - `burstiness = 5.0`：均匀流量 (CV ≈ 0.45) - 可控负载测试

![负载模式示例](../assets/contributing/load-pattern-examples.png)

*图：每种用例的负载模式示例。顶行：请求到达时间线，显示随时间累积的请求数。底行：到达间隔时间分布，显示流量变化模式。每列代表不同的用例及其特定参数设置和产生的流量特征。*

按用例的负载模式建议：

| 用例             | Burstiness   | 请求率      | 最大并发数 | 描述                                                                              |
| ---                | ---          | ---             | ---             | ---                                                                                |
| 最大吞吐量         | 不适用          | 无限            | 有限            | **最常见**：在无限用户需求下模拟负载均衡器/网关限制                                |
| 逼真测试           | 1.0          | 中等 (5-20)     | 无限            | 用于基准性能的自然泊松流量模式                                                     |
| 压力测试           | 0.1-0.5      | 高 (20-100)     | 无限            | 测试弹性的挑战性突发模式                                                           |
| 延迟分析           | 2.0-5.0      | 低 (1-10)       | 无限            | 用于一致时序分析的均匀负载                                                         |
| 容量规划           | 1.0          | 可变            | 有限            | 在现实约束下测试资源限制                                                           |
| SLA 验证           | 1.0          | 目标速率        | SLA 限制        | 用于合规性测试的生产级约束                                                         |

这些负载模式有助于评估 vLLM 部署的不同方面，从基本性能特征到在挑战性流量条件下的弹性。

**最大吞吐量**模式（`--request-rate=inf --max-concurrency=<限制>`）是生产基准测试中最常用的配置。这模拟了实际部署架构，其中：

- 用户尽可能快地发送请求（无限速率）
- 负载均衡器或 API 网关控制最大并发连接数
- 系统在其并发限制下运行，揭示真实的吞吐量能力
- `--burstiness` 无效，因为当速率为无限时不控制请求时序

此模式有助于确定生产负载均衡器配置的最佳并发设置。

为了有效配置负载模式，特别是对于**容量规划**和**SLA 验证**用例，你需要了解系统的资源限制。在启动时，vLLM 报告直接影响到负载测试参数的 KV 缓存配置：

```text
GPU KV 缓存大小：15,728,640 tokens
每请求 8,192 tokens 的最大并发数：1920
```

其中：

- GPU KV 缓存大小：可以在所有并发请求之间缓存的总 token 数
- 最大并发数：在给定 `max_model_len` 下的理论最大并发请求数
- 计算方式：`max_concurrency = kv_cache_size / max_model_len`

使用 KV 缓存指标进行负载模式配置：

- 容量规划：将 `--max-concurrency` 设置为报告最大值的 80-90%，以测试现实资源限制
- SLA 验证：使用报告的最大值作为你的 SLA 限制，以确保合规性测试与生产容量匹配
- 逼真测试：接近理论极限时监控内存使用情况，以了解可持续的请求率
- 请求率指南：使用 KV 缓存大小来估算特定工作负载和序列长度的可持续请求率

</details>

### 📈 离线吞吐量基准测试

<details class="admonition abstract" markdown="1">
<summary>展开</summary>

```bash
vllm bench throughput \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset-name sonnet \
  --dataset-path vllm/benchmarks/sonnet.txt \
  --num-prompts 10
```

如果成功，你将看到以下输出：

```text
吞吐量：7.15 请求/秒，4656.00 总 token/秒，1072.15 输出 token/秒
提示 token 总数：  5014
输出 token 总数：  1500
```

#### 视觉语言模型的 VisionArena 基准测试

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 1000 \
  --hf-split train
```

`num prompt tokens` 现在包括图像 token 计数

```text
吞吐量：2.55 请求/秒，4036.92 总 token/秒，326.90 输出 token/秒
提示 token 总数：  14527
输出 token 总数：  1280
```

#### 使用推测解码的 InstructCoder 基准测试

``` bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm bench throughput \
    --dataset-name=hf \
    --dataset-path=likaixin/InstructCoder \
    --model=meta-llama/Meta-Llama-3-8B-Instruct \
    --input-len=1000 \
    --output-len=100 \
    --num-prompts=2048 \
    --async-engine \
    --speculative-config $'{"method": "ngram",
    "num_speculative_tokens": 5, "prompt_lookup_max": 5,
    "prompt_lookup_min": 2}'
```

```text
吞吐量：104.77 请求/秒，23836.22 总 token/秒，10477.10 输出 token/秒
提示 token 总数：  261136
输出 token 总数：  204800
```

#### 其他 HuggingFace 数据集示例

`lmms-lab/LLaVA-OneVision-Data`：

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path lmms-lab/LLaVA-OneVision-Data \
  --hf-split train \
  --hf-subset "chart2text(cauldron)" \
  --num-prompts 10
```

`Aeala/ShareGPT_Vicuna_unfiltered`：

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
  --hf-split train \
  --num-prompts 10
```

`AI-MO/aimo-validation-aime`：

```bash
vllm bench throughput \
  --model Qwen/QwQ-32B \
  --backend vllm \
  --dataset-name hf \
  --dataset-path AI-MO/aimo-validation-aime \
  --hf-split train \
  --num-prompts 10
```

使用 LoRA 适配器进行基准测试：

``` bash
# 下载数据集
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench throughput \
  --model meta-llama/Llama-2-7b-hf \
  --backend vllm \
  --dataset_path <你的数据路径>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset_name sharegpt \
  --num-prompts 10 \
  --max-loras 2 \
  --max-lora-rank 8 \
  --enable-lora \
  --lora-path yard1/llama-2-7b-sql-lora-test
```

#### 合成随机多模态 (random-mm)

生成合成多模态输入用于离线吞吐量测试，无需外部数据集。使用 `--backend vllm-chat` 以确保正确计算图像 token。

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name random-mm \
  --num-prompts 100 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```

</details>

### 🛠️ 结构化输出基准测试

<details class="admonition abstract" markdown="1">
<summary>展开</summary>

对结构化输出生成（JSON、grammar、regex）的性能进行基准测试。

#### 服务器设置

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B
```

#### JSON Schema 基准测试

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset json \
  --structured-output-ratio 1.0 \
  --request-rate 10 \
  --num-prompts 1000
```

#### 基于 Grammar 的生成基准测试

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset grammar \
  --structure-type grammar \
  --request-rate 10 \
  --num-prompts 1000
```

#### 基于正则表达式的生成基准测试

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset regex \
  --request-rate 10 \
  --num-prompts 1000
```

#### 基于选择的生成基准测试

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset choice \
  --request-rate 10 \
  --num-prompts 1000
```

#### XGrammar 基准测试数据集

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset xgrammar_bench \
  --request-rate 10 \
  --num-prompts 1000
```

</details>

### 📚 长文档问答基准测试

<details class="admonition abstract" markdown="1">
<summary>展开</summary>

对带有前缀缓存的长文档问答性能进行基准测试。

#### 基本长文档问答测试

```bash
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 16 \
  --document-length 2000 \
  --output-len 50 \
  --repeat-count 5
```

#### 不同的重复模式

```bash
# 随机模式（默认）- 随机打乱提示
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 8 \
  --document-length 3000 \
  --repeat-count 3 \
  --repeat-mode random

# Tile 模式 - 按顺序重复整个提示列表
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 8 \
  --document-length 3000 \
  --repeat-count 3 \
  --repeat-mode tile

# 交错模式 - 连续重复每个提示
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 8 \
  --document-length 3000 \
  --repeat-count 3 \
  --repeat-mode interleave
```

</details>

### 🗂️ 前缀缓存基准测试

<details class="admonition abstract" markdown="1">
<summary>展开</summary>

对自动前缀缓存的效率进行基准测试。

#### 固定提示与前缀缓存

```bash
python3 benchmarks/benchmark_prefix_caching.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-prompts 1 \
  --repeat-count 100 \
  --input-length-range 128:256
```

#### ShareGPT 数据集与前缀缓存

```bash
# 下载数据集
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python3 benchmarks/benchmark_prefix_caching.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset-path /path/ShareGPT_V3_unfiltered_cleaned_split.json \
  --enable-prefix-caching \
  --num-prompts 20 \
  --repeat-count 5 \
  --input-length-range 128:256
```

##### 前缀重复数据集

```bash
vllm bench serve \
  --backend openai \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset-name prefix_repetition \
  --num-prompts 100 \
  --prefix-repetition-prefix-len 512 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 5 \
  --prefix-repetition-output-len 128
```

</details>

### 🧪 哈希基准测试

<details class="admonition abstract" markdown="1">
<summary>展开</summary>

`benchmarks/` 目录下有两个辅助脚本，用于比较前缀缓存和相关工具使用的哈希选项。它们是独立的（无需服务器），有助于在生产环境中启用前缀缓存之前选择哈希算法。

- `benchmarks/benchmark_hash.py`：微基准测试，测量三种实现在代表性 `(bytes, tuple[int])` 负载上的每次调用延迟。

```bash
python benchmarks/benchmark_hash.py --iterations 20000 --seed 42
```

- `benchmarks/benchmark_prefix_block_hash.py`：端到端块哈希基准测试，跨多个伪块运行完整的前缀缓存哈希流水线（`hash_block_tokens`）并报告吞吐量。

```bash
python benchmarks/benchmark_prefix_block_hash.py --num-blocks 20000 --block-size 32 --trials 5
```

支持的算法：`sha256`、`sha256_cbor`、`xxhash`、`xxhash_cbor`。安装可选依赖以测试所有变体：

```bash
uv pip install xxhash cbor2
```

如果某个算法的依赖缺失，脚本将跳过它并继续运行。

</details>

### ⚡ 请求优先级基准测试

<details class="admonition abstract" markdown="1">
<summary>展开</summary>

对 vLLM 中请求优先级调度的性能进行基准测试。

#### 基本优先级测试

```bash
python3 benchmarks/benchmark_prioritization.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --input-len 128 \
  --output-len 64 \
  --num-prompts 100 \
  --scheduling-policy priority
```

#### 每个提示多个序列

```bash
python3 benchmarks/benchmark_prioritization.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --input-len 128 \
  --output-len 64 \
  --num-prompts 100 \
  --scheduling-policy priority \
  --n 2
```

</details>

### 👁️ 多模态基准测试

<details class="admonition abstract" markdown="1">
<summary>展开</summary>

对 vLLM 中多模态请求的性能进行基准测试。

#### 图像 (ShareGPT4V)

启动 vLLM：

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 1}' \
  --allowed-local-media-path /path/to/sharegpt4v/images
```

发送带有图像的请求：

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path /path/to/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k.json \
  --num-prompts 100 \
  --save-result \
  --result-dir ~/vllm_benchmark_results \
  --save-detailed \
  --endpoint /v1/chat/completions
```

#### 视频 (ShareGPT4Video)

启动 vLLM：

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"video": 1}' \
  --allowed-local-media-path /path/to/sharegpt4video/videos
```

发送带有视频的请求：

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path /path/to/ShareGPT4Video/llava_v1_5_mix665k_with_video_chatgpt72k_share4video28k.json \
  --num-prompts 100 \
  --save-result \
  --result-dir ~/vllm_benchmark_results \
  --save-detailed \
  --endpoint /v1/chat/completions
```

#### 合成随机图像 (random-mm)

生成合成图像输入以及随机文本提示，用于在无需外部数据集的情况下压力测试视觉模型。

注意：

- 对于在线基准测试，使用 `--backend openai-chat` 和端点 `/v1/chat/completions`。
- 对于离线基准测试，使用 `--backend vllm-chat`（参见[离线吞吐量基准测试](#-离线吞吐量基准测试)的示例）。

启动服务器（示例）：

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --mm-processor-kwargs max_pixels=1003520
```

基准测试。建议使用 `--ignore-eos` 标志来模拟真实响应。你可以通过 `random-output-len` 参数设置输出的大小。

示例 1：固定数量的项目和单一图像分辨率，强制生成约 40 个 token：

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name random-mm \
  --num-prompts 100 \
  --max-concurrency 10 \
  --random-prefix-len 25 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-range-ratio 0.2 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --random-mm-bucket-config '{(224, 224, 1): 1.0}' \
  --request-rate inf \
  --ignore-eos \
  --seed 42
```

每个请求的项目数可以通过传递多个图像桶来控制：

```bash
  --random-mm-base-items-per-request 2 \
  --random-mm-num-mm-items-range-ratio 0.5 \
  --random-mm-limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}' \
```

`random-mm` 特定的标志：

- `--random-mm-base-items-per-request`：每个请求的多模态项目基础数量。
- `--random-mm-num-mm-items-range-ratio`：在闭整数范围 [floor(n·(1−r)), ceil(n·(1+r))] 内均匀变化项目计数。设置 r=0 保持固定；r=1 允许 0 个项目。
- `--random-mm-limit-mm-per-prompt`：按模态的硬限制，例如 '{"image": 3, "video": 0}'。
- `--random-mm-bucket-config`：映射 (H, W, T) → 概率的字典。概率为 0 的条目被移除；剩余概率重新归一化以和为 1。图像使用 T=1。设置任何 T>1 用于视频（视频采样尚未支持）。

行为说明：

- 如果请求的基础项目数在提供的每个提示限制下无法满足，工具将报错而非静默截断。

采样工作原理：

- 通过从 `--random-mm-base-items-per-request` 和 `--random-mm-num-mm-items-range-ratio` 定义的整数范围中均匀采样来确定每个请求的项目数 k，然后将 k 限制为不超过每个模态限制的总和。
- 对于 k 个项目中的每一个，根据 `--random-mm-bucket-config` 中的归一化概率采样一个桶 (H, W, T)，同时跟踪已添加的每种模态的项目数。
- 如果某个模态（例如，图像）达到了 `--random-mm-limit-mm-per-prompt` 的限制，则排除该模态的所有桶，并在继续前重新归一化剩余的桶概率。这应被视为一种边缘情况，如果可以通过将 `--random-mm-limit-mm-per-prompt` 设置为一个大数字来避免此行为。注意，这可能会由于引擎配置 `--limit-mm-per-prompt` 而导致错误。
- 生成的请求包含 `multi_modal_data`（OpenAI Chat 格式）中的合成图像数据。当 `random-mm` 与 OpenAI Chat 后端一起使用时，提示保持为文本，MM 内容通过 `multi_modal_data` 附加。

</details>

### 🔬 多模态处理器基准测试

对多模态（MM）输入处理器流水线的每个阶段延迟进行基准测试，包括编码器前向传递。这对于分析视觉语言模型中的预处理瓶颈很有用。

<details class="admonition abstract" markdown="1">
<summary>展开</summary>

该基准测试测量每个请求的以下阶段：

| 阶段 | 描述 |
| ----- | ----------- |
| `get_mm_hashes_secs` | 对多模态输入进行哈希处理的时间 |
| `get_cache_missing_items_secs` | 查找处理器缓存的时间 |
| `apply_hf_processor_secs` | HuggingFace 处理器中的处理时间 |
| `merge_mm_kwargs_secs` | 合并多模态 kwargs 的时间 |
| `apply_prompt_updates_secs` | 更新提示 token 的时间 |
| `preprocessor_total_secs` | 预处理总时间 |
| `encoder_forward_secs` | 编码器模型前向传递时间 |
| `num_encoder_calls` | 每个请求的编码器调用次数 |

该基准测试还会报告每个请求的端到端延迟（TTFT + 解码时间）。使用 `--metric-percentiles` 选择要报告的百分位数（默认：p99），使用 `--output-json` 保存结果。

#### 合成数据基本示例 (random-mm)

```bash
vllm bench mm-processor \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name random-mm \
  --num-prompts 50 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```

#### 使用 HuggingFace 数据集

```bash
vllm bench mm-processor \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --hf-split train \
  --num-prompts 100
```

#### 预热、自定义百分位数和 JSON 输出

```bash
vllm bench mm-processor \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name random-mm \
  --num-prompts 200 \
  --num-warmups 5 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 1 \
  --metric-percentiles 50,90,95,99 \
  --output-json results.json
```

参见 [`vllm bench mm-processor`](../cli/bench/mm_processor.md) 获取完整参数参考。

</details>

### 嵌入基准测试

对 vLLM 中嵌入请求的性能进行基准测试。

<details class="admonition abstract" markdown="1">
<summary>展开</summary>

#### 文本嵌入

与使用 Completions API 或 Chat Completions API 的生成模型不同，你应该设置 `--backend openai-embeddings` 和 `--endpoint /v1/embeddings` 来使用 Embeddings API。

你可以使用任何文本数据集对模型进行基准测试，例如 ShareGPT。

启动服务器：

```bash
vllm serve jinaai/jina-embeddings-v3 --trust-remote-code
```

运行基准测试：

```bash
# 下载数据集
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --model jinaai/jina-embeddings-v3 \
  --backend openai-embeddings \
  --endpoint /v1/embeddings \
  --dataset-name sharegpt \
  --dataset-path <你的数据路径>/ShareGPT_V3_unfiltered_cleaned_split.json
```

#### 多模态嵌入

与使用 Completions API 或 Chat Completions API 的生成模型不同，你应该设置 `--endpoint /v1/embeddings` 来使用 Embeddings API。使用的后端取决于模型：

- CLIP：`--backend openai-embeddings-clip`
- VLM2Vec：`--backend openai-embeddings-vlm2vec`

对于其他模型，请在 [vllm/benchmarks/lib/endpoint_request_func.py](../../vllm/benchmarks/lib/endpoint_request_func.py) 中添加你自己的实现以匹配预期的指令格式。

你可以使用任何文本或多模态数据集对模型进行基准测试，只要模型支持即可。例如，你可以使用 ShareGPT 和 VisionArena 对视觉语言嵌入进行基准测试。

服务并基准测试 CLIP：

```bash
# 在另一个进程中运行
vllm serve openai/clip-vit-base-patch32

# 服务器启动后逐个运行
# 下载数据集
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --model openai/clip-vit-base-patch32 \
  --backend openai-embeddings-clip \
  --endpoint /v1/embeddings \
  --dataset-name sharegpt \
  --dataset-path <你的数据路径>/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
  --model openai/clip-vit-base-patch32 \
  --backend openai-embeddings-clip \
  --endpoint /v1/embeddings \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat
```

服务并基准测试 VLM2Vec：

```bash
# 在另一个进程中运行
vllm serve TIGER-Lab/VLM2Vec-Full --runner pooling \
  --trust-remote-code \
  --chat-template examples/template_vlm2vec_phi3v.jinja

# 服务器启动后逐个运行
# 下载数据集
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --model TIGER-Lab/VLM2Vec-Full \
  --backend openai-embeddings-vlm2vec \
  --endpoint /v1/embeddings \
  --dataset-name sharegpt \
  --dataset-path <你的数据路径>/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
  --model TIGER-Lab/VLM2Vec-Full \
  --backend openai-embeddings-vlm2vec \
  --endpoint /v1/embeddings \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat
```

</details>

### 重排序器基准测试

对 vLLM 中重排序请求的性能进行基准测试。

<details class="admonition abstract" markdown="1">
<summary>展开</summary>

与使用 Completions API 或 Chat Completions API 的生成模型不同，你应该设置 `--backend vllm-rerank` 和 `--endpoint /v1/rerank` 来使用 Reranker API。

对于重排序，唯一支持的数据集是 `--dataset-name random-rerank`

启动服务器：

```bash
vllm serve BAAI/bge-reranker-v2-m3
```

运行基准测试：

```bash
vllm bench serve \
  --model BAAI/bge-reranker-v2-m3 \
  --backend vllm-rerank \
  --endpoint /v1/rerank \
  --dataset-name random-rerank \
  --tokenizer BAAI/bge-reranker-v2-m3 \
  --random-input-len 512 \
  --num-prompts 10 \
  --random-batch-size 5
```

对于重排序器模型，这将创建 `num_prompts / random_batch_size` 个请求，每个请求包含 `random_batch_size` 个"文档"，每个文档大约有 `random_input_len` 个 token。在上面的示例中，这将产生 2 个重排序请求，每个请求包含 5 个"文档"，每个文档大约有 512 个 token。

请注意，`/v1/rerank` 也支持嵌入模型。因此，如果你使用嵌入模型运行，还要设置 `--no_reranker`。因为在这种情况下，查询被服务器视为单独的提示，我们发送 `random_batch_size - 1` 个文档以考虑作为查询的额外提示。用于正确报告吞吐量数字的 token 统计也会相应调整。

</details>
