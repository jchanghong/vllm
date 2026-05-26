# 多模态输入

本页面介绍如何在 vLLM 中将多模态输入传递给[多模态模型](../models/supported_models.md#list-of-multimodal-language-models)。

!!! note
    我们正在积极探索多模态支持。请参阅[此 RFC](https://github.com/vllm-project/vllm/issues/4194)了解即将到来的更改，
    如有任何反馈或功能请求，请[在 GitHub 上提交 issue](https://github.com/vllm-project/vllm/issues/new/choose)。

!!! tip
    在服务多模态模型时，请考虑设置 `--allowed-media-domains` 来限制 vLLM 可以访问的域名，以防止它访问可能容易受到服务端请求伪造（SSRF）攻击的任意端点。您可以为该参数提供一个域名列表。例如：`--allowed-media-domains upload.wikimedia.org github.com www.bogotobogo.com`

    另外，请考虑设置 `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0` 以防止 HTTP 重定向被用于绕过域名限制。

    如果您在容器化环境中运行 vLLM，这种限制尤其重要，因为 vLLM pods 可能不受限制地访问内部网络。

## 离线推理

要输入多模态数据，请遵循 [vllm.inputs.PromptType][] 中的以下模式：

- `prompt`：提示词应遵循 HuggingFace 上记录的格式。
- `multi_modal_data`：这是一个字典，遵循 [vllm.inputs.MultiModalDataDict][] 中定义的模式。

### 图像输入

您可以将单张图像传递给多模态字典的 `'image'` 字段，如下例所示：

??? code

    ```python
    from vllm import LLM

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    # 参考 HuggingFace 仓库了解正确的格式
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    # 使用 PIL.Image 加载图像
    image = PIL.Image.open(...)

    # 单提示词推理
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

    # 批量推理
    image_1 = PIL.Image.open(...)
    image_2 = PIL.Image.open(...)
    outputs = llm.generate(
        [
            {
                "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
                "multi_modal_data": {"image": image_1},
            },
            {
                "prompt": "USER: <image>\nWhat's the color of this image?\nASSISTANT:",
                "multi_modal_data": {"image": image_2},
            }
        ]
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

完整示例：[examples/generate/multimodal/vision_language_offline.py](../../examples/generate/multimodal/vision_language_offline.py)

要在同一文本提示中替换多张图像，您可以传入一个图像列表：

??? code

    ```python
    from vllm import LLM

    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,  # 加载 Phi-3.5-vision 所需
        max_model_len=4096,  # 否则可能无法放入较小的 GPU
        limit_mm_per_prompt={"image": 2},  # 最大可接受数量
    )

    # 参考 HuggingFace 仓库了解正确的格式
    prompt = "<|user|>\n<|image_1|>\n<|image_2|>\nWhat is the content of each image?<|end|>\n<|assistant|>\n"

    # 使用 PIL.Image 加载图像
    image1 = PIL.Image.open(...)
    image2 = PIL.Image.open(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": [image1, image2]},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

完整示例：[examples/generate/multimodal/vision_language_multi_image_offline.py](../../examples/generate/multimodal/vision_language_multi_image_offline.py)

如果使用 [LLM.chat](../models/generative_models.md#llmchat) 方法，您可以直接在消息内容中使用多种格式传递图像：图像 URL、PIL Image 对象或预计算的嵌入：

??? code

    ```python
    from vllm import LLM
    from vllm.assets.image import ImageAsset

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")
    image_url = "https://picsum.photos/id/32/512/512"
    image_pil = ImageAsset('cherry_blossom').pil_image
    image_embeds = torch.load(...)

    conversation = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {
                    "type": "image_pil",
                    "image_pil": image_pil,
                },
                {
                    "type": "image_embeds",
                    "image_embeds": image_embeds,
                },
                {
                    "type": "text",
                    "text": "What's in these images?",
                },
            ],
        },
    ]

    # 执行推理并输出日志。
    outputs = llm.chat(conversation)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

多图像输入可以扩展用于视频字幕。我们以 [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) 为例，因为它支持视频：

??? code

    ```python
    from vllm import LLM

    # 指定每个视频的最大帧数为 4。此值可更改。
    llm = LLM("Qwen/Qwen2-VL-2B-Instruct", limit_mm_per_prompt={"image": 4})

    # 创建请求负载。
    video_frames = ... # 加载您的视频，确保其帧数不超过前面指定的数量。
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe this set of frames. Consider the frames to be a part of the same video.",
            },
        ],
    }
    for i in range(len(video_frames)):
        base64_image = encode_image(video_frames[i]) # base64 编码。
        new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        message["content"].append(new_image)

    # 执行推理并输出日志。
    outputs = llm.chat([message])

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

#### 自定义 RGBA 背景颜色

当加载 RGBA 图像（带透明度的图像）时，vLLM 会将其转换为 RGB 格式。默认情况下，透明像素将以白色背景替换。您可以使用 `media_io_kwargs` 中的 `rgba_background_color` 参数自定义此背景颜色。

??? code

    ```python
    from vllm import LLM

    # 默认白色背景（无需配置）
    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    # 深色主题的自定义黑色背景
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        media_io_kwargs={"image": {"rgba_background_color": [0, 0, 0]}},
    )

    # 自定义品牌颜色背景（例如蓝色）
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        media_io_kwargs={"image": {"rgba_background_color": [0, 0, 255]}},
    )
    ```

!!! note
    - `rgba_background_color` 接受 RGB 值作为列表 `[R, G, B]` 或元组 `(R, G, B)`，其中每个值的范围为 0-255
    - 此设置仅影响带透明度的 RGBA 图像；RGB 图像不受影响
    - 如果未指定，默认使用白色背景 `(255, 255, 255)` 以保持向后兼容性

#### Moondream3 提示词模板 { #moondream3-prompt-recipes }

`Moondream3ForCausalLM` 支持两种特定任务的提示词格式：

- `query`：询问关于图像的问题。
- `caption`：为图像生成字幕。

```python
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

llm = LLM(
    model="moondream/moondream3-preview",
    tokenizer="moondream/starmie-v1",
    trust_remote_code=True,
    max_model_len=2048,
    limit_mm_per_prompt={"image": 1},
)

image = ImageAsset("stop_sign").pil_image


def make_query_prompt(question: str) -> str:
    return (
        "<|endoftext|><image><|md_reserved_0|>query<|md_reserved_1|>"
        f"{question}<|md_reserved_2|>"
    )


def make_caption_prompt(length: str = "normal") -> str:
    return (
        "<|endoftext|><image><|md_reserved_0|>"
        f"describe<|md_reserved_1|>{length}<|md_reserved_2|>"
    )


query_out = llm.generate(
    {
        "prompt": make_query_prompt("What is shown in this image?"),
        "multi_modal_data": {"image": image},
    },
    SamplingParams(max_tokens=64, temperature=0),
)[0].outputs[0].text

caption_out = llm.generate(
    {
        "prompt": make_caption_prompt(),
        "multi_modal_data": {"image": image},
    },
    SamplingParams(max_tokens=100, temperature=0),
)[0].outputs[0].text

print("query:", query_out)
print("caption:", caption_out)
```

!!! note
    原生 Moondream3 模型还具有 `detect` 和 `point` 技能。这些
    需要自定义坐标解码，本 vLLM 实现未提供。

### 视频输入

您可以将 NumPy 数组列表直接传递给多模态字典的 `'video'` 字段，
而不是使用多图像输入。

除了 NumPy 数组，您也可以传递 `'torch.Tensor'` 实例，如下面的 Qwen2.5-VL 示例所示：

??? code

    ```python
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    video_path = "https://content.pexels.com/videos/free-videos.mp4"

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 1},
    )

    sampling_params = SamplingParams(max_tokens=1024)

    video_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this video."},
                {
                    "type": "video",
                    "video": video_path,
                    "total_pixels": 20480 * 28 * 28,
                    "min_pixels": 16 * 28 * 28,
                },
            ]
        },
    ]

    messages = video_messages
    processor = AutoProcessor.from_pretrained(model_path)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)
    mm_data = {}
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

    !!! note
        'process_vision_info' 仅适用于 Qwen2.5-VL 及类似模型。

完整示例：[examples/generate/multimodal/vision_language_offline.py](../../examples/generate/multimodal/vision_language_offline.py)

### 音频输入

您可以将元组 `(array, sampling_rate)` 传递给多模态字典的 `'audio'` 字段。

完整示例：[examples/generate/multimodal/audio_language_offline.py](../../examples/generate/multimodal/audio_language_offline.py)

#### 长音频分块转录

像 Whisper 这样的语音转文本模型有最大音频长度限制（通常为 30 秒）。对于更长的音频文件，vLLM 提供了一种实用工具，可以在静音处智能地将音频分割成块，以最大限度地减少对语音的切割。

```python
from vllm import LLM, SamplingParams
from vllm.multimodal.audio import split_audio
from vllm.multimodal.media.audio import load_audio

# 加载长音频文件
audio, sr = load_audio("long_audio.wav", sr=16000)

# 在低能量（静音）区域分割成块
chunks = split_audio(
    audio_data=audio,
    sample_rate=sr,
    max_clip_duration_s=30.0,      # 最大块长度（秒）
    overlap_duration_s=1.0,         # 寻找安静分割点的搜索窗口
    min_energy_window_size=1600,    # 能量计算的窗口大小（16kHz 下约 100ms）
)

# 初始化 Whisper 模型
llm = LLM(model="openai/whisper-large-v3-turbo")
sampling_params = SamplingParams(temperature=0, max_tokens=256)

# 转录每个块
transcriptions = []
for chunk in chunks:
    outputs = llm.generate({
        "prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        "multi_modal_data": {"audio": (chunk, sr)},
    }, sampling_params)
    transcriptions.append(outputs[0].outputs[0].text)

# 合并结果
full_transcription = " ".join(transcriptions)
```

`split_audio` 函数：

- 在静音点分割音频，避免切割语音
- 使用 RMS 能量在重叠窗口内查找低振幅区域
- 保留所有音频样本（无数据丢失）
- 支持任何采样率

#### 自动音频通道归一化

vLLM 会自动为需要特定音频格式的模型归一化音频通道。当使用 `torchaudio` 等库加载音频时，立体声文件返回形状 `[channels, time]`，但许多音频模型（特别是基于 Whisper 的模型）期望单声道音频，形状为 `[time]`。

**支持自动单声道转换的模型：**

- **Whisper** 及所有基于 Whisper 的模型
- **Qwen2-Audio**
- **Qwen2.5-Omni** / **Qwen3-Omni**（继承自 Qwen2.5-Omni）
- **Ultravox**

对于这些模型，vLLM 会自动：

1. 通过特征提取器检测模型是否需要单声道音频
2. 使用声道平均将多声道音频转换为单声道
3. 处理 `(channels, time)` 格式（torchaudio）和 `(time, channels)` 格式（soundfile）

**立体声音频示例：**

```python
import torchaudio
from vllm import LLM

# 加载立体声音频文件 - 返回 (channels, time) 形状
audio, sr = torchaudio.load("stereo_audio.wav")
print(f"原始形状: {audio.shape}")  # 例如 torch.Size([2, 16000])

# vLLM 自动为基于 Whisper 的模型转换为单声道
llm = LLM(model="openai/whisper-large-v3")

outputs = llm.generate({
    "prompt": "",
    "multi_modal_data": {"audio": (audio.numpy(), sr)},
})
```

无需手动转换 - vLLM 根据模型需求自动处理通道归一化。

### 嵌入输入

要将属于某种数据类型（如图像、视频或音频）的预计算嵌入直接输入到语言模型，
请将形状为 `(..., hidden_size of LM)` 的张量传递给多模态字典的相应字段。
具体形状取决于所使用的模型。

您必须通过 `enable_mm_embeds=True` 启用此功能。

!!! warning
    如果传入的嵌入形状不正确，vLLM 引擎可能崩溃。
    仅对受信任的用户启用此标志！

#### 图像嵌入

??? code

    ```python
    from vllm import LLM

    # 使用图像嵌入进行推理
    llm = LLM(model="llava-hf/llava-1.5-7b-hf", enable_mm_embeds=True)

    # 参考 HuggingFace 仓库了解正确的格式
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    # 对于大多数模型，`image_embeds` 的形状为：(num_images, image_feature_size, hidden_size)
    image_embeds = torch.load(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image_embeds},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

    # 需要额外字段的模型的额外示例
    llm = LLM(
        "Qwen/Qwen2-VL-2B-Instruct",
        limit_mm_per_prompt={"image": 4},
        enable_mm_embeds=True,
    )
    mm_data = {
        "image": {
            # 形状：(total_feature_size, hidden_size)
            # total_feature_size = sum(image_feature_size for image in images)
            "image_embeds": torch.load(...),
            # 形状：(num_images, 3)
            # image_grid_thw 用于计算位置编码。
            "image_grid_thw": torch.load(...),
        }
    }

    llm = LLM(
        "openbmb/MiniCPM-V-2_6",
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 4},
        enable_mm_embeds=True,
    )
    mm_data = {
        "image": {
            # 形状：(num_images, num_slices, hidden_size)
            # num_slices 可能因每个图像而异
            "image_embeds": [torch.load(...) for image in images],  
            # 形状：(num_images, 2)
            # image_sizes 用于计算切片图像的细节。
            "image_sizes": [image.size for image in images],
        }
    }
    ```

对于 Qwen3-VL，`image_embeds` 应同时包含基础图像嵌入和 deepstack 特征。

#### 音频嵌入输入

您可以像图像嵌入一样传递预计算的音频嵌入：

??? code

    ```python
    from vllm import LLM
    import torch

    # 启用音频嵌入支持
    llm = LLM(model="fixie-ai/ultravox-v0_5-llama-3_2-1b", enable_mm_embeds=True)

    # 参考 HuggingFace 仓库了解正确的格式
    prompt = "USER: <audio>\nWhat is in this audio?\nASSISTANT:"

    # 加载预计算的音频嵌入，通常形状为：
    # (num_audios, audio_feature_size, hidden_size of LM)
    audio_embeds = torch.load(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"audio": audio_embeds},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

### 缓存输入

使用多模态输入时，vLLM 通常通过内容对每个媒体项进行哈希以实现跨请求缓存。您可以选择传递 `multi_modal_uuids` 来为每个项提供自己的稳定 ID，以便缓存无需重新哈希原始内容即可跨请求重用工作。

??? code

    ```python
    from vllm import LLM
    from PIL import Image

    # Qwen2.5-VL 双图像示例
    llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct")

    prompt = "USER: <image><image>\nDescribe the differences.\nASSISTANT:"
    img_a = Image.open("/path/to/a.jpg")
    img_b = Image.open("/path/to/b.jpg")

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": [img_a, img_b]},
        # 为缓存提供稳定 ID。
        # 要求（由此示例满足）：
        #  - 包含 multi_modal_data 中的每种模态。
        #  - 对于列表，提供相同数量的条目。
        #  - 使用 None 为该条目回退到内容哈希。
        "multi_modal_uuids": {"image": ["sku-1234-a", None]},
    })

    for o in outputs:
        print(o.outputs[0].text)
    ```

使用 UUID，如果预期相应项目能命中缓存，您还可以完全跳过发送媒体数据。请注意，如果跳过的媒体没有对应的 UUID，或者 UUID 未能命中缓存，请求将失败。

??? code

    ```python
    from vllm import LLM
    from PIL import Image

    # Qwen2.5-VL 双图像示例
    llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct")

    prompt = "USER: <image><image>\nDescribe the differences.\nASSISTANT:"
    img_b = Image.open("/path/to/b.jpg")

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": [None, img_b]},
        # 由于 img_a 预期被缓存，我们可以完全跳过发送实际图像。
        "multi_modal_uuids": {"image": ["sku-1234-a", None]},
    })

    for o in outputs:
        print(o.outputs[0].text)
    ```

!!! warning
    如果多模态处理器缓存和前缀缓存都被禁用，用户提供的 `multi_modal_uuids` 将被忽略。

## 在线服务

我们的 OpenAI 兼容服务器通过 [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) 接受多模态数据。媒体输入还支持用户可选的 UUID，用于唯一标识每个媒体，从而实现跨请求的媒体结果缓存。

!!! important
    使用 Chat Completions API **必须**提供对话模板。
    对于 HF 格式的模型，默认对话模板定义在 `chat_template.json` 或 `tokenizer_config.json` 中。

    如果没有可用的默认对话模板，我们将首先在 [vllm/transformers_utils/chat_templates/registry.py](../../vllm/transformers_utils/chat_templates/registry.py) 中查找内置的回退模板。
    如果没有可用的回退，则会引发错误，您必须通过 `--chat-template` 参数手动提供对话模板。

    对于某些模型，我们在 [examples](../../examples) 中提供了替代的对话模板。
    例如，VLM2Vec 使用 [examples/pooling/embed/template/vlm2vec_phi3v.jinja](../../examples/pooling/embed/template/vlm2vec_phi3v.jinja)，这与 Phi-3-Vision 的默认模板不同。

### 图像输入

图像输入根据 [OpenAI Vision API](https://platform.openai.com/docs/guides/vision) 提供支持。
以下是一个使用 Phi-3.5-Vision 的简单示例。

首先，启动 OpenAI 兼容服务器：

```bash
vllm serve microsoft/Phi-3.5-vision-instruct --runner generate \
  --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt.image 2
```

然后，您可以按如下方式使用 OpenAI 客户端：

??? code

    ```python
    import os
    from openai import OpenAI

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # 单图像输入推理

    # 用于测试远程图像处理的公共图像 URL
    image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    # 使用远程图像创建聊天补全
    chat_response = client.chat.completions.create(
        model="microsoft/Phi-3.5-vision-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    # 注意：不需要在提示中包含图像标记 `<image>` 的格式
                    # 因为 API 服务器会自动处理提示。
                    {
                        "type": "text",
                        "text": "What's in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "uuid": image_url,  # 可选
                    },
                ],
            }
        ],
    )
    print("Chat completion output:", chat_response.choices[0].message.content)

    # 本地图像文件路径（请更新为指向实际的图像文件）
    image_file = "/path/to/image.jpg"

    # 使用本地图像文件创建聊天补全
    # 使用 --allowed-local-media-path 参数启动 API 服务器/引擎。
    if os.path.exists(image_file):
        chat_completion_from_local_image_url = client.chat.completions.create(
            model="microsoft/Phi-3.5-vision-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"file://{image_file}"},
                        },
                    ],
                }
            ],
        )
        result = chat_completion_from_local_image_url.choices[0].message.content
        print("来自本地图像文件的聊天补全输出:\n", result)
    else:
        print(f"本地图像文件 {image_file} 未找到，跳过本地文件测试。")

    # 多图像输入推理
    image_url_duck = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg"
    image_url_lion = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/lion.jpg"

    chat_response = client.chat.completions.create(
        model="microsoft/Phi-3.5-vision-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What are the animals in these images?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_duck},
                        "uuid": image_url_duck,  # 可选
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_lion},
                        "uuid": image_url_lion,  # 可选
                    },
                ],
            }
        ],
    )
    print("Chat completion output:", chat_response.choices[0].message.content)
    ```

完整示例：[examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py](../../examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py)

!!! tip
    vLLM 也支持从本地文件路径加载：您可以在启动 API 服务器/引擎时通过 `--allowed-local-media-path` 指定允许的本地媒体路径，
    并在 API 请求中将文件路径作为 `url` 传递。

!!! tip
    无需在 API 请求的文本内容中放置图像占位符——它们已由图像内容表示。
    实际上，您可以通过交错排列文本和图像内容，在文本中间放置图像占位符。

!!! note
    默认情况下，通过 HTTP URL 获取图像的超时时间为 `5` 秒。
    您可以通过设置环境变量来覆盖此值：

    ```bash
    export VLLM_IMAGE_FETCH_TIMEOUT=<timeout>
    ```

### 视频输入

您可以通过 `video_url` 传递视频文件，而不是 `image_url`。以下是一个使用 [LLaVA-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) 的简单示例。

首先，启动 OpenAI 兼容服务器：

```bash
vllm serve llava-hf/llava-onevision-qwen2-0.5b-ov-hf --runner generate --max-model-len 8192
```

然后，您可以按如下方式使用 OpenAI 客户端：

??? code

    ```python
    from openai import OpenAI

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"

    ## 在负载中使用视频 URL
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this video?",
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                        "uuid": video_url,  # 可选
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("来自图像 URL 的聊天补全输出:", result)
    ```

完整示例：[examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py](../../examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py)

!!! note
    默认情况下，通过 HTTP URL 获取视频的超时时间为 `30` 秒。
    您可以通过设置环境变量来覆盖此值：

    ```bash
    export VLLM_VIDEO_FETCH_TIMEOUT=<timeout>
    ```

#### 视频帧恢复

为了提高处理可能损坏或截断的视频文件的鲁棒性，vLLM 支持使用动态窗口前向扫描方法进行可选的帧恢复。启用后，如果在顺序读取过程中目标帧加载失败，下一个成功获取的帧（在下一个目标帧之前）将用于替代。

要启用视频帧恢复，请通过 `--media-io-kwargs` 传递 `frame_recovery` 参数：

```bash
# 示例：启用帧恢复
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --media-io-kwargs '{"video": {"frame_recovery": true}}'
```

**参数：**

- `frame_recovery`：布尔标志，用于启用前向扫描恢复。当为 `true` 时，失败的帧将使用动态窗口内下一个可用帧（直到下一个目标帧）进行恢复。默认值为 `false`。

**工作原理：**

1. 系统按顺序读取帧
2. 如果目标帧获取失败，则标记为"失败"
3. 下一个成功获取的帧（在到达下一个目标之前）用于恢复失败的帧
4. 这种方法同时处理视频中间损坏和视频末尾截断

使用 OpenCV 后端时，适用于 MP4 等常见视频格式。

#### 使用 `media_io_kwargs` 的预提取帧序列

当您在客户端提取视频帧并将其作为 `video/jpeg`（base64 连接的 JPEG 帧）发送时，您可以通过在请求中使用 `media_io_kwargs` 来保留原始视频元数据。这通过保留在客户端提取帧期间可能丢失的时间信息，实现更准确的视频理解。

**支持的参数：**

| 参数 | 类型 | 描述 |
| --------- | ---- | ----------- |
| `fps` | float | 原始视频的帧率 |
| `frames_indices` | list[int] | 实际采样帧的索引 |
| `total_num_frames` | int | 原始视频的总帧数 |
| `duration` | float | 原始视频的持续时间（秒） |
| `do_sample_frames` | bool | 是否执行帧采样 |

??? code

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    # 客户端帧提取
    frames = extract_frames(video_path, num_frames=32)
    frames_b64 = ",".join([encode_image(f) for f in frames])
    video_url = f"data:video/jpeg;base64,{frames_b64}"

    # 通过 media_io_kwargs 传递视频元数据
    response = client.chat.completions.create(
        model="your-multimodal-model",
        messages=[{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": "Describe what happens in this video."}
            ]
        }],
        extra_body={
            "media_io_kwargs": {
                "video": {
                    "fps": 30.0,
                    "frames_indices": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                       100, 110, 120, 130, 140, 150, 160, 170,
                                       180, 190, 200, 210, 220, 230, 240, 250,
                                       260, 270, 280, 290, 300, 310],
                    "total_num_frames": 900,
                    "duration": 30.0,
                }
            }
        },
    )

    print(response.choices[0].message.content)
    ```

**为什么使用 `media_io_kwargs`？**

在客户端提取帧时，服务器会丢失关于原始视频的重要上下文：

- **时间信息**：哪些帧被采样及其在原始时间线中的位置
- **视频时长**：原始视频持续了多长时间
- **帧率**：原始播放速度

通过传递这些元数据，模型可以更好地理解采样帧的时间分布，以及是否可能跳过了重要时刻。

#### 自定义 RGBA 背景颜色

要为 RGBA 图像使用自定义背景颜色，请通过 `--media-io-kwargs` 传递 `rgba_background_color` 参数：

```bash
# 示例：深色主题的黑色背景
vllm serve llava-hf/llava-1.5-7b-hf \
  --media-io-kwargs '{"image": {"rgba_background_color": [0, 0, 0]}}'

# 示例：自定义灰色背景
vllm serve llava-hf/llava-1.5-7b-hf \
  --media-io-kwargs '{"image": {"rgba_background_color": [128, 128, 128]}}'
```

### 音频输入

音频输入根据 [OpenAI Audio API](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in) 提供支持。
以下是一个使用 Ultravox-v0.5-1B 的简单示例。

首先，启动 OpenAI 兼容服务器：

```bash
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b
```

然后，您可以按如下方式使用 OpenAI 客户端：

??? code

    ```python
    import base64
    import requests
    from openai import OpenAI
    from vllm.assets.audio import AudioAsset

    def encode_base64_content_from_url(content_url: str) -> str:
        """将从远程 URL 获取的内容编码为 base64 格式。"""

        with requests.get(content_url) as response:
            response.raise_for_status()
            result = base64.b64encode(response.content).decode('utf-8')

        return result

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # 支持 soundfile/PyAV 的任何格式
    audio_url = AudioAsset("winning_call").url
    audio_base64 = encode_base64_content_from_url(audio_url)

    chat_completion_from_base64 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this audio?",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "wav",
                        },
                        "uuid": audio_url,  # 可选
                    },
                ],
            },
        ],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_base64.choices[0].message.content
    print("来自输入音频的聊天补全输出:", result)
    ```

或者，您可以传递 `audio_url`，它是图像输入中 `image_url` 的音频对应项：

??? code

    ```python
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this audio?",
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_url},
                        "uuid": audio_url,  # 可选
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("来自音频 URL 的聊天补全输出:", result)
    ```

完整示例：[examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py](../../examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py)

!!! note
    默认情况下，通过 HTTP URL 获取音频的超时时间为 `10` 秒。
    您可以通过设置环境变量来覆盖此值：

    ```bash
    export VLLM_AUDIO_FETCH_TIMEOUT=<timeout>
    ```

### 嵌入输入

要将属于某种数据类型（如图像、视频或音频）的预计算嵌入直接输入到语言模型，
请将形状为 `(..., hidden_size of LM)` 的张量传递給多模态字典的相应字段，每个项目一个。

!!! important
    与离线推理不同，每个项目的嵌入必须单独传递，
    以便对话模板正确应用占位符标记。

您必须通过 `vllm serve` 中的 `--enable-mm-embeds` 标志启用此功能。

!!! warning
    如果传入的嵌入形状不正确，vLLM 引擎可能崩溃。
    仅对受信任的用户启用此标志！

#### 图像嵌入输入

对于图像嵌入，您可以将 base64 编码的张量传递给 `image_embeds` 字段。
以下示例演示如何将图像嵌入传递给 OpenAI 服务器：

??? code

    ```python
    from vllm.utils.serial_utils import tensor2base64

    client = OpenAI(
        # 默认为 os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # 基本用法 - 这等同于离线推理的 LLaVA 示例
    model = "llava-hf/llava-1.5-7b-hf"
    embeds = {
        "type": "image_embeds",
        "image_embeds": tensor2base64(torch.load(...)),  # 形状：(image_feature_size, hidden_size)
        "uuid": image_url,  # 可选
    }


    # 需要额外字段的模型的额外示例
    model = "Qwen/Qwen2-VL-2B-Instruct"
    embeds = {
        "type": "image_embeds",
        "image_embeds": {
            "image_embeds": tensor2base64(torch.load(...)),  # 形状：(image_feature_size, hidden_size)
            "image_grid_thw": tensor2base64(torch.load(...)),  # 形状：(3,)
        },
        "uuid": image_url,  # 可选
    }

    model = "openbmb/MiniCPM-V-2_6"
    embeds = {
        "type": "image_embeds",
        "image_embeds": {
            "image_embeds": tensor2base64(torch.load(...)),  # 形状：(num_slices, hidden_size)
            "image_sizes": tensor2base64(torch.load(...)),  # 形状：(2,)
        },
        "uuid": image_url,  # 可选
    }

    # 单图像输入
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?",
                    },
                    embeds,
                ],
            },
        ],
        model=model,
    )

    # 多图像输入
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?",
                    },
                    embeds,
                    embeds,
                ],
            },
        ],
        model=model,
    )

    # 多图像输入（交错）
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": [
                    embeds,
                    {
                        "type": "text",
                        "text": "What's in this image?",
                    },
                    embeds,
                ],
            },
        ],
        model=model,
    )
    ```

### 缓存输入

与离线推理一样，如果您预期使用所提供的 UUID 命中缓存，则可以跳过发送媒体。您可以通过如下方式发送媒体：

??? code

    ```python
        # 图像/视频/音频 URL：
        {
            "type": "image_url",
            "image_url": None,
            "uuid": image_uuid,
        },

        # image_embeds
        {
            "type": "image_embeds",
            "image_embeds": None,
            "uuid": image_uuid,
        },

        # input_audio:
        {
            "type": "input_audio",
            "input_audio": None,
            "uuid": audio_uuid,
        },

        # PIL Image:
        {
            "type": "image_pil",
            "image_pil": None,
            "uuid": image_uuid,
        },

    ```
