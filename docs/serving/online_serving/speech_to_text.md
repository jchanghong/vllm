# 语音转文本 API

## 转录 API

我们的转录 API 与 [OpenAI 的转录 API](https://platform.openai.com/docs/api-reference/audio/createTranscription) 兼容；
你可以使用[官方 OpenAI Python 客户端](https://github.com/openai/openai-python)与之交互。

!!! note
    要使用转录 API，请使用 `pip install vllm[audio]` 安装额外的音频依赖项。

代码示例：[examples/speech_to_text/openai/openai_transcription_client.py](../../../examples/speech_to_text/openai/openai_transcription_client.py)

注意：转录端点目前支持编码器-解码器多模态模型（例如 whisper）的束搜索，但由于处理编码器/解码器缓存的工作正在积极进行中，效率非常低。这是当前正在进行的优化重点，将在不久的将来得到妥善处理。

### API 强制限制

通过 `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` 环境变量设置 vLLM 将接受的最大音频文件大小（以 MB 为单位）。默认为 25 MB。

### 上传音频文件

转录 API 支持上传多种格式的音频文件，包括 FLAC、MP3、MP4、MPEG、MPGA、M4A、OGG、WAV 和 WEBM。

**使用 OpenAI Python 客户端：**

??? code

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    # 从磁盘上传音频文件
    with open("audio.mp3", "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="openai/whisper-large-v3-turbo",
            file=audio_file,
            language="en",
            response_format="verbose_json",
        )

    print(transcription.text)
    ```

**使用 curl 和 multipart/form-data：**

??? code

    ```bash
    curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
      -H "Authorization: Bearer token-abc123" \
      -F "file=@audio.mp3" \
      -F "model=openai/whisper-large-v3-turbo" \
      -F "language=en" \
      -F "response_format=verbose_json"
    ```

**支持的参数：**

- `file`：要转录的音频文件（必需）
- `model`：用于转录的模型（必需）
- `language`：语言代码（例如 "en"、"zh"）（可选）
- `prompt`：引导转录风格的可选文本（可选）
- `response_format`：响应的格式（"json"、"text"）（可选）
- `temperature`：介于 0 和 1 之间的采样温度（可选）

有关包括采样参数和 vLLM 扩展的完整支持参数列表，请参见[协议定义](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/protocol.py#L2182)。

**响应格式：**

对于 `verbose_json` 响应格式：

??? code

    ```json
    {
      "text": "Hello, this is a transcription of the audio file.",
      "language": "en",
      "duration": 5.42,
      "segments": [
        {
          "id": 0,
          "seek": 0,
          "start": 0.0,
          "end": 2.5,
          "text": "Hello, this is a transcription",
          "tokens": [50364, 938, 428, 307, 275, 28347],
          "temperature": 0.0,
          "avg_logprob": -0.245,
          "compression_ratio": 1.235,
          "no_speech_prob": 0.012
        }
      ]
    }
    ```
目前 "verbose_json" 响应格式不支持 no_speech_prob。

### 额外参数

支持以下[采样参数](../../api/README.md#inference-parameters)。

??? code

    ```python
    --8<-- "vllm/entrypoints/speech_to_text/transcription/protocol.py:transcription-sampling-params"
    ```

支持以下额外参数：

??? code

    ```python
    --8<-- "vllm/entrypoints/speech_to_text/transcription/protocol.py:transcription-extra-params"
    ```

## 翻译 API

我们的翻译 API 与 [OpenAI 的翻译 API](https://platform.openai.com/docs/api-reference/audio/createTranslation) 兼容；
你可以使用[官方 OpenAI Python 客户端](https://github.com/openai/openai-python)与之交互。
Whisper 模型可以将音频从 55 种非英语支持语言中的任意一种翻译成英语。
请注意，流行的 `openai/whisper-large-v3-turbo` 模型不支持翻译。

!!! note
    要使用翻译 API，请使用 `pip install vllm[audio]` 安装额外的音频依赖项。

代码示例：[examples/speech_to_text/openai/openai_translation_client.py](../../../examples/speech_to_text/openai/openai_translation_client.py)

### 额外参数

支持以下[采样参数](../../api/README.md#inference-parameters)。

```python
--8<-- "vllm/entrypoints/speech_to_text/translation/protocol.py:translation-sampling-params"
```

支持以下额外参数：

```python
--8<-- "vllm/entrypoints/speech_to_text/translation/protocol.py:translation-extra-params"
```

## 实时 API

实时 API 提供基于 WebSocket 的流式音频转录，允许在录制音频时进行实时语音转文本。

!!! note
    要使用实时 API，请使用 `uv pip install vllm[audio]` 安装额外的音频依赖项。

### 音频格式

音频必须以 16kHz 采样率、单声道的 base64 编码 PCM16 音频格式发送。

### 协议概述

1. 客户端连接到 `ws://host/v1/realtime`
2. 服务器发送 `session.created` 事件
3. 客户端可选地发送 `session.update` 与模型/参数
4. 客户端在准备好时发送 `input_audio_buffer.commit`
5. 客户端发送带有 base64 PCM16 块的 `input_audio_buffer.append` 事件
6. 服务器发送带有增量文本的 `transcription.delta` 事件
7. 服务器发送最终文本 + 使用情况的 `transcription.done` 事件
8. 从步骤 5 开始重复，处理下一段话语
9. 可选地，客户端可以发送带有 `final=True` 的 `input_audio_buffer.commit`
    以表示音频输入已完成。在流式传输音频文件时很有用

### 客户端 → 服务器事件

| 事件 | 描述 |
| ----- | ----------- |
| `input_audio_buffer.append` | 发送 base64 编码的音频块：`{"type": "input_audio_buffer.append", "audio": "<base64>"}` |
| `input_audio_buffer.commit` | 触发转录处理或结束：`{"type": "input_audio_buffer.commit", "final": bool}` |
| `session.update` | 配置会话：`{"type": "session.update", "model": "model-name"}` |

### 服务器 → 客户端事件

| 事件 | 描述 |
| ----- | ----------- |
| `session.created` | 连接已建立，包含会话 ID 和时间戳 |
| `transcription.delta` | 增量转录文本：`{"type": "transcription.delta", "delta": "text"}` |
| `transcription.done` | 最终转录及使用统计信息 |
| `error` | 错误通知，包含消息和可选的错误代码 |

#### 示例客户端

- [openai_realtime_client.py](https://github.com/vllm-project/vllm/tree/main/examples/speech_to_text/realtime/openai_realtime_client.py)——上传并转录音频文件
- [openai_realtime_microphone_client.py](https://github.com/vllm-project/vllm/tree/main/examples/speech_to_text/realtime/openai_realtime_microphone_client.py)——用于实时麦克风转录的 Gradio 演示
