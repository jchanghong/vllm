# 语音转文本（转录/翻译）支持

本文档将引导您完成通过实现 [SupportsTranscription][vllm.model_executor.models.interfaces.SupportsTranscription] 来为 vLLM 的转录和翻译 API 添加语音转文本（ASR）模型支持的步骤。
请参考[支持的模型](../../models/supported_models.md#transcription)获取进一步指导。

## 更新基础 vLLM 模型

假设您已经根据基础模型指南在 vLLM 中实现了您的模型。使用 [SupportsTranscription][vllm.model_executor.models.interfaces.SupportsTranscription] 接口扩展您的模型，并实现以下类属性和方法。

### `supported_languages` 和 `supports_transcription_only`

声明支持的语言和能力：

- `supported_languages` 映射在初始化时进行验证。
- 如果模型不应提供文本生成服务（例如 Whisper），请设置 `supports_transcription_only=True`。

??? code "supported_languages 和 supports_transcription_only"

    ```python
    from typing import ClassVar, Mapping, Literal
    import numpy as np
    import torch
    from torch import nn

    from vllm.config import ModelConfig, SpeechToTextConfig
    from vllm.inputs import PromptType
    from vllm.model_executor.models.interfaces import SupportsTranscription
    
    class YourASRModel(nn.Module, SupportsTranscription):
        # ISO 639-1 语言代码到语言名称的映射
        supported_languages: ClassVar[Mapping[str, str]] = {
            "en": "English",
            "it": "Italian",
            # ... 根据需要添加更多
        }
        
        # 如果您的模型仅支持音频条件生成
        #（不提供纯文本生成），请启用此标志。
        supports_transcription_only: ClassVar[bool] = True
    ```

通过 [get_speech_to_text_config][vllm.model_executor.models.interfaces.SupportsTranscription.get_speech_to_text_config] 提供 ASR 配置。

这是用于控制服务您的模型时 API 的通用行为：

??? code "get_speech_to_text_config()"

    ```python
    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_speech_to_text_config(
            cls,
            model_config: ModelConfig,
            task_type: Literal["transcribe", "translate"],
        ) -> SpeechToTextConfig:
            return SpeechToTextConfig(
                sample_rate=16_000,
                max_audio_clip_s=30,
                # 如果您的模型/处理器已经处理了分块，
                # 请设置为 None 以禁用服务端分块
                min_energy_split_window_size=None,
            )
    ```

请参阅[音频预处理和分块](#audio-preprocessing-and-chunking)了解每个字段的控制内容。

通过 [get_generation_prompt][vllm.model_executor.models.interfaces.SupportsTranscription.get_generation_prompt] 实现提示构建。服务器构建一个 [SpeechToTextParams][vllm.config.speech_to_text.SpeechToTextParams] 对象，该对象包含重采样后的波形、任务参数和请求特定的选项。您的模型接收这个单一对象并返回一个有效的 [PromptType][vllm.inputs.llm.PromptType]。有两种常见模式：

#### 带音频嵌入的多模态 LLM（例如 Voxtral, Gemma3n）

返回一个包含 `multi_modal_data`（音频）以及 `prompt` 字符串或 `prompt_token_ids` 的字典：

??? code "get_generation_prompt()"

    ```python
    from vllm.config.speech_to_text import SpeechToTextParams

    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_generation_prompt(
            cls,
            stt_params: SpeechToTextParams,
        ) -> PromptType:
            audio = stt_params.audio
            stt_config = stt_params.stt_config
            task_type = stt_params.task_type

            task_word = "Transcribe" if task_type == "transcribe" else "Translate"
            prompt = (
                "<start_of_turn>user\n"
                f"{task_word} this audio: <audio_soft_token>"
                "<end_of_turn>\n<start_of_turn>model\n"
            )

            return {
                "multi_modal_data": {"audio": (audio, stt_config.sample_rate)},
                "prompt": prompt,
            }
    ```

    有关多模态输入的进一步说明，请参考[多模态输入](../../features/multimodal_inputs.md)。

#### 编码器-解码器纯音频（例如 Whisper）

返回一个包含独立 `encoder_prompt` 和 `decoder_prompt` 条目的字典：

??? code "get_generation_prompt()"

    ```python
    from vllm.config.speech_to_text import SpeechToTextParams

    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_generation_prompt(
            cls,
            stt_params: SpeechToTextParams,
        ) -> PromptType:
            audio = stt_params.audio
            stt_config = stt_params.stt_config
            language = stt_params.language
            task_type = stt_params.task_type
            request_prompt = stt_params.request_prompt

            if language is None:
                raise ValueError("Language must be specified")

            prompt = {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {
                        "audio": (audio, stt_config.sample_rate),
                    },
                },
                "decoder_prompt": (
                    (f"<|prev|>{request_prompt}" if request_prompt else "")
                    + f"<|startoftranscript|><|{language}|>"
                    + f"<|{task_type}|><|notimestamps|>"
                ),
            }
            return cast(PromptType, prompt)
    ```

### `validate_language`（可选）

通过 [validate_language][vllm.model_executor.models.interfaces.SupportsTranscription.validate_language] 进行语言验证

如果您的模型需要语言并且您想要一个默认值，请重写此方法（参见 Whisper）：

??? code "validate_language()"

    ```python
    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None:
            logger.warning(
                "Defaulting to language='en'. If you wish to transcribe "
                "audio in a different language, pass the `language` field "
                "in the TranscriptionRequest."
            )
            language = "en"
        return super().validate_language(language)
    ```

### `get_num_audio_tokens`（可选）

通过 [get_num_audio_tokens][vllm.model_executor.models.interfaces.SupportsTranscription.get_num_audio_tokens] 实现流式的 token 计数

提供一个快速的持续时间到 token 的估算，以改善流式使用统计：

??? code "get_num_audio_tokens()"

    ```python
    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_num_audio_tokens(
            cls,
            audio_duration_s: float,
            stt_config: SpeechToTextConfig,
            model_config: ModelConfig,
        ) -> int | None:
            # 如果未知则返回 None；否则返回一个估算值。
            return int(audio_duration_s * stt_config.sample_rate // 320)  # 示例
    ```

## 音频预处理和分块

API 服务器在构建提示之前负责基本的音频 I/O 和可选的分块：

- 重采样：输入的音频使用 `AudioResampler` 重采样到 `SpeechToTextConfig.sample_rate`。
- 分块：如果 `SpeechToTextConfig.allow_audio_chunking` 为 True 且持续时间超过 `max_audio_clip_s`，服务器会将音频分割成重叠的块，并为每个块生成一个提示。重叠部分由 `overlap_chunk_second` 控制。
- 能量感知分割：当设置了 `min_energy_split_window_size` 时，服务器会寻找低能量区域以最小化在单词中间切割。

相关的服务器逻辑：

??? code "_preprocess_speech_to_text()"

    ```python
    # vllm/entrypoints/openai/speech_to_text.py
    async def _preprocess_speech_to_text(...):
        language = self.model_cls.validate_language(request.language)
        ...
        y, sr = load_audio(bytes_, sr=self.asr_config.sample_rate)
        duration = get_audio_duration(y=y, sr=sr)
        do_split_audio = (self.asr_config.allow_audio_chunking
                        and duration > self.asr_config.max_audio_clip_s)
        chunks = [y] if not do_split_audio else self._split_audio(y, int(sr))
        prompts = []
        for chunk in chunks:
            stt_params = request.build_stt_params(
                audio=chunk,
                stt_config=self.asr_config,
                model_config=self.model_config,
                task_type=self.task_type,
            )
            prompt = self.model_cls.get_generation_prompt(stt_params)
            prompts.append(prompt)
        return prompts, duration
    ```

## 自动暴露任务

如果您的模型实现了该接口，vLLM 会自动声明转录支持：

```python
if supports_transcription(model):
    if model.supports_transcription_only:
        return ["transcription"]
    supported_tasks.append("transcription")
```

启用后，服务器会初始化转录和翻译处理器：

```python
state.openai_serving_transcription = OpenAIServingTranscription(...) if "transcription" in supported_tasks else None
state.openai_serving_translation = OpenAIServingTranslation(...) if "transcription" in supported_tasks else None
```

除了通过模型注册表提供您的模型类并实现 `SupportsTranscription` 之外，无需额外注册。

## 树内示例

- Whisper 编码器-解码器（纯音频）：[vllm/model_executor/models/whisper.py](../../../vllm/model_executor/models/whisper.py)
- Voxtral 仅解码器（音频嵌入 + LLM）：[vllm/model_executor/models/voxtral.py](../../../vllm/model_executor/models/voxtral.py)。确保已安装 `mistral-common[audio]`。
- Gemma3n 仅解码器，带固定指令提示：[vllm/model_executor/models/gemma3n_mm.py](../../../vllm/model_executor/models/gemma3n_mm.py)
- Qwen3-Omni 多模态，带音频嵌入：[vllm/model_executor/models/qwen3_omni_moe_thinker.py](../../../vllm/model_executor/models/qwen3_omni_moe_thinker.py)

## 使用 API 测试

一旦您的模型实现了 `SupportsTranscription`，您可以测试端点（API 模仿 OpenAI）：

- 转录（ASR）：

    ```bash
    curl -s -X POST \
      -H "Authorization: Bearer $VLLM_API_KEY" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@/path/to/audio.wav" \
      -F "model=$MODEL_ID" \
      http://localhost:8000/v1/audio/transcriptions
    ```

- 翻译（源语言 → 英文，除非另有支持）：

    ```bash
    curl -s -X POST \
      -H "Authorization: Bearer $VLLM_API_KEY" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@/path/to/audio.wav" \
      -F "model=$MODEL_ID" \
      http://localhost:8000/v1/audio/translations
    ```

或在 [examples/speech_to_text](../../../examples/speech_to_text) 中查看更多示例。

!!! note
    - 如果您的模型在内部处理分块（例如，通过其处理器或编码器），请在返回的 `SpeechToTextConfig` 中设置 `min_energy_split_window_size=None` 以禁用服务端分块。
    - 实现 `get_num_audio_tokens` 可以在无需额外前向传播的情况下提高流式使用指标（`prompt_tokens`）的准确性。
    - 对于多语言行为，请保持 `supported_languages` 与实际模型能力一致。
