# Qwen2.5-Omni 离线推理示例

此文件夹提供了多个关于如何离线推理 Qwen2.5-Omni 的示例脚本。

## 仅 Thinker 模式

```bash
# 音频 + 图像 + 视频
python examples/generate/multimodal/qwen2_5_omni/only_thinker.py \
    -q mixed_modalities

# 从单个视频文件读取视觉和音频输入
python examples/generate/multimodal/qwen2_5_omni/only_thinker.py \
    -q use_audio_in_video

# 多个音频
python examples/generate/multimodal/qwen2_5_omni/only_thinker.py \
    -q multi_audios
```

此脚本将运行 Qwen2.5-Omni 的 thinker 部分，并生成文本响应。

您还可以在单一模态上测试 Qwen2.5-Omni：

```bash
# 处理音频输入
python examples/generate/multimodal/audio_language_offline.py \
    --model-type qwen2_5_omni

# 处理图像输入
python examples/generate/multimodal/vision_language_offline.py \
    --modality image \
    --model-type qwen2_5_omni

# 处理视频输入
python examples/generate/multimodal/vision_language_offline.py \
    --modality video \
    --model-type qwen2_5_omni
```
