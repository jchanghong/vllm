# 基于人类反馈的强化学习

基于人类反馈的强化学习（RLHF）是一种使用人类生成的偏好数据来微调语言模型的技术，旨在使模型输出与期望的行为对齐。vLLM 可用于生成 RLHF 所需的补全内容。

以下开源 RL 库使用 vLLM 进行快速 rollout（按字母顺序排列，非详尽列表）：

- [Cosmos-RL](https://github.com/nvidia-cosmos/cosmos-rl)
- [ms-swift](https://github.com/modelscope/ms-swift/tree/main)
- [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
- [Open Instruct](https://github.com/allenai/open-instruct)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [PipelineRL](https://github.com/ServiceNow/PipelineRL)
- [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)
- [SkyRL](https://github.com/NovaSky-AI/SkyRL)
- [TRL](https://github.com/huggingface/trl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [verl](https://github.com/volcengine/verl)

关于训练和推理之间的权重同步，请参阅[权重传输](weight_transfer/README.md)文档，其中涵盖了使用 [NCCL](weight_transfer/nccl.md)（多 GPU）和 [IPC](weight_transfer/ipc.md)（同 GPU）引擎的可插拔后端系统。

关于将生成和训练流水线化以提高 GPU 利用率和吞吐量，请参阅[异步强化学习](async_rl.md)指南，其中介绍了用于在运行中安全更新权重的暂停/恢复 API。

参见以下展示如何使用 vLLM 进行 GRPO 的笔记本：

- [Efficient Online Training with GRPO and vLLM in TRL](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)
- [Qwen-3 4B GRPO using Unsloth + vLLM](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)
