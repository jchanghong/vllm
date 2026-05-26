# XPU - Intel® GPUs

## 已验证的硬件

| 硬件 |
| -------- |
| [Intel® Arc™ Pro B 系列显卡](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html) |

## 推荐模型

### 纯文本语言模型

| 模型 | 架构 | BF16/FP16/动态 FP8 | Compressed_tensors FP8 | MXFP4 |
| -------------------------------------------------- | ------------------------------------------------ | --------------------- | ---------------------- | ----- |
| openai/gpt-oss-20b                                 | GPTForCausalLM                                   |                       |                        | ✅    |
| openai/gpt-oss-120b                                | GPTForCausalLM                                   |                       |                        | ✅    |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B           | LlamaForCausalLM                                 | ✅                    |                        |       |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B           | QwenForCausalLM                                  | ✅                    |                        |       |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B           | QwenForCausalLM                                  | ✅                    |                        |       |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B          | LlamaForCausalLM                                 | ✅                    |                        |       |
| Qwen/Qwen2.5-72B-Instruct                          | Qwen2ForCausalLM                                 | ✅                    |                        |       |
| Qwen/Qwen3-14B                                     | Qwen3ForCausalLM                                 | ✅                    |                        |       |
| Qwen/Qwen3-32B                                     | Qwen3ForCausalLM                                 | ✅                    |                        |       |
| Qwen/Qwen3-30B-A3B                                 | Qwen3ForCausalLM                                 | ✅                    |                        |       |
| Qwen/Qwen3-30B-A3B-GPTQ-Int4                       | Qwen3ForCausalLM                                 | ✅                    |                        |       |
| Qwen/Qwen3-coder-30B-A3B-Instruct                  | Qwen3ForCausalLM                                 | ✅                    |                        |       |
| Qwen/QwQ-32B                                       | QwenForCausalLM                                  | ✅                    |                        |       |
| deepseek-ai/DeepSeek-V2-Lite                       | DeepSeekForCausalLM                              | ✅                    |                        |       |
| meta-llama/Llama-3.1-8B-Instruct                   | LlamaForCausalLM                                 | ✅                    |                        |       |
| baichuan-inc/Baichuan2-13B-Chat                    | BaichuanForCausalLM                              | ✅                    |                        |       |
| THUDM/GLM-4-9B-chat                                | GLMForCausalLM                                   | ✅                    |                        |       |
| THUDM/CodeGeex4-All-9B                             | CodeGeexForCausalLM                              | ✅                    |                        |       |
| chuhac/TeleChat2-35B                               | LlamaForCausalLM (TeleChat2 基于 Llama 架构)       | ✅                    |                        |       |
| 01-ai/Yi1.5-34B-Chat                               | YiForCausalLM                                    | ✅                    |                        |       |
| THUDM/CodeGeex4-All-9B                             | CodeGeexForCausalLM                              | ✅                    |                        |       |
| deepseek-ai/DeepSeek-Coder-33B-base                | DeepSeekCoderForCausalLM                         | ✅                    |                        |       |
| baichuan-inc/Baichuan2-13B-Chat                    | BaichuanForCausalLM                              | ✅                    |                        |       |
| meta-llama/Llama-2-13b-chat-hf                     | LlamaForCausalLM                                 | ✅                    |                        |       |
| THUDM/CodeGeex4-All-9B                             | CodeGeexForCausalLM                              | ✅                    |                        |       |
| Qwen/Qwen1.5-14B-Chat                              | QwenForCausalLM                                  | ✅                    |                        |       |
| Qwen/Qwen1.5-32B-Chat                              | QwenForCausalLM                                  | ✅                    |                        |       |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic    | LlamaForCausalLM                                 |                       | ✅                     |       |

### 多模态语言模型

| 模型 | 架构 | BF16 | 动态 FP8 | MXFP4 |
| ---------------------------- | -------------------------------- | ---- | ----------- | ----- |
| OpenGVLab/InternVL3_5-8B     | InternVLForConditionalGeneration | ✅   | ✅          |       |
| OpenGVLab/InternVL3_5-14B    | InternVLForConditionalGeneration | ✅   | ✅          |       |
| OpenGVLab/InternVL3_5-38B    | InternVLForConditionalGeneration | ✅   | ✅          |       |
| Qwen/Qwen2-VL-7B-Instruct    | Qwen2VLForConditionalGeneration  | ✅   | ✅          |       |
| Qwen/Qwen2.5-VL-72B-Instruct | Qwen2VLForConditionalGeneration  | ✅   | ✅          |       |
| Qwen/Qwen2.5-VL-32B-Instruct | Qwen2VLForConditionalGeneration  | ✅   | ✅          |       |
| THUDM/GLM-4v-9B              | GLM4vForConditionalGeneration    | ✅   | ✅          |       |
| openbmb/MiniCPM-V-4          | MiniCPMVForConditionalGeneration | ✅   | ✅          |       |

### 嵌入和重排序语言模型

| 模型 | 架构 | BF16 | 动态 FP8 | MXFP4 |
| ----------------------- | ------------------------------ | ---- | ----------- | ----- |
| Qwen/Qwen3-Embedding-8B | Qwen3ForTextEmbedding          | ✅   | ✅          |       |
| Qwen/Qwen3-Reranker-8B  | Qwen3ForSequenceClassification | ✅   | ✅          |       |

✅ 已运行并优化。  
🟨 已运行且结果正确，但尚未优化至绿色级别。  
❌ 未通过精度测试或无法运行。  
