# 检索增强生成

[检索增强生成 (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) 是一种技术，使生成式人工智能 (Gen AI) 模型能够检索并整合新信息。它修改了与大语言模型 (LLM) 的交互方式，使模型在响应查询时能够参考指定的文档集合，利用这些信息补充其预先训练数据中的信息。这使得 LLM 能够使用领域特定和/或更新的信息。用例包括为聊天机器人提供内部公司数据的访问权限，或基于权威来源生成响应。

以下是相关集成：

- vLLM + [langchain](https://github.com/langchain-ai/langchain) + [milvus](https://github.com/milvus-io/milvus)
- vLLM + [llamaindex](https://github.com/run-llama/llama_index) + [milvus](https://github.com/milvus-io/milvus)

## vLLM + langchain

### 前提条件

设置 vLLM 和 langchain 环境：

```bash
pip install -U vllm \
            langchain_milvus langchain_openai \
            langchain_community beautifulsoup4 \
            langchain-text-splitters
```

### 部署

1. 启动 vLLM 服务器，使用受支持的嵌入模型，例如：

    ```bash
    # 启动嵌入服务（端口 8000）
    vllm serve ssmits/Qwen2-7B-Instruct-embed-base
    ```

1. 启动 vLLM 服务器，使用受支持的聊天补全模型，例如：

    ```bash
    # 启动聊天服务（端口 8001）
    vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001
    ```

1. 使用脚本：[examples/applications/rag/retrieval_augmented_generation_with_langchain.py](../../../examples/applications/rag/retrieval_augmented_generation_with_langchain.py)

1. 运行脚本：

    ```bash
    python retrieval_augmented_generation_with_langchain.py
    ```

## vLLM + llamaindex

### 前提条件

设置 vLLM 和 llamaindex 环境：

```bash
pip install vllm \
            llama-index llama-index-readers-web \
            llama-index-llms-openai-like    \
            llama-index-embeddings-openai-like \
            llama-index-vector-stores-milvus \
```

### 部署

1. 启动 vLLM 服务器，使用受支持的嵌入模型，例如：

    ```bash
    # 启动嵌入服务（端口 8000）
    vllm serve ssmits/Qwen2-7B-Instruct-embed-base
    ```

1. 启动 vLLM 服务器，使用受支持的聊天补全模型，例如：

    ```bash
    # 启动聊天服务（端口 8001）
    vllm serve qwen/Qwen1.5-0.5B-Chat --port 8001
    ```

1. 使用脚本：[examples/applications/rag/retrieval_augmented_generation_with_llamaindex.py](../../../examples/applications/rag/retrieval_augmented_generation_with_llamaindex.py)

1. 运行脚本：

    ```bash
    python retrieval_augmented_generation_with_llamaindex.py
    ```
