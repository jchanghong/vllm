# Hugging Face Inference Endpoints

## 概述

与 vLLM 兼容的模型可以部署到 Hugging Face Inference Endpoints，既可以从 [Hugging Face Hub](https://huggingface.co) 开始，也可以直接从 [Inference Endpoints](https://endpoints.huggingface.co/) 界面开始。这样您就可以在完全托管的环境中提供模型服务，支持 GPU 加速、自动扩展和监控，无需手动管理基础设施。

有关 vLLM 集成和部署选项的高级详细信息，请参阅[高级部署细节](#advanced-deployment-details)。

## 部署方法

- [**方法 1：从目录部署。**](#method-1-deploy-from-the-catalog) 一键从 Hugging Face Hub 部署模型，附带现成的优化配置。
- [**方法 2：引导部署（Transformers 模型）。**](#method-2-guided-deployment-transformers-models) 从 Hub UI 使用**部署**按钮即时部署标记有 `transformers` 的模型。
- [**方法 3：手动部署（高级模型）。**](#method-3-manual-deployment-advanced-models) 适用于使用带有 `transformers` 标签的自定义代码的模型，或无法通过标准 `transformers` 运行但受 vLLM 支持的模型。此方法需要手动配置。

### 方法 1：从目录部署

这是在 Hugging Face Inference Endpoints 上开始使用 vLLM 的最简单方式。您可以在 [Inference Endpoints](https://endpoints.huggingface.co/catalog) 浏览已验证且经过优化部署配置的模型目录，以获得最佳性能。

1. 前往 [Endpoints Catalog](https://endpoints.huggingface.co/catalog)，在 **Inference Server** 选项中选择 `vLLM`。这将显示当前具有优化预配置选项的模型列表。

    ![Endpoints Catalog](../../assets/deployment/hf-inference-endpoints-catalog.png)

1. 选择所需的模型并点击 **Create Endpoint**。

    ![Create Endpoint](../../assets/deployment/hf-inference-endpoints-create-endpoint.png)

1. 部署就绪后，您就可以使用该端点。将 `DEPLOYMENT_URL` 替换为控制台中提供的 URL，记得按需追加 `/v1`。

    ```python
    # pip install openai
    from openai import OpenAI
    import os

    client = OpenAI(
        base_url=DEPLOYMENT_URL,
        api_key=os.environ["HF_TOKEN"],  # https://huggingface.co/settings/tokens
    )

    chat_completion = client.chat.completions.create(
        model="HuggingFaceTB/SmolLM3-3B",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Give me a brief explanation of gravity in simple terms.",
                    }
                ],
            }
        ],
        stream=True,
    )

    for message in chat_completion:
        print(message.choices[0].delta.content, end="")
    ```

!!! note
    目录提供了针对 vLLM 优化的模型，包括 GPU 设置和推理引擎配置。您可以从 Inference Endpoints UI 监控端点并更新**容器或其配置**。

### 方法 2：引导部署（Transformers 模型）

此方法适用于在其元数据中包含 [`transformers` 库标签](https://huggingface.co/models?library=transformers) 的模型。它允许您直接从 Hub UI 部署模型，无需手动配置。

1. 在 [Hugging Face Hub](https://huggingface.co/models) 上导航到一个模型。  
   本示例将使用 [`ibm-granite/granite-docling-258M`](https://huggingface.co/ibm-granite/granite-docling-258M) 模型。您可以通过检查 [README](https://huggingface.co/ibm-granite/granite-docling-258M/blob/main/README.md) 中的前置信息来验证模型的兼容性，其中库被标记为 `library: transformers`。

2. 找到**部署**按钮。该按钮出现在标记为 `transformers` 的模型卡片右上角，如 [模型卡片](https://huggingface.co/ibm-granite/granite-docling-258M) 所示。

    ![Locate deploy button](../../assets/deployment/hf-inference-endpoints-locate-deploy-button.png)

3. 点击**部署**按钮 > **HF Inference Endpoints**。您将被带到 Inference Endpoints 界面来配置部署。

    ![Click deploy button](../../assets/deployment/hf-inference-endpoints-click-deploy-button.png)

4. 选择硬件（本示例中我们选择 AWS>GPU>T4）和容器配置。选择 `vLLM` 作为容器类型，然后按 **Create Endpoint** 完成部署。

    ![Select Hardware](../../assets/deployment/hf-inference-endpoints-select-hardware.png)

5. 使用已部署的端点。将 `DEPLOYMENT_URL` 替换为控制台中提供的 URL（记得添加所需的 `/v1`）。然后您可以通过编程方式或使用 SDK 使用端点。

    ```python
    # pip install openai
    from openai import OpenAI
    import os

    client = OpenAI(
        base_url=DEPLOYMENT_URL,
        api_key=os.environ["HF_TOKEN"],  # https://huggingface.co/settings/tokens
    )

    chat_completion = client.chat.completions.create(
        model="ibm-granite/granite-docling-258M",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://huggingface.co/ibm-granite/granite-docling-258M/resolve/main/assets/new_arxiv.png",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Convert this page to docling.",
                    },
                ]
            }
        ],
        stream=True,
    )

    for message in chat_completion:
        print(message.choices[0].delta.content, end="")
    ```

!!! note
    此方法使用最佳猜测默认值。您可能需要调整配置以满足您的特定需求。

### 方法 3：手动部署（高级模型）

某些模型需要手动部署，因为它们：

- 使用带有 `transformers` 标签的自定义代码
- 无法通过标准 `transformers` 运行，但受 `vLLM` 支持

这些模型无法使用模型卡片上的**部署**按钮进行部署。

在本指南中，我们使用 [`rednote-hilab/dots.ocr`](https://huggingface.co/rednote-hilab/dots.ocr) 模型演示手动部署，这是一个与 vLLM 集成的 OCR 模型（请参阅 vLLM [PR](https://github.com/vllm-project/vllm/pull/24645)）。

1. 开始新的部署。前往 [Inference Endpoints](https://endpoints.huggingface.co/) 并点击 `New`。

    ![New Endpoint](../../assets/deployment/hf-inference-endpoints-new-endpoint.png)

2. 在 Hub 中搜索模型。在对话框中，切换到 **Hub** 并搜索所需模型。

    ![Select model](../../assets/deployment/hf-inference-endpoints-select-model.png)

3. 选择基础设施。在配置页面上，从可用选项中选择云提供商和硬件。  
   本演示中，我们选择 AWS 和 L4 GPU。请根据您的硬件需求进行调整。

    ![Choose Infra](../../assets/deployment/hf-inference-endpoints-choose-infra.png)

4. 配置容器。滚动到**容器配置**，选择 `vLLM` 作为容器类型。

    ![Configure Container](../../assets/deployment/hf-inference-endpoints-configure-container.png)

5. 创建端点。点击 **Create Endpoint** 部署模型。

    端点就绪后，您可以使用 OpenAI Completion API、cURL 或其他 SDK 来使用它。如果需要，请记得在部署 URL 后追加 `/v1`。

!!! note
    您可以从 Inference Endpoints UI 调整**容器设置**（容器 URI、容器参数），然后按 **Update Endpoint**。这会使用更新后的容器配置重新部署端点。模型本身的更改需要创建新端点或使用其他模型重新部署。例如，对于本演示，您可能需要将容器 URI 更新为 nightly 镜像（`vllm/vllm-openai:nightly`）并在容器参数中添加 `--trust-remote-code` 标志。

## 高级部署细节

通过 [Transformers 建模后端集成](https://blog.vllm.ai/2025/04/11/transformers-backend.html)，vLLM 现在为任何与 `transformers` 兼容的模型提供 Day 0 支持。这意味着您可以立即部署此类模型，利用 vLLM 的优化推理，无需额外的后端修改。

Hugging Face Inference Endpoints 提供了一个完全托管的环境，用于通过 vLLM 提供模型服务。您可以部署模型，无需配置服务器、安装依赖项或管理集群。端点还支持跨多个云提供商（AWS、Azure、GCP）部署，无需单独的账户。

该平台与 Hugging Face Hub 无缝集成，允许您部署任何与 vLLM 或 `transformers` 兼容的模型，跟踪使用情况，并直接更新推理引擎。vLLM 引擎预配置好，支持优化的推理，并可在模型或引擎之间轻松切换，无需修改代码。这种设置简化了生产部署：端点在数分钟内准备就绪，包含监控和日志记录，让您专注于提供模型服务，而不是维护基础设施。

## 后续步骤

- 探索 [Inference Endpoints](https://endpoints.huggingface.co/catalog) 模型目录
- 阅读 Inference Endpoints [文档](https://huggingface.co/docs/inference-endpoints/en/index)
- 了解 [Inference Endpoints 引擎](https://huggingface.co/docs/inference-endpoints/en/engines/vllm)
- 理解 [Transformers 建模后端集成](https://blog.vllm.ai/2025/04/11/transformers-backend.html)
