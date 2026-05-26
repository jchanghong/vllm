# Cerebrium

<p align="center">
    <img src="https://i.ibb.co/hHcScTT/Screenshot-2024-06-13-at-10-14-54.png" alt="vLLM_plus_cerebrium"/>
</p>

vLLM 可以在基于云的 GPU 机器上通过 [Cerebrium](https://www.cerebrium.ai/) 运行，Cerebrium 是一个无服务器 AI 基础设施平台，使企业能够更轻松地构建和部署基于 AI 的应用程序。

安装 Cerebrium 客户端：

```bash
pip install cerebrium
cerebrium login
```

接下来，创建您的 Cerebrium 项目：

```bash
cerebrium init vllm-project
```

然后，要安装所需的包，请将以下内容添加到您的 cerebrium.toml 中：

```toml
[cerebrium.deployment]
docker_base_image_url = "nvidia/cuda:12.1.1-runtime-ubuntu22.04"

[cerebrium.dependencies.pip]
vllm = "latest"
```

接下来，让我们添加代码来处理您选择的 LLM 的推理（本示例使用 `mistralai/Mistral-7B-Instruct-v0.1`），将以下代码添加到您的 `main.py`：

??? code

    ```python
    from vllm import LLM, SamplingParams

    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")

    def run(prompts: list[str], temperature: float = 0.8, top_p: float = 0.95):

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
        outputs = llm.generate(prompts, sampling_params)

        # 打印输出。
        results = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            results.append({"prompt": prompt, "generated_text": generated_text})

        return {"results": results}
    ```

然后，运行以下命令将其部署到云端：

```bash
cerebrium deploy
```

如果成功，您将收到一个 CURL 命令，可以使用它来调用推理。只需记住在 URL 末尾添加您要调用的函数名称（在我们的例子中是 `/run`）

??? console "Command"

    ```bash
    curl -X POST https://api.cortex.cerebrium.ai/v4/p-xxxxxx/vllm/run \
    -H 'Content-Type: application/json' \
    -H 'Authorization: <JWT TOKEN>' \
    --data '{
    "prompts": [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is"
    ]
    }'
    ```

您应该会得到类似以下的响应：

??? console "Response"

    ```json
    {
        "run_id": "52911756-3066-9ae8-bcc9-d9129d1bd262",
        "result": {
            "result": [
                {
                    "prompt": "Hello, my name is",
                    "generated_text": " Sarah, and I'm a teacher. I teach elementary school students. One of"
                },
                {
                    "prompt": "The president of the United States is",
                    "generated_text": " elected every four years. This is a democratic system.\n\n5. What"
                },
                {
                    "prompt": "The capital of France is",
                    "generated_text": " Paris.\n"
                },
                {
                    "prompt": "The future of AI is",
                    "generated_text": " bright, but it's important to approach it with a balanced and nuanced perspective."
                }
            ]
        },
        "run_time_ms": 152.53663063049316
    }
    ```

现在您拥有了一个自动扩展的端点，只需为您使用的计算资源付费！
