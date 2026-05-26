# 使用 OpenAI 批处理文件格式进行离线推理

```{important}
本文是使用 OpenAI 批处理文件格式执行批处理推理的指南，**而非**完整的批处理（REST）API。
```

## 文件格式

OpenAI 批处理文件格式由一系列以换行符分隔的 JSON 对象组成。

[在此查看示例文件。](https://github.com/vllm-project/vllm/blob/main/examples/features/openai_batch/openai_example_batch.jsonl)

每一行代表一个独立的请求。有关更多详细信息，请参阅 [OpenAI 包参考](https://platform.openai.com/docs/api-reference/batch/requestInput)。

```{note}
我们目前支持 `/v1/chat/completions`、`/v1/embeddings` 和 `/v1/score` 端点（completions 即将推出）。
```

## 前提条件

* 本文档中的示例使用 `meta-llama/Meta-Llama-3-8B-Instruct`。
    * 创建一个[用户访问令牌](https://huggingface.co/docs/hub/en/security-tokens)
    * 在您的机器上安装该令牌（运行 `hf auth login`）。
    * 通过[访问模型卡](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)并同意条款和条件来获取受限模型的访问权限。

## 示例 1：使用本地文件运行

### 步骤 1：创建批处理文件

要跟随此示例操作，您可以下载示例批处理文件，或在您的工作目录中创建自己的批处理文件。

```bash
wget https://raw.githubusercontent.com/vllm-project/vllm/main/examples/features/openai_batch/openai_example_batch.jsonl
```

创建后的批处理文件应如下所示：

```bash
cat features/openai_batch/openai_example_batch.jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
```

### 步骤 2：运行批处理

批处理运行工具设计为从命令行使用。

您可以使用以下命令运行批处理，结果将写入名为 `results.jsonl` 的文件：

```bash
python -m vllm.entrypoints.openai.run_batch \
    -i features/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

或使用命令行：

```bash
vllm run-batch \
    -i features/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

### 步骤 3：检查结果

现在您的结果应该在 `results.jsonl` 中。您可以通过运行 `cat results.jsonl` 来检查结果。

```bash
cat results.jsonl
{"id":"vllm-383d1c59835645aeb2e07d004d62a826","custom_id":"request-1","response":{"id":"cmpl-61c020e54b964d5a98fa7527bfcdd378","object":"chat.completion","created":1715633336,"model":"meta-llama/Meta-Llama-3-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! It's great to meet you! I'm here to help with any questions or tasks you may have. What's on your mind today?"},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":25,"total_tokens":56,"completion_tokens":31}},"error":null}
{"id":"vllm-42e3d09b14b04568afa3f1797751a267","custom_id":"request-2","response":{"id":"cmpl-f44d049f6b3a42d4b2d7850bb1e31bcc","object":"chat.completion","created":1715633336,"model":"meta-llama/Meta-Llama-3-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"*silence*"},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":27,"total_tokens":32,"completion_tokens":5}},"error":null}
```

## 示例 2：使用远程文件

批处理运行器支持通过 http/https 可访问的远程输入和输出 URL。

例如，要针对位于 `https://raw.githubusercontent.com/vllm-project/vllm/main/examples/features/openai_batch/openai_example_batch.jsonl` 的示例输入文件运行，您可以执行：

```bash
python -m vllm.entrypoints.openai.run_batch \
    -i https://raw.githubusercontent.com/vllm-project/vllm/main/examples/features/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

或使用命令行：

```bash
vllm run-batch \
    -i https://raw.githubusercontent.com/vllm-project/vllm/main/examples/features/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

## 示例 3：集成 AWS S3

要与云对象存储集成，我们推荐使用预签名 URL。

[在此了解有关 S3 预签名 URL 的更多信息]

### 额外前提条件

* [创建一个 S3 存储桶](https://docs.aws.amazon.com/AmazonS3/latest/userguide/creating-bucket.html)。
* `awscli` 包（运行 `pip install awscli`）用于配置您的凭据并交互式使用 S3。
    * [配置您的凭据](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html)。
* `boto3` Python 包（运行 `pip install boto3`）用于生成预签名 URL。

### 步骤 1：上传输入脚本

要跟随此示例操作，您可以下载示例批处理文件，或在您的工作目录中创建自己的批处理文件。

```bash
wget https://raw.githubusercontent.com/vllm-project/vllm/main/examples/features/openai_batch/openai_example_batch.jsonl
```

创建后的批处理文件应如下所示：

```bash
cat features/openai_batch/openai_example_batch.jsonl
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
```

现在将您的批处理文件上传到 S3 存储桶。

```bash
aws s3 cp features/openai_batch/openai_example_batch.jsonl s3://MY_BUCKET/MY_INPUT_FILE.jsonl
```

### 步骤 2：生成预签名 URL

预签名 URL 只能通过 SDK 生成。您可以运行以下 Python 脚本来生成预签名 URL。请务必将 `MY_BUCKET`、`MY_INPUT_FILE.jsonl` 和 `MY_OUTPUT_FILE.jsonl` 占位符替换为您的存储桶和文件名。

（该脚本改编自 <https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/s3/s3_basics/presigned_url.py>）

```python
import boto3
from botocore.exceptions import ClientError

def generate_presigned_url(s3_client, client_method, method_parameters, expires_in):
    """
    Generate a presigned Amazon S3 URL that can be used to perform an action.

    :param s3_client: A Boto3 Amazon S3 client.
    :param client_method: The name of the client method that the URL performs.
    :param method_parameters: The parameters of the specified client method.
    :param expires_in: The number of seconds the presigned URL is valid for.
    :return: The presigned URL.
    """
    try:
        url = s3_client.generate_presigned_url(
            ClientMethod=client_method,
            Params=method_parameters,
            ExpiresIn=expires_in,
        )
    except ClientError:
        raise
    return url


s3_client = boto3.client("s3")
input_url = generate_presigned_url(
    s3_client,
    "get_object",
    {"Bucket": "MY_BUCKET", "Key": "MY_INPUT_FILE.jsonl"},
    expires_in=3600,
)
output_url = generate_presigned_url(
    s3_client,
    "put_object",
    {"Bucket": "MY_BUCKET", "Key": "MY_OUTPUT_FILE.jsonl"},
    expires_in=3600,
)
print(f"{input_url=}")
print(f"{output_url=}")
```

该脚本应输出：

```text
input_url='https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_INPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091'
output_url='https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_OUTPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091'
```

### 步骤 3：使用预签名 URL 运行批处理运行器

现在您可以使用上一节生成的 URL 来运行批处理运行器。

```bash
python -m vllm.entrypoints.openai.run_batch \
    -i "https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_INPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091" \
    -o "https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_OUTPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091" \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

或使用命令行：

```bash
vllm run-batch \
    -i "https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_INPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091" \
    -o "https://s3.us-west-2.amazonaws.com/MY_BUCKET/MY_OUTPUT_FILE.jsonl?AWSAccessKeyId=ABCDEFGHIJKLMNOPQRST&Signature=abcdefghijklmnopqrstuvwxyz12345&Expires=1715800091" \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

### 步骤 4：查看结果

您的结果现在位于 S3 上。可以通过运行以下命令在终端中查看：

```bash
aws s3 cp s3://MY_BUCKET/MY_OUTPUT_FILE.jsonl -
```

## 示例 4：使用 embeddings 端点

### 额外前提条件

* 确保您使用的是 `vllm >= 0.5.5`。

### 步骤 1：创建批处理文件

将 embedding 请求添加到您的批处理文件中。以下是示例：

```text
{"custom_id": "request-1", "method": "POST", "url": "/v1/embeddings", "body": {"model": "intfloat/e5-mistral-7b-instruct", "input": "You are a helpful assistant."}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/embeddings", "body": {"model": "intfloat/e5-mistral-7b-instruct", "input": "You are an unhelpful assistant."}}
```

您甚至可以在批处理文件中混合聊天补全和 embedding 请求，只要您使用的模型同时支持聊天补全和 embeddings（注意：所有请求必须使用相同的模型）。

### 步骤 2：运行批处理

您可以使用与前面示例相同的命令运行批处理。

### 步骤 3：检查结果

您可以通过运行 `cat results.jsonl` 来检查结果。

```bash
cat results.jsonl
{"id":"vllm-db0f71f7dec244e6bce530e0b4ef908b","custom_id":"request-1","response":{"status_code":200,"request_id":"vllm-batch-3580bf4d4ae54d52b67eee266a6eab20","body":{"id":"embd-33ac2efa7996430184461f2e38529746","object":"list","created":444647,"model":"intfloat/e5-mistral-7b-instruct","data":[{"index":0,"object":"embedding","embedding":[0.016204833984375,0.0092010498046875,0.0018358230590820312,-0.0028228759765625,0.001422882080078125,-0.0031147003173828125,...]}],"usage":{"prompt_tokens":8,"total_tokens":8,"completion_tokens":0}}},"error":null}
...
```

## 示例 5：使用 score 端点

### 额外前提条件

* 确保您使用的是 `vllm >= 0.7.0`。

### 步骤 1：创建批处理文件

将 score 请求添加到您的批处理文件中。以下是示例：

```text
{"custom_id": "request-1", "method": "POST", "url": "/v1/score", "body": {"model": "BAAI/bge-reranker-v2-m3", "queries": "What is the capital of France?", "documents": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/score", "body": {"model": "BAAI/bge-reranker-v2-m3", "queries": "What is the capital of France?", "documents": ["The capital of Brazil is Brasilia.", "The capital of France is Paris."]}}
```

您可以在批处理文件中混合聊天补全、embedding 和 score 请求，只要您使用的模型都支持它们（注意：所有请求必须使用相同的模型）。

### 步骤 2：运行批处理

您可以使用与前面示例相同的命令运行批处理。

### 步骤 3：检查结果

您可以通过运行 `cat results.jsonl` 来检查结果。

```bash
cat results.jsonl
{"id":"vllm-f87c5c4539184f618e555744a2965987","custom_id":"request-1","response":{"status_code":200,"request_id":"vllm-batch-806ab64512e44071b37d3f7ccd291413","body":{"id":"score-4ee45236897b4d29907d49b01298cdb1","object":"list","created":1737847944,"model":"BAAI/bge-reranker-v2-m3","data":[{"index":0,"object":"score","score":0.0010900497436523438},{"index":1,"object":"score","score":1.0}],"usage":{"prompt_tokens":37,"total_tokens":37,"completion_tokens":0,"prompt_tokens_details":null}}},"error":null}
{"id":"vllm-41990c51a26d4fac8419077f12871099","custom_id":"request-2","response":{"status_code":200,"request_id":"vllm-batch-73ce66379026482699f81974e14e1e99","body":{"id":"score-13f2ffe6ba40460fbf9f7f00ad667d75","object":"list","created":1737847944,"model":"BAAI/bge-reranker-v2-m3","data":[{"index":0,"object":"score","score":0.001094818115234375},{"index":1,"object":"score","score":1.0}],"usage":{"prompt_tokens":37,"total_tokens":37,"completion_tokens":0,"prompt_tokens_details":null}}},"error":null}
```
