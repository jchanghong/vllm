# vLLM Wheels 的每日构建

vLLM 维护一个每次提交的 wheel 仓库（通常称为"每日构建"），地址为 `https://wheels.vllm.ai`，提供自 `v0.5.3` 以来 `main` 分支上每次提交的预构建 wheel。本文档解释了每日构建的 wheel 索引机制的工作原理。

## CI 上的构建和上传流程

### Wheel 构建

Wheel 在 PR 合并到主分支后在 `Release` 流水线（`.buildkite/release-pipeline.yaml`）中构建，包含多个变体：

- **后端变体**：`cpu` 和 `cuXXX`（例如 `cu129`、`cu130`）。
- **架构变体**：`x86_64` 和 `aarch64`。

每个构建步骤：

1. 在 Docker 容器中构建 wheel。
2. 重命名 wheel 文件名以使用正确的 manylinux 标签（当前为 `manylinux_2_31`），以符合 PEP 600 标准。
3. 将 wheel 上传到 S3 存储桶 `vllm-wheels` 下的 `/{commit_hash}/`。

### 索引生成

上传每个 wheel 后，`.buildkite/scripts/upload-wheels.sh` 脚本：

1. **列出 S3 中提交目录下所有现有的 wheel**
2. **使用 `.buildkite/scripts/generate-nightly-index.py` 生成索引**：
    - 解析 wheel 文件名以提取元数据（版本、变体、平台标签）。
    - 创建用于 PyPI 兼容性的 HTML 索引文件（`index.html`）。
    - 生成机器可读的 `metadata.json` 文件。
3. **将索引上传到多个位置**（覆盖现有文件）：
    - `/{commit_hash}/` - 始终上传，用于特定提交的访问。
    - `/nightly/` - 仅针对 `main` 分支的提交（非 PR）。
    - `/{version}/` - 仅针对发布 wheel（版本中不含 `dev`）。

!!! tip "处理并发构建"
    索引生成脚本可以处理多个变体同时构建的情况，它总是在生成索引前列出提交目录下的所有 wheel，从而避免竞态条件。

## 目录结构

S3 存储桶结构遵循以下模式：

```text
s3://vllm-wheels/
├── {commit_hash}/              # 特定提交的 wheel 和索引
│   ├── vllm-*.whl              # 所有 wheel 文件
│   ├── index.html              # 项目列表（默认变体）
│   ├── vllm/
│   │   ├── index.html          # 包索引（默认变体）
│   │   └── metadata.json       # 元数据（默认变体）
│   ├── cu129/                  # 变体子目录
│   │   ├── index.html          # 项目列表（cu129 变体）
│   │   └── vllm/
│   │       ├── index.html      # 包索引（cu129 变体）
│   │       └── metadata.json   # 元数据（cu129 变体）
│   ├── cu130/                  # 变体子目录
│   ├── cpu/                    # 变体子目录
│   └── .../                    # 更多变体子目录
├── nightly/                    # 最新的主分支 wheel（镜像最新提交）
└── {version}/                  # 发布版本索引（例如 0.11.2）
```

所有构建的 wheel 都存储在 `/{commit_hash}/` 中，而不同的索引则生成并引用它们。
这避免了 wheel 文件的重复。

例如，您可以指定以下 URL 来使用不同的索引：

- `https://wheels.vllm.ai/nightly/cu130` 获取使用 CUDA 13.0 构建的最新主分支 wheel。
- `https://wheels.vllm.ai/{commit_hash}` 获取特定提交构建的 wheel（默认变体）。
- `https://wheels.vllm.ai/0.12.0/cpu` 获取为 CPU 变体构建的 0.12.0 发布 wheel。

请注意，并非每个提交都存在所有变体。可用的变体会随时间变化，例如将 cu130 更改为 cu131。

### 变体组织

索引按变体组织：

- **默认变体**：没有变体后缀的 wheel（即使用当前 `VLLM_MAIN_CUDA_VERSION` 构建的）放置在根目录。
- **变体子目录**：带有变体后缀的 wheel（例如 `+cu130`、`.cpu`）组织在子目录中。
- **默认变体别名**：默认变体可以有一个别名（例如当前为 `cu129`），以保持一致性并方便使用。

变体从 wheel 文件名中提取（如[文件命名约定](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#file-name-convention)中所述）：

- 变体编码在本地版本标识符中（例如 `+cu129` 或 `dev<N>+g<hash>.cu130`）。
- 示例：
    - `vllm-0.11.2.dev278+gdbc3d9991-cp38-abi3-manylinux1_x86_64.whl` → 默认变体
    - `vllm-0.10.2rc2+cu129-cp38-abi3-manylinux2014_aarch64.whl` → `cu129` 变体
    - `vllm-0.11.1rc8.dev14+gaa384b3c0.cu130-cp38-abi3-manylinux1_x86_64.whl` → `cu130` 变体

## 索引生成细节

`generate-nightly-index.py` 脚本执行以下操作：

1. **使用正则表达式解析 wheel 文件名**以提取：
    - 包名称
    - 版本（同时提取变体）
    - Python 标签、ABI 标签、平台标签
    - 构建标签（如果存在）
2. **按变体分组 wheel**，然后按包名称分组：
    - 目前只构建 `vllm`，但该结构支持将来添加多个包。
3. **生成 HTML 索引**（符合[简单仓库 API](https://packaging.python.org/en/latest/specifications/simple-repository-api/#simple-repository-api) 标准）：
    - 顶层 `index.html`：列出所有包和变体子目录
    - 包级别 `index.html`：列出该包的所有 wheel 文件
    - 使用指向 wheel 文件的相对路径以确保可移植性
4. **生成 metadata.json**：
    - 包含所有 wheel 元数据的机器可读 JSON
    - 包含带有 URL 编码相对路径的 `path` 字段，指向 wheel 文件
    - 由 `setup.py` 用于在仅 Python 构建期间定位兼容的预编译 wheel

### AWS 服务的特殊处理

Wheel 和索引直接存储在 AWS S3 上，我们使用 AWS CloudFront 作为 S3 存储桶前的 CDN。

由于 S3 不提供适当的目录列表功能，为了支持 PyPI 兼容的简单仓库 API 行为，我们部署了一个 CloudFront Function，用于：

- 将任何不以 `/` 结尾且看起来不像文件（即最后一个路径段中不包含点 `.`）的 URL 重定向到带有尾部 `/` 的相同 URL
- 将任何以 `/` 结尾的 URL 附加 `/index.html`

例如，以下请求将被处理为：

- `/nightly` -> `/nightly/index.html`
- `/nightly/cu130/` -> `/nightly/cu130/index.html`
- `/nightly/index.html` 或 `/nightly/vllm.whl` -> 保持不变

!!! note "AWS S3 文件名转义"

    S3 会根据其[命名规则](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html)在上传时自动转义文件名。对 vllm 的直接影响是文件名中的 `+` 将被转换为 `%2B`。我们在索引生成脚本中特别注意正确转义文件名，以确保 URL 正确且可直接使用。

## `setup.py` 中预编译 wheel 的使用 {#precompiled-wheels-usage}

当使用 `VLLM_USE_PRECOMPILED=1` 安装 vLLM 时，`setup.py` 脚本：

1. **通过 `precompiled_wheel_utils.determine_wheel_url()` 确定 wheel 位置**：
    - 环境变量 `VLLM_PRECOMPILED_WHEEL_LOCATION`（用户指定的 URL/路径）始终优先，并跳过所有其他步骤。
    - 从 `VLLM_MAIN_CUDA_VERSION` 确定变体（可通过环境变量 `VLLM_PRECOMPILED_WHEEL_VARIANT` 覆盖）；默认变体也将作为后备尝试。
    - 确定此分支的_基础提交_（将在后面解释）（可通过环境变量 `VLLM_PRECOMPILED_WHEEL_COMMIT` 覆盖）。
2. **从 `https://wheels.vllm.ai/{commit}/vllm/metadata.json`（针对默认变体）或 `https://wheels.vllm.ai/{commit}/{variant}/vllm/metadata.json`（针对特定变体）获取元数据**。
3. **基于以下条件选择兼容的 wheel**：
    - 包名称（`vllm`）
    - 平台标签（架构匹配）
4. **从 wheel 下载并提取预编译产物**：
    - 原生扩展模块（`.so` 文件）
    - `vllm-rs` Rust 前端二进制文件
    - Flash Attention Python 模块和 Triton/FlashMLA Python 文件
5. **修补 package_data**以将提取的文件包含在安装中

!!! note "什么是基础提交？"

    基础提交是通过找到当前分支与上游 `main` 之间的合并基础来确定的，确保源代码与预编译二进制文件之间的兼容性。

_注意：使用预编译 wheel 之前，确保没有原生代码（例如 C++ 或 CUDA）更改是用户的责任。_

## 实现文件

涉及每日构建 wheel 机制的关键文件：

- **`.buildkite/release-pipeline.yaml`**：构建 wheel 的 CI 流水线
- **`.buildkite/scripts/upload-wheels.sh`**：上传 wheel 并生成索引的脚本
- **`.buildkite/scripts/generate-nightly-index.py`**：生成 PyPI 兼容索引的 Python 脚本
- **`setup.py`**：包含用于获取和使用预编译 wheel 的 `precompiled_wheel_utils` 类
