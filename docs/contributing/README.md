# 为 vLLM 做贡献

感谢您有兴趣为 vLLM 做贡献！我们的社区向所有人开放，欢迎各种形式的贡献，无论大小。您可以通过以下几种方式为项目做贡献：

- 识别并报告任何问题或错误。
- 请求或添加对新模型的支持。
- 建议或实现新功能。
- 改进文档或贡献操作指南。

我们也相信社区支持的力量；因此，回答问题、提供 PR 审查以及帮助他人同样是被高度认可且有益的贡献。

最后，支持我们最有影响力的方式之一是提高 vLLM 的知名度。在您的博客文章中谈论它，并强调它如何推动您的出色项目。如果您正在使用 vLLM，请在社交媒体上表达您的支持，或者只需给我们的仓库点星以表达您的赞赏！

## 任务板

不确定从哪里开始？请查看以下链接以寻找可处理的任务：

- [好的入门问题](https://github.com/vllm-project/vllm/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
    - [精选入职任务](https://github.com/orgs/vllm-project/projects/6)
- [新模型请求](https://github.com/vllm-project/vllm/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22new-model%22)
    - [具有多模态能力的模型](https://github.com/orgs/vllm-project/projects/10)

## 许可证

参见 [LICENSE](../../LICENSE)。

## 开发

为 vLLM 做贡献的第一步是克隆 GitHub 仓库：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

然后，配置您的 Python 虚拟环境。

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

如果您只开发 vLLM 的 Python 代码，请使用以下命令安装 vLLM：

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

要仅重新构建 Rust 前端二进制文件：

```bash
./build_rust.sh          # 发布构建
./build_rust.sh --debug  # 更快的开发构建
```

如果您同时开发 vLLM 的 Python 和 CUDA/C++ 代码，请先安装 PyTorch：

```bash
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
```

然后从 `requirements/build/cuda.txt` 安装必要的构建依赖项，跳过已在上一步中安装的 `torch`：

```bash
grep -v '^torch==' requirements/build/cuda.txt | uv pip install -r -
```

最后使用以下命令安装 vLLM：

```bash
uv pip install -e . --no-build-isolation
```

有关从源码安装以及为其他硬件安装的更多详细信息，请查看适用于您硬件的[安装说明](../getting_started/installation/README.md)，并跳转到"从源码构建 wheel"部分。

有关迭代 C++/CUDA 内核时的优化工作流程，请参阅[增量编译工作流程](./incremental_build.md)获取建议。

!!! tip
    vLLM 兼容 Python 3.10 至 3.13 版本。但是，vLLM 的默认 [Dockerfile](../../docker/Dockerfile) 使用 Python 3.12，CI 中的测试（`mypy` 除外）也使用 Python 3.12 运行。

    因此，我们建议使用 Python 3.12 进行开发，以最大程度降低本地环境与 CI 环境冲突的可能性。

### 代码检查

vLLM 使用 `pre-commit` 来检查和格式化代码库。如果您不熟悉 `pre-commit`，请参见 <https://pre-commit.com/#usage>。设置 `pre-commit` 非常简单：

```bash
uv pip install pre-commit>=4.5.1
pre-commit install
```

现在，每次提交时，vLLM 的 `pre-commit` 钩子都会自动运行。

!!! tip "提示"
    您可以使用以下命令手动运行 `pre-commit` 钩子：

    ```bash
    pre-commit run     # 对已暂存的文件运行
    pre-commit run -a  # 对所有文件运行（--all-files 的简写）
    ```

    ---

    某些 `pre-commit` 钩子仅在 CI 中运行。如果需要，您可以在本地运行它们：

    ```bash
    pre-commit run --hook-stage manual mypy-3.10
    ```

### 文档

MkDocs 是一个快速、简单且非常漂亮的静态站点生成器，专注于构建项目文档。文档源文件使用 Markdown 编写，并通过一个 YAML 配置文件 [mkdocs.yaml](../../mkdocs.yaml) 进行配置。

开始使用：

```bash
uv pip install -r requirements/docs.txt
```

!!! tip
    确保您的 Python 版本与插件兼容
    （例如，`mkdocs-awesome-nav` 需要 Python 3.10+）

MkDocs 带有内置的开发服务器，让您可以在编写文档时预览效果。
在仓库根目录下运行：

```bash
mkdocs serve                           # 包含 API 参考（约 10 分钟）
API_AUTONAV_EXCLUDE=vllm mkdocs serve  # 不含 API 参考（约 15 秒）
```

一旦在日志中看到 `Serving on http://127.0.0.1:8000/`，实时预览就准备好了！
在浏览器中打开 <http://127.0.0.1:8000/> 即可查看。

有关更多功能和高级配置，请参考：

- [MkDocs 文档](https://www.mkdocs.org/)
- [Material for MkDocs 文档](https://squidfunk.github.io/mkdocs-material/)（我们使用的 MkDocs 主题）

### 测试

vLLM 使用 `pytest` 来测试代码库。

```bash
# 安装 CI 中使用的测试依赖项（仅 CUDA）
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto

# 安装一些常见的测试依赖项（与硬件无关）
uv pip install pytest pytest-asyncio

# 运行所有测试
pytest tests/

# 运行单个测试文件并显示详细输出
pytest -s -v tests/test_logger.py
```

!!! tip "如果缺少 Python.h，请安装 python3-dev"
    如果上述任何命令因 `Python.h: No such file or directory` 而失败，请使用
    `sudo apt install python3-dev` 安装 `python3-dev`。

!!! warning "警告"
    目前，代码库尚未被 `mypy` 完全检查。

    ---

    目前，并非所有单元测试都能在 CPU 平台上通过。如果您没有 GPU
    平台来在本地运行单元测试，请暂时依赖持续集成系统来运行测试。

## 问题

如果您遇到错误或有功能请求，请先[搜索现有问题](https://github.com/vllm-project/vllm/issues?q=is%3Aissue)以查看是否已被报告。如果没有，请[提交新问题](https://github.com/vllm-project/vllm/issues/new/choose)，并提供尽可能多的相关信息。

!!! important
    如果您发现安全漏洞，请按照[此处](../../SECURITY.md)的说明操作。

## 拉取请求与代码审查

感谢您对 vLLM 的贡献！在提交拉取请求之前，
请确保 PR 满足以下标准。这有助于 vLLM 保持
代码质量并提高审查过程的效率。

### DCO 和 Signed-off-by

在向此项目贡献更改时，您必须同意 [DCO](../../DCO)。
提交必须包含 `Signed-off-by:` 头信息，以证明同意
DCO 的条款。

使用 `git commit` 的 `-s` 选项将自动添加此头信息。

!!! tip
    您可以通过 IDE 启用自动 sign-off：

    - **PyCharm**：在"提交"窗口中，点击"Commit and Push..."按钮右侧的"Show Commit Options"图标。
      这将打开一个 `git` 窗口，您可以在其中修改"Author"并启用"Sign-off commit"。
    - **VSCode**：打开[设置编辑器](https://code.visualstudio.com/docs/configure/settings)
      并启用 `Git: Always Sign Off`（`git.alwaysSignOff`）字段。

### AI 辅助贡献

在进行 AI 辅助贡献之前，您必须：

1. **亲自参与**：不要提交"纯 agent"PR。人类提交者负责审查所有更改的行、端到端验证行为以及运行相关测试。
2. **确保重要性**：避免一次性的"琐碎"PR（单个拼写错误、孤立的样式清理、单个可变默认值修复等）。将机械性的清理工作纳入清晰、系统的范围。

当 AI 工具在生成或修改代码时提供了非平凡的帮助，您必须：

1. **彻底审查**：您仍然对提交的所有代码负责。请以与手动编写代码相同的谨慎程度审查和理解 AI 生成的代码。
2. **在 PR 中披露**：始终说明拉取请求是否包含 AI 生成的代码。在 PR 描述中添加说明。
3. **标记提交**：使用提交跟踪信息（如 `Co-authored-by:`，其他项目使用 `Assisted-by:` 或 `Generated-by:`）添加归属信息。例如：

   ```text
   您的提交信息在此

   Co-authored-by: GitHub Copilot
   Co-authored-by: Claude
   Co-authored-by: gemini-code-assist
   Signed-off-by: 您的姓名 <your.email@example.com>
   ```

AI 辅助代码必须满足所有质量标准：适当的测试、文档、遵循风格指南以及彻底审查。归属信息有助于审阅者在上下文中评估贡献，并保持项目的法律清晰性。

### PR 标题与分类

只有特定类型的 PR 会被审查。PR 标题应带有适当的前缀以指示更改类型。请使用以下前缀之一：

- `[Bugfix]` 用于错误修复。
- `[CI/Build]` 用于构建或持续集成改进。
- `[Doc]` 用于文档修复和改进。
- `[Model]` 用于添加新模型或改进现有模型。模型名称应出现在标题中。
- `[Frontend]` 用于 vLLM 前端的更改（例如，OpenAI API 服务器、`LLM` 类等）
- `[Kernel]` 用于影响 CUDA 内核或其他计算内核的更改。
- `[Core]` 用于 vLLM 核心逻辑的更改（例如，`LLMEngine`、`AsyncLLMEngine`、`Scheduler` 等）
- `[Hardware][Vendor]` 用于特定于硬件的更改。供应商名称应出现在前缀中（例如，`[Hardware][AMD]`）。
- `[Misc]` 用于不属于上述类别的 PR。请谨慎使用此类别。

!!! note
    如果 PR 涉及多个类别，请包含所有相关的前缀。

### 代码质量

PR 需要满足以下代码质量标准：

- 我们遵循 [Google Python 风格指南](https://google.github.io/styleguide/pyguide.html)和 [Google C++ 风格指南](https://google.github.io/styleguide/cppguide.html)。
- 通过所有 lint 检查。
- 代码需要有良好的文档，以确保未来的贡献者能够轻松理解代码。
- 包含足够的测试以确保项目保持正确和健壮。这包括单元测试和集成测试。
- 如果 PR 修改了 vLLM 面向用户的行为，请在 `docs/` 中添加文档。这有助于 vLLM 用户理解和使用新功能或更改。

### 添加或更改内核

在积极开发或修改内核时，强烈建议使用[增量编译工作流程](./incremental_build.md)以获得更快的构建时间。
每个自定义内核都需要一个模式和一个或多个实现来注册到 PyTorch。

- 确保自定义操作按照 PyTorch 指南注册：
  [自定义 C++ 和 CUDA 运算符](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial)
  和[自定义运算符手册](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU)。
- 返回 `Tensor` 的自定义操作需要元函数。
  元函数应在 Python 中实现和注册，以便动态维度可以自动处理。请参见上述文档以了解元函数的描述。
- 使用 [torch.library.opcheck()](https://pytorch.org/docs/stable/library.html#torch.library.opcheck)
  来测试任何已注册操作的函数注册和元函数。参见 `tests/kernels` 获取示例。
- 当更改现有操作的 C++ 签名时，必须更新模式以反映更改。
- 如果需要新的自定义类型，请参见以下文档：
  [PT2 中的自定义类支持](https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA)。

### 大型更改说明

请尽量保持更改的简洁性。对于重大的架构更改
（超过 500 行，不包括内核/数据/配置/测试），我们希望有一个 GitHub Issue
（RFC）讨论技术设计和理由。否则，我们将标记
为 `rfc-required`，并且可能不会通过 PR。

### 审查流程预期

vLLM 团队的目标是成为一个*透明的审查机器*。我们希望
使审查过程透明高效，并确保没有贡献者感到困惑或沮丧。但是，vLLM 团队规模较小，因此我们
需要优先处理某些 PR。以下是您在审查过程中可以预期的：

- 提交 PR 后，PR 将被分配给审阅者。每位审阅者将根据其专业知识和可用性选取 PR。
- 分配 PR 后，审阅者将每 2-3 天提供状态更新。如果 PR 在 7 天内未获审查，请随时提醒审阅者或 vLLM 团队。
- 审查后，如果需要更改，审阅者将在 PR 上添加 `action-required` 标签。贡献者应处理评论并提醒审阅者重新审查 PR。
- 请在合理的时间范围内回复所有评论。如果评论不清楚或您不同意某个建议，请随时要求澄清或讨论该建议。
- 请注意，由于计算资源有限，并非所有 CI 检查都会执行。审阅者将在 PR 准备合并或需要完整 CI 运行时添加 `ready` 标签。

### 升级停滞的贡献

如果您有重要的贡献尚未获得维护者的关注，请通过以下邮箱联系我们：

<pr-review-request@vllm.ai>

请使用可验证的公司或大学邮箱，包括：

- 您的生产或研究用例
- 您遇到的问题
- 您的贡献如何解决该问题

## 感谢

最后，感谢您花时间阅读这些指南，并感谢您有兴趣为 vLLM 做贡献。
您的所有贡献都有助于使 vLLM 成为对所有人来说都出色的工具和社区！
