<!-- markdownlint-disable MD041 -->
建议使用 [uv](https://docs.astral.sh/uv/)（一个非常快速的 Python 环境管理器）来创建和管理 Python 环境。请按照[文档](https://docs.astral.sh/uv/#getting-started)安装 `uv`。安装 `uv` 后，您可以使用以下命令创建新的 Python 环境：

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
```
