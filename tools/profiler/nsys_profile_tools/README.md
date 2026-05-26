# gputrc2graph.py

此脚本处理启用了 -t cuda 追踪的 NVIDIA Nsight Systems（`nsys`）GPU 追踪文件
（`.nsys-rep`），并生成 GPU 和非 GPU 时间的内核级摘要和可视化。它对于
分析和剖析 nsys profile 输出非常有用。

## 使用方法

### 命令行参数

- `--in_file`  
  **（必需）**  
  输入文件及其元数据的列表。每个条目的格式应为：  
  `<nsys-rep>,<engine>,<model>,<elapsed_nonprofiled_sec>`  
    - `nsys-rep`：`.nsys-rep` 文件的路径。
    - `engine`：引擎名称（例如 `vllm`）。
    - `model`：模型名称（例如 `llama`、`gpt-oss`、`ds`）。
    - `elapsed_nonprofiled_sec`：未经分析的运行时间（秒）。指定 `0` 则使用 nsys-rep 文件中的运行时间
    （如果未分析的实际运行时间更短，这可能会夸大非 GPU 时间）。多个条目可用空格分隔。

- `--out_dir`  
  生成的 CSV 和 HTML 文件的输出目录。  
  如果未指定，结果将保存在当前目录中。

- `--title`  
  HTML 图表/可视化的标题。

- `--nsys_cmd`  
  `nsys` 命令的路径。  
  默认值：`nsys`（假设它已在您的 PATH 中）。  
  如果 `nsys` 不在您的系统 PATH 中，请使用此选项。

## 注意事项

- 确保已安装 pandas。
- 确保已安装 [nsys](https://developer.nvidia.com/nsight-systems/get-started)，如果 `nsys` 命令不在您的 PATH 中，请使用 `--nsys_cmd` 指定路径。
- 有关可用引擎和模型的更多详细信息，请参见脚本中的帮助字符串或运行：

```bash
python3 gputrc2graph.py --help
```

## 示例 1：分析单个 profile

要分析例如使用 vLLM 引擎的 gpt-oss 模型的 GPU 周期：

1. 运行以下命令以收集 nsys profile，用于 vllm serve 配置。

   ```bash
   nsys profile -t cuda -o run1 -f true --trace-fork-before-exec=true \
   --cuda-graph-trace=node --delay <DELAY> --duration <DURATION> \
   vllm serve openai/gpt-oss-120b ...
   ```

   其中：

   - DELAY：延迟 nsys 收集 profile 的秒数，确保 vllm 服务器启动并开始加载生成后才捕获 profile。
   - DURATION：nsys profile 运行和生成 profile 文件的时间。这应大于运行持续时间。

2. 再次运行，这次不收集 profile，并获取总运行时间（秒）。该值将被脚本用于计算分析的
   CPU（非 GPU）时间。

3. 假设第 2 步的运行耗时为 306 秒。运行脚本进行分析：

   ```bash
   python3 gputrc2graph.py \
   --in_file run1.nsys-rep,vllm,gpt-oss,306 \
   --title "vLLM-gpt-oss profile"
   ```

该命令将生成 2 个分析文件：

- result.html：将内核名称分类到不同类别中，以堆叠条形图形式呈现。
- result.csv：显示内核名称如何映射到不同类别。

### HTML 可视化 result.html

HTML 文件显示了由不同 GPU 子阶段或类别导致的运行时间秒数，其中 moe_gemm（混合专家 GEMM）内核是最大的类别，为 148 秒，其次是 "attn" 或注意力内核。这使用户能够优先关注需要优化的内核以进行性能优化。

![GPU 追踪可视化示例](images/html.png)

条形图下方还有一个附加数据表，可用于复制到其他后处理工具中。

![GPU 追踪表示例](images/html_tbl.png)

### 内核到类别映射 result.csv

假设用户希望专注于改进 triton 内核。它并不是最大的周期消耗者（9.74 秒），但可能尚未优化。
下一步是使用 result.csv 深入了解构成 triton 内核 GPU 周期的内核。下图显示
triton_poi_fused__to_copy_add_addmm_cat_.. 内核是 GPU 周期的最大贡献者。

![GPU 追踪 CSV 示例](images/csv1.png)

## 示例 2：分析多个 profile

假设用户有多个 nsys 追踪文件，分别针对不同的模型（例如本例中的 llama 和 gpt-oss），并希望比较它们的 GPU/非 GPU 时间，可以使用如下命令。

```bash
python3 gputrc2graph.py \
--in_file run1.nsys-rep,vllm,llama,100 run2.nsys-rep,vllm,gpt-oss,102 \
--out_dir results \
--title "vLLM Models Comparison"
```

分析过程与示例 1 类似，但现在将显示多个可以比较的堆叠条形图。不同内核的类别将保持不变，以便于比较同一类别的 GPU 周期。

一旦发现某个配置在某个类别上比其他配置有更多周期，下一步就是使用 CSV 文件查看哪些内核被映射到该类别，以及哪些内核占据了最多的时间，从而导致整个类别的差异。

## 示例 3：为新模型添加新的分类

要创建引擎 DEF 和模型 ABC，只需在与 gputrc2graph.py 相同的目录中添加一个与其他 JSON 文件格式相同的 JSON 文件。脚本将自动加载同一目录中所有作为引擎/模型规范的 JSON 文件。

然后，对于这个新模型，假设有 4 个内核需要分类为 "gemm" 和 "attn"，其中 gemm 内核的名称中包含 "*H*" 或 "*I*"，attn 内核的名称中包含 "*J*" 或 "*K*"，只需在与 gputrc2graph.py 相同的目录中添加另一个 .json 文件，格式与其他 JSON 文件相同，如下所示：

```json
{
  "DEF": {
      "ABC": { 
          "H|I": "gemm",
          "J|K": "attn",
          "CUDA mem": "non-gpu-H_D_memops",
          ".*": "misc"
      }
  }
}
```

字典中的每个条目包含：

- key：用于分类内核的正则表达式
- value：内核被分类到的类别。

最后 2 个条目对所有引擎/模型都是通用的，包括 CUDA 内存操作和用于无法分类的剩余内容的 'misc'。

当调用 gputrc2graph.py 时，使用以下方式指定包含此新模型/引擎的追踪文件：

```bash
--infile new.nsys-rep,DEF,ABC,<runtime>
```

如果 engine_DEF.json 文件已存在，只需将模型作为新节点添加到现有引擎文件中，放在其他模型之后。
