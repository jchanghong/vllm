# 分离式编码器

**分离式编码器** 在多模态 LLM 的一个独立进程中运行视觉编码器阶段，该进程与预填充/解码器阶段分开。将这两个阶段部署在独立的 vLLM 实例中带来了三个实际好处：

1. **独立、精细的扩展**
2. **更低的首次令牌延迟（TTFT）**
3. **跨进程复用和缓存编码器输出**

设计文档：<https://docs.google.com/document/d/1aed8KtC6XkXtdoV87pWT0a8OJlZ-CpnuLLzmR8l9BAE>

---

## 1  动机

### 1. 独立、精细的扩展

* 视觉编码器轻量级，而语言模型则大几个数量级。
* 语言模型可以并行化而不影响编码器集群。
* 编码器节点可以独立地添加或移除。

### 2. 更低的首次令牌延迟（TTFT）

* 纯语言请求完全绕过视觉编码器。
* 编码器输出仅在所需的注意力层注入，缩短了预填充的关键路径。

### 3. 跨进程复用和缓存

* 进程内编码器将复用限制在单个工作节点内。
* 远程共享缓存允许任何工作节点检索现有的嵌入，从而消除冗余计算。

---

## 2  使用示例

当前的参考路径是 **ExampleConnector**。
以下可立即运行的脚本展示了工作流程：

1 个编码器实例 + 1 个 PD 实例：
`examples/disaggregated/disaggregated_encoder/disagg_1e1pd_example.sh`

1 个编码器实例 + 1 个预填充实例 + 1 个解码实例：
`examples/disaggregated/disaggregated_encoder/disagg_1e1p1d_example.sh`

---

## 3  测试脚本

请参阅目录 `tests/v1/ec_connector`

## 4  开发

分离式编码通过运行两部分来实现：

* **编码器实例** – 执行视觉编码的 vLLM 实例。
* **预填充/解码（PD）实例** – 运行语言预填充和解码。
    * PD 可以是单个普通实例（使用 `disagg_encoder_example.sh`，即 E->PD），也可以是分离式实例（使用 `disagg_epd_example.sh`，即 E->P->D）。

连接器将编码器缓存（EC）嵌入从编码器实例传输到 PD 实例。
所有相关代码位于 `vllm/distributed/ec_transfer` 下。

### 关键抽象

* **ECConnector** – 用于检索编码器产生的 EC 缓存的接口。
    * *调度器角色* – 检查缓存是否存在并调度加载。
    * *工作节点角色* – 将嵌入加载到内存中。

下图说明了分离式编码器的流程：

![分离式编码器流程](../assets/features/disagg_encoder/disagg_encoder_flow.png)

对于 PD 分离部分，预填充实例接收缓存的方式与上述分离式编码器流程完全相同。预填充实例执行 1 步（预填充 -> 1 个 token 输出），然后将 KV 缓存传输到解码实例以执行剩余的生成。KV 传输部分完全在 PD 实例执行完成后进行。

`docs/features/disagg_prefill.md` 展示了关于分离式预填充（v0）的简要概念。

我们使用来自 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/` 的 **NixlConnector** 创建了示例设置，并参考了 `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py` 来促进 P 和 D 之间的 KV 传输。
