# BLagent 切块参考

这两个文件由用户提供，从项目根目录原样归档：

| 文件 | SHA-256 |
| --- | --- |
| `chroma_store.py` | `ce7d6da8e09d567e80a417f2ea8d4640de67e5adb452b37584aaa7b6d2636159` |
| `code_retriever.py` | `91af279e79662db9ba61424d536227c1db6e6f3e8ca31098d82760292be1da97` |

它们通过 Python Tree-sitter parser 调用 LlamaIndex `CodeSplitter`，包含 Chroma、LangChain、不同嵌入模型和文件筛选逻辑。本项目仅参考其 AST 切块方案；这些文件不被运行或导入，其中的 Chroma 和相关框架不加入依赖。

当前路线 `blagent_ast` 位于 `src/bug_localization/ast_chunking.py`，使用本地 Qwen 模型和 Elasticsearch。递归拆分超长语法节点、合并相邻节点的思路参考 [LlamaIndex CodeSplitter 上游实现](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/text/code.py)。用户提供的文件没有锁定 LlamaIndex 版本，因此本路线明确标为适配实现，不能当作 BLagent 完整系统或论文结果的严格复现。

显式使用默认 1500 字符预算，再适配共同的 512-token 输入和 64-token 结构前缀。源码空白、超长叶节点和解析错误文件均保留并记录，不按原脚本过滤测试文件。配置中的行数及重叠参数不被冒称为 AST 分块的实际控制参数。
