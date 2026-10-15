# BLAZE 原始代码参考

这两个文件由用户提供，整理时仅从项目根目录移动到此目录，内容未修改：

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `processor.py` | 原动态规划分块、语法节点代价和注释清理代码 | `549994613b8a7fa15611a065cbd7d57cd8fdcae2cf8dd6758e85544b5891c922` |
| `convert_to_chunks.py` | 原训练数据转换和上传流程参考 | `63883218899aff35a1c0ed4a8373a3c76957634b1309d63346ebd1b0286f5eb7` |

它们不作为项目运行入口。当前实现位于 `src/bug_localization/chunking.py`；单元测试只提取原文件中的纯 DP 方法，核对边界一致性，不导入或执行原文件的其他逻辑。

原代码使用的 llama-index、loguru、tree_sitter_languages 和 `utils.processor` 等旧依赖不加入当前项目 requirements；数据上传入口也不用于本地检索。
