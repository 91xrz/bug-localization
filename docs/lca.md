# LCA 缺陷定位数据

本项目接入 Long Code Arena 的 **150 条人工核验测试报告**：Java、Python、Kotlin 各 50 条。原始数据来自 [JetBrains Research](https://huggingface.co/datasets/JetBrains-Research/lca-bug-localization)，论文为 [Long Code Arena (2024)](https://arxiv.org/abs/2406.11612)。固定 HF 修订为 `f2bb26a1c3320ccc1d7b02d2630077ff327a80f6`。

## 数据与源码位置

服务器的 `datasets/`、`repos/`、`cache/`、`results/` 均通过软链接指向 `/data/xrz/bug-localization/` 机械盘。

- `datasets/LCA_bug_localization/`：作者 Parquet 和 126 个 Git ZIP 包；保留原始下载 SHA256。
- `datasets/LCA_bug_localization/repositories/<语言>/<owner>__<repo>/`：解压出的离线 Git 仓库。
- `datasets/lca_java_formatted/`、`datasets/lca_py_formatted/`、`datasets/lca_kt_formatted/`：各自包含 50 条 `test` 记录和 `lca_sources.json` 注册表。
- `repos/lca/<格式化数据集>/<owner>/<repo>/<base_sha>/source/`：实际实验读取的源码快照。
- 原始目录中的 `metadata/preparation-status.json`：解压、提交核验与快照准备进度；`metadata/split-source-coverage.json`：所有原始划分的源码包覆盖情况。

运行 `python scripts/datasets/prepare_lca.py` 可准备数据。它只读取本地压缩包和 Git 对象，不联网、不 checkout、不执行仓库中的程序。ZIP 路径经过校验；归档工作区中的符号链接保存为普通文本，**实验不读取这个工作区**。源码由 Git 对象按精确 `base_sha` 提取，跳过符号链接与子模块，逐 blob 校验 Git SHA1，并保存源码 SHA256 清单。已有缓存被修改时拒绝复用。

## 评价口径

`before_fix_sha` 对应作者的 PR `base_sha`，不能替换为 `head_sha`、仓库 HEAD 或标签版本。查询只由原始 issue 标题和正文构成，不使用 diff、补丁或修复 PR 标题。标签保留原始 `changed_files`，仅去除重复项；即便某个标签不在修复前版本或候选文件中，也保留在评价分母中。

兼容字段 `issue_id` 使用 **LCA 原始行 ID**，不等于 GitHub issue 编号。原始 `issue_url`、`pull_url`、`text_id` 另行保留，避免同一 issue 对应多个 PR 时被错误去重。

仅转换测试划分。原始训练／开发数据保留在原目录；作者发布包未覆盖其中所有项目，不能当作源码齐全的数据集。`dev` 包含 `train` 与 `test`，不得混合后当作独立样本。测试集来源不同也不保证与 BeetleBox 等数据无交叉，跨数据集训练前应按仓库、issue 和提交去重。

默认 Python 候选为 `.py/.pyi`；Java 和 Kotlin 配置使用 `.java/.kt/.kts`，保留混合项目中的两种语言。标准的候选大小、二进制、排除目录规则仍然适用。Kotlin AST 使用 `tree-sitter-kotlin==1.1.0`；不需要下载额外模型。

## 运行

先启动项目原有 Elasticsearch，再在项目根目录激活 `.venv`。下面每个比较配置均包含该语言全部 50 条测试记录，使用 BM25 与 AST96；三种语言应串行运行。GPU 配置沿用现有双卡设置，执行前应确保没有其他实验占用资源。

```bash
python scripts/environment/verify_environment.py --check-lca
# 对150份缓存逐文件校验，并审计标签覆盖
python scripts/datasets/audit_lca.py

# 先做单条 BM25 验证
python scripts/retrieval/run_bm25.py --config configs/retrieval/lca/py/bm25.json

# 各语言完整 50 条比较；正式运行时一次执行一个
python scripts/retrieval/run_retrieval_comparison.py --config configs/retrieval/lca/java/comparison.json
python scripts/retrieval/run_retrieval_comparison.py --config configs/retrieval/lca/py/comparison.json
python scripts/retrieval/run_retrieval_comparison.py --config configs/retrieval/lca/kt/comparison.json
```

也可使用同目录的 `bm25_batch.json` 或 `vector.json` 分别运行现有批量入口，或加 `--limit 1` 进行单条检查。主配置默认保留标题和正文，关闭融合，评价 @10。不得将单条功能检查的指标当作完整基准结果。
