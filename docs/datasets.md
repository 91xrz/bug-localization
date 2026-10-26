# 本地数据与评测口径

仓库不包含数据集和模型。当前已接入的数据使用 Hugging Face `save_to_disk` 格式，放在项目根目录的 `datasets/`：

```text
datasets/
├── BeetleBox/
│   ├── dataset_dict.json
│   ├── train/
│   └── test/
├── swe_bench_formatted/
│   ├── dataset_dict.json
│   └── test/
└── ye/
    ├── dataset_dict.json
    └── train/
```

划分目录保留 `data-*.arrow`、`dataset_info.json` 和 `state.json`。只有 Parquet 文件时，需要先转换为上述本地格式。

## Bench4BL 作者原始包

原始文件另存于 `datasets/Bench4BL/`，不属于上述已接入的 HF 数据集。官方来源为 [exatoa/Bench4BL](https://github.com/exatoa/Bench4BL)，按作者脚本列出的51个项目包获取；`_archives/` 保留TAR，`data/<组>/<项目>/` 保存解压内容，`metadata/` 保存作者说明、下载清单及日志。实际完成状态与逐包SHA256以根目录的 `download_status.json` 为准。

**适配后的数据已另存为 `datasets/bench4bl_formatted/`**：46个现代项目、9,459份评价报告、690份发布版本注册信息，可用于当前BM25、Java AST和模块2。运行命令及旧项目限制见 [Bench4BL使用说明](bench4bl.md)。

2026-10-26本机已完成51/51个项目包的下载和解压：原始TAR约5.59 GB，解压出的普通文件约5.34 GB，合计约10.93 GB（不含文件系统额外占用）。逐包SHA256和198,127个原始普通文件的解压大小检查通过，记录在 `metadata/verification.json` 和 `metadata/SHA256SUMS.txt`。

已检查的原始项目包括 `bugrepo/repository.xml`（报告和 `fixedFiles` 标签）、`versions.txt`（版本标签映射）、`gitrepo/`（源码仓库）及其他评价元数据。部分TXT是Python字面量风格，不能统一当标准JSON解析，更不能直接执行。

适配器解析原始报告、按作者分组确定发布版本、映射真实文件路径并保存逐条审计；源码在运行时从本地Git对象按需读取，不使用当前工作区替代版本。兼容字段 `before_fix_sha` 在该数据中表示发布提交，结果明确记录其真实语义。若改用逐缺陷修复前提交，需额外解析映射并明确评价协议变化。

此数据使用独立Java候选配置和现有Java AST路线；下载脚本只准备原始数据，转换入口为 `scripts/datasets/prepare_bench4bl.py`。无需安装作者旧环境。Windows对包内安全的文档符号链接可能保存为内容副本；转换记录保存在下载状态中，原TAR保持不变。

## 本项目当前数据版本

| 数据集 | 划分 | 记录数 | 原始下载元数据中的 HF 修订 |
| --- | --- | ---: | --- |
| BeetleBox | train | 10,191 | `ba0088aad87bd53632cb7d92c00a086551f908ea` |
| BeetleBox，清洗后 | test | 8,955 | 原始输入同上，执行下述清洗 |
| SWE-bench 格式化版 | test | 2,294 | `dffcc648c8b2fda4b0056fa67f841e8aaabb0a92` |
| Ye 派生版，内部名 `apache_bugs` | train | 22,709 | `b40a330ed5d54bbd0329878674ec2e892ab7c694` |
| Bench4BL，本地作者发布版本适配 | test（评价语料） | 9,459 | 本地51包中的46个现代项目；精确tag/SHA与文件映射见转换审计 |

这些条数描述本项目使用的版本。远端数据可能变化，复现实验时应记录来源与修订。

## 来源

- [BeetleBox](https://huggingface.co/datasets/bug-localization/BeetleBox)
- [BLAZE 格式化的 SWE-bench](https://huggingface.co/datasets/Partha117/swe_bench_formatted)
- [SWE-bench 官方原始数据](https://huggingface.co/datasets/SWE-bench/SWE-bench)：字段与格式化版不同。
- [BLAZE 作者发布包](https://zenodo.org/records/15122980)：[作者 readme](https://zenodo.org/records/15122980/files/readme.md?download=1) 将 Ye 数据标为 `Dataset.zip/apache_bugs`；本项目将该目录保存为 `datasets/ye`。
- [Ye 原始 bugfiles 发布说明](https://openscience.us/repo/issues/bugfiles.html)：包含 AspectJ、Birt、Eclipse Platform UI、JDT、SWT 和 Tomcat。

## BeetleBox 清洗

原始本地测试集为 10,445 条。仅按全部字段完全相同去重，保留第一次出现及其原始顺序，删除 1,490 条重复记录；将保留记录中 7 个 `body=null` 改为空字符串。最终为 8,955 条。

字段类型、文件标签和提交 SHA 保留。不按空正文删除样本，不过滤测试、文档等修改文件；训练数据未进行去重或正文填充。评测时需说明使用这一清洗版，避免将结果直接当作原始测试集上的结果。

## 已知限制

- BeetleBox 数据卡文字与当前 HF 划分规模不一致。
- Ye 的 `before_fix_sha`、`issue_url`、`pull_url` 全部为空，部分 `body` 为空。需先核实修复前提交，再构建快照。
- Ye 的 22,709 条记录保存为 `train`，与 BLAZE 论文报告的 16,314 条测试数据存在差异，筛选规则尚待核实。存储划分名称不等于实际实验用途。
- `updated_files` 为字符串形式的列表，使用 `ast.literal_eval` 解析。
- 本地 SWE-bench 格式化版的 `body` 包含原始问题标题；单独的 `title` 可能来自修复 PR，所以当前检索仅允许 `body`。

## 数据验证

准备好三份数据后，在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -B scripts\environment\verify_environment.py --check-data
```

检查划分、上述条数、首条样本标签和 BeetleBox 测试集正文空值。只有环境检查时省略 `--check-data`。
