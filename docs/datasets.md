# 本地数据与评测口径

仓库不包含数据集和模型。数据应保存为 Hugging Face `save_to_disk` 格式，放在项目根目录的 `datasets/`：

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

## 本项目当前数据版本

| 数据集 | 划分 | 记录数 | 原始下载元数据中的 HF 修订 |
| --- | --- | ---: | --- |
| BeetleBox | train | 10,191 | `ba0088aad87bd53632cb7d92c00a086551f908ea` |
| BeetleBox，清洗后 | test | 8,955 | 原始输入同上，执行下述清洗 |
| SWE-bench 格式化版 | test | 2,294 | `dffcc648c8b2fda4b0056fa67f841e8aaabb0a92` |
| Ye 派生版，内部名 `apache_bugs` | train | 22,709 | `b40a330ed5d54bbd0329878674ec2e892ab7c694` |

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
