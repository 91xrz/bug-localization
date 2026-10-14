# Bug Localization

面向软件缺陷定位的 Elasticsearch 混合检索基准项目。计划以缺陷报告为查询，在修复前代码快照中检索相关文件，逐步比较 BM25、向量检索和混合检索的效果。

当前初版已完成本地数据检查、BeetleBox 测试集清洗及 Python 环境准备，提供本地数据读取验证脚本。Elasticsearch 索引、BM25 检索、向量检索和正式评测尚未实现。

## 环境安装

已验证环境：Windows、Python 3.12.14。使用 Python 3.12 创建项目独立的虚拟环境。

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

如果 `python` 未加入 PATH，请使用本机 Python 3.12 解释器的完整路径替换第一条命令中的 `python`。

`requirements.txt` 固定直接依赖版本；要恢复本次完整依赖组合，可改用：

```powershell
.\.venv\Scripts\python.exe -m pip install -r configs\environment\requirements.lock.txt
```

锁定清单来自 Windows / Python 3.12 环境。虚拟环境应在新机器上重新创建。

需要激活 PowerShell 环境时执行：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止激活脚本，可以继续直接调用 `.venv\Scripts\python.exe`。

## 目录说明

```text
bug-localization/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── environment/
│       └── requirements.lock.txt
├── scripts/
│   └── environment/
│       └── verify_environment.py
├── src/                        # 后续核心实现，当前预留
├── datasets/                   # 本地数据，不提交
├── models/                     # 本地模型，不提交
├── results/                    # 运行日志和结果，不提交
└── .venv/                      # 本地 Python 环境，不提交
```

`configs` 保存配置和完整依赖清单，`scripts` 保存辅助程序，`src` 预留给后续索引、检索和评测代码。`.gitkeep` 用于保留尚无代码的 `src` 目录。

## 本地数据

数据集和模型文件不随仓库提供，按需自行准备。数据应保存为 Hugging Face `save_to_disk` 格式，并按以下结构放置：

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

每个划分目录需要保留 `data-*.arrow`、`dataset_info.json` 和 `state.json`。仅下载 Parquet 文件并不等于上述本地目录格式。

当前验证脚本对应的本地数据版本与条数：

| 数据集 | 划分 | 记录数 | 本地原始下载元数据中的 HF 修订 |
| --- | --- | ---: | --- |
| BeetleBox | train | 10,191 | `ba0088aad87bd53632cb7d92c00a086551f908ea` |
| BeetleBox，清洗后 | test | 8,955 | 原始输入同上，随后执行下述清洗 |
| SWE-bench 格式化版 | test | 2,294 | `dffcc648c8b2fda4b0056fa67f841e8aaabb0a92` |
| Ye 派生版，内部名称 `apache_bugs` | train | 22,709 | `b40a330ed5d54bbd0329878674ec2e892ab7c694` |

这些条数描述本项目当前使用的版本，不能直接替代论文的数据口径。远端数据可能发生变化，复现实验时应记录具体来源和修订。

### 来源

- [BeetleBox：bug-localization/BeetleBox](https://huggingface.co/datasets/bug-localization/BeetleBox)
- [BLAZE 格式化的 SWE-bench：Partha117/swe_bench_formatted](https://huggingface.co/datasets/Partha117/swe_bench_formatted)
- [SWE-bench 官方原始数据](https://huggingface.co/datasets/SWE-bench/SWE-bench)：字段与上述格式化版不同。
- [BLAZE 作者发布包](https://zenodo.org/records/15122980)：其 [readme](https://zenodo.org/records/15122980/files/readme.md?download=1) 将 Ye et al. 数据标为 `Dataset.zip/apache_bugs`；本项目将该目录保存为 `datasets/ye`。
- [Ye 原始 bugfiles 发布说明](https://openscience.us/repo/issues/bugfiles.html)：包含 AspectJ、Birt、Eclipse Platform UI、JDT、SWT 和 Tomcat 六个 Java 项目。

### BeetleBox 测试集清洗口径

原始本地测试集有 10,445 条记录。对全部字段完全相同的记录去重，按原始顺序保留第一次出现，删除 1,490 条重复记录；将保留记录中的 7 个 `body=null` 替换为空字符串。最终测试集为 8,955 条。

清洗保留字段类型、修复文件标签和提交 SHA，不按正文是否为空删除样本，不过滤测试、文档等修改文件。训练数据未进行去重或正文填充。

正式评测应标明使用这一清洗版本，避免将其结果直接视为原始 10,445 条测试集上的结果。

### 已知数据限制

- BeetleBox 数据卡文字中的规模与当前 HF 划分规模不一致。
- Ye 派生版的 `before_fix_sha`、`issue_url` 和 `pull_url` 全部为空，部分 `body` 为空。构建代码快照前需核实修复前提交。
- Ye 保存为 `train` 的 22,709 条记录，与 BLAZE 论文报告的 16,314 条测试记录存在差异，筛选规则尚待核实。
- `updated_files` 为字符串形式的列表；现有验证脚本使用 `ast.literal_eval` 解析。

## 验证

无需数据即可检查 Python 和依赖是否可导入：

```powershell
.\.venv\Scripts\python.exe -c "import sys, datasets, pyarrow; print(sys.version); print(datasets.__version__, pyarrow.__version__)"
```

准备好上述三个本地数据集后运行：

```powershell
.\.venv\Scripts\python.exe -B scripts\environment\verify_environment.py
```

脚本检查项目 `.venv`、数据划分、记录数、首条样本的文件标签及 BeetleBox 测试集的正文空值，成功时输出 `"status": "passed"`。缺少数据或使用其他版本时，检查会失败，应先核对数据目录和版本。

验证结果写入 `results/environment/verification.json`，该目录已被 Git 忽略。

## PyCharm

打开项目根目录，将项目解释器设置为 `.venv/Scripts/python.exe`，然后右键运行 `scripts/environment/verify_environment.py`。参数和数据路径由脚本根据项目目录确定。

## 仓库内容

提交 README、依赖清单、配置和脚本。`.gitignore` 已排除本地数据、模型、生成结果、虚拟环境、PyCharm 配置、缓存，以及常见数据/模型二进制文件和下载压缩包。
