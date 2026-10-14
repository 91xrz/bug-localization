# Bug Localization

基于 Elasticsearch 的文件级缺陷定位基准项目，以缺陷报告查询修复前代码快照。目前已完成数据准备与 BM25 单样本验证；向量检索、混合检索和完整基准评测待实现。

## 1. Python 环境

已验证 Windows / Python 3.12.14。在项目根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B scripts\environment\verify_environment.py
```

如果 `python` 不在 PATH，使用本机 Python 3.12 的完整路径创建环境。验证默认只检查依赖，不要求下载数据；完整依赖版本保存在 `configs/environment/requirements.lock.txt`，可替代 `requirements.txt` 安装。

PyCharm 打开项目根目录，选择 `.venv/Scripts/python.exe` 作为解释器。直接运行 Python 脚本时不必激活环境。

## 2. 本地数据

数据和模型不随仓库上传。[数据目录、来源、修订、清洗规则和限制](docs/datasets.md)见单独文档。

准备好三份本地数据后验证：

```powershell
.\.venv\Scripts\python.exe -B scripts\environment\verify_environment.py --check-data
```

当前版本：BeetleBox train 10,191 / 清洗后 test 8,955，SWE-bench 格式化版 test 2,294，Ye 派生版 22,709。默认 BM25 示例只需要 SWE-bench 格式化版；三份数据检查是可选步骤。

## 3. 配置 Elasticsearch

固定版本：ES Windows x86_64 **9.5.5**，Python 客户端 **9.5.1**。BM25 为 ES 内置能力，无需单独安装；ZIP 自带 JDK，本流程不依赖 Docker。

请自行下载 [官方 ES Windows ZIP](https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-9.5.5-windows-x86_64.zip)（约 648 MiB）。[官方 SHA-512](https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-9.5.5-windows-x86_64.zip.sha512)已固定在 `configs/elasticsearch/distribution.json`。

假设安装包位于“下载”目录：

```powershell
.\.venv\Scripts\python.exe -B scripts\elasticsearch\setup_elasticsearch.py --archive "$env:USERPROFILE\Downloads\elasticsearch-9.5.5-windows-x86_64.zip"
```

脚本只读取本地安装包、校验、解压和配置，不下载软件。首次安装成功后，可删除 ZIP；保留已校验的安装目录和标记后，省略 `--archive` 仍可重新生成配置。

启动并保持当前终端打开：

```powershell
.\scripts\elasticsearch\start_elasticsearch.ps1
```

本机开发配置：单节点，绑定 `127.0.0.1:9200`，1 GiB JVM 堆，认证关闭。已运行时脚本不会重复启动。另开终端检查健康状态：

```powershell
Invoke-RestMethod http://127.0.0.1:9200/_cluster/health
```

在启动终端按 `Ctrl+C` 停止；也可运行 `scripts/elasticsearch/stop_elasticsearch.ps1` 停止经进程路径核对的本项目节点。索引数据保留在 `.runtime/elasticsearch/data`。

## 4. 运行 BM25

ES 启动后，在另一终端或 PyCharm 中运行：

```powershell
.\.venv\Scripts\python.exe -B scripts\retrieval\run_bm25.py
```

默认配置是 `configs/retrieval/bm25.json`；可用 `--config` 指定其他配置。

流程与口径：

- 默认选择 SWE-bench 的 `psf/requests #1142`，修复前提交为 `22623bd8c265b78b161542663ee980738441c307`。
- 首次获取该提交的单份 GitHub ZIP 代码快照，不克隆仓库历史；缓存文件按 SHA-256 核对。这里下载的是实验代码数据，不是软件安装包。
- 每个可按 UTF-8 解码且不超过 2 MiB 的 `.py` 文件建立一个文档；包含 Python 测试及随仓库发布的 Python 依赖文件。结果记录所有候选文件及排除原因。
- 分词拆分标点、下划线和驼峰，统一小写。BM25 使用 `k1=1.2`、`b=0.75`，未按测试成绩调参。
- 本地 SWE-bench 的 `body` 包含原始问题标题，单独的 `title` 可能来自修复 PR，因此只允许用 `body` 查询。
- 最多取 512 个不同的已分词查询项，以 OR 方式执行 BM25 term 查询，避免再次分词；默认返回前 50 个匹配文件。
- 标签只用于结果核对。候选集之外的标签会明确列出，仍保留在指标分母中。
- 索引绑定仓库、修复前提交和代码/配置指纹，可复用或恢复未完成的写入，避免混用不同快照。

输出 Top-1/5/10 命中、Recall@K、倒数排名@K、AP@K，默认 K=50。单个报告的结果仅用于流程验证，截断指标不能直接当作完整基准的 MRR/MAP。Ye 缺少修复前 SHA，需要先核实提交。

结果在 `results/bm25/psf__requests-1142/`：`result.json` 保存版本、查询、候选、排名、标签和指标；`effective_config.json` 保存本次配置。

## 5. 文件组织

| 目录或文件 | 用途 | 上传仓库 |
| --- | --- | --- |
| `README.md`、`docs/` | 操作说明与数据口径 | 是 |
| `requirements.txt`、`configs/` | 依赖、软件版本、节点和检索配置 | 是 |
| `scripts/` | 环境验证、软件配置及运行入口 | 是 |
| `src/bug_localization/` | 快照、索引、检索及指标实现 | 是 |
| `tests/` | 输入校验、文件筛选和指标测试 | 是 |
| `datasets/`、`models/`、`repos/` | 本地数据、模型和代码快照 | 否 |
| `.runtime/`、`.venv/` | 软件程序、索引数据和 Python 环境 | 否 |
| `results/` | 验证、检索结果和运行日志 | 否 |

完整版本锁定清单保留用于复现；软件安装和启动脚本保留用于新机器安装、重新配置和日常使用。

## 6. 检查

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

单元测试不依赖 ES 或数据集，也不需要第三方 Python 包。实际检索需要 ES 和所选数据集就绪。

参考：[Windows ZIP 安装](https://www.elastic.co/docs/deploy-manage/deploy/self-managed/install-elasticsearch-with-zip-on-windows)、[BM25 相似度](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity)、[Python 客户端](https://www.elastic.co/docs/reference/elasticsearch/clients/python/getting-started)。
