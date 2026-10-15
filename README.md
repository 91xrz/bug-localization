# Bug Localization

基于 Elasticsearch 的文件级缺陷定位项目，已跑通 **BM25、动态块向量、方法块向量、固定块 400-token 向量**四条路线。当前默认对比 SWE-bench 格式化版中 `psf/requests` 的全部 44 条报告，统一使用 @10 指标，关闭 RRF。

## 1. Python 环境

本机已验证 Windows / Python 3.12.14 / RTX 5080。已有 `.venv` 可直接验证；新机器在项目根目录创建环境并安装完整依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B scripts\environment\verify_environment.py --semantic
```

`requirements.txt` 包含当前四路检索所需的直接依赖，固定 CUDA 12.8 版 PyTorch，已配置官方 CUDA wheel 索引。无需单独安装 CUDA Toolkit。完整已安装版本锁定在 `configs/environment/requirements.lock.txt`，可用于复现相同环境。

只运行数据检查或 BM25 时，可安装较小的依赖集合：

```powershell
.\.venv\Scripts\python.exe -m pip install -r configs\environment\requirements-base.txt
```

PyCharm 将项目根目录作为工程，解释器选择 `.venv/Scripts/python.exe`，直接运行 `scripts/` 中对应 Python 入口即可。

## 2. 数据、模型和 Elasticsearch

数据、模型和实验输出不上传 Git。目录如下，来源与数据限制见 [数据说明](docs/datasets.md)：

```text
datasets/BeetleBox/
datasets/swe_bench_formatted/
datasets/ye/
models/Qwen3-Embedding-0.6B/
```

当前检索使用 `swe_bench_formatted/test` 和本地 Qwen 模型。可选检查全部三份数据：

```powershell
.\.venv\Scripts\python.exe -B scripts\environment\verify_environment.py --semantic --check-data
```

Elasticsearch 使用 Windows 9.5.5，本机已有安装。新机器请按 [ES 安装和启停说明](docs/elasticsearch.md) 自行下载安装包，再运行本地配置脚本。

```powershell
.\scripts\elasticsearch\start_elasticsearch.ps1
```

保持该终端打开，在另一终端运行检索。ES 只监听 `127.0.0.1:9200`；BM25 是 ES 内置能力。

## 3. 运行四路对比

```powershell
.\.venv\Scripts\python.exe -B scripts\retrieval\run_retrieval_comparison.py
```

主配置：`configs/retrieval/comparison.json`。先运行 BM25 固定样本清单，再让三条语义路线绑定同一清单；候选文件、修复前提交、报告和标签逐条核对。该入口始终关闭 RRF。

结果保存在 `results/comparison/<run_id>/`：

| 文件 | 内容 |
| --- | --- |
| `summary.csv`、`summary.json` | 四路指标、成功/失败数量、标签覆盖情况及原始运行路径 |
| `samples.csv`、`samples.json` | 每条报告的四路 AP@10、RR@10、Top-1/5/10 和 Recall@10 |
| `selection.json` | 数据行号、仓库、issue、提交、报告及标签指纹 |
| `verification.json` | 样本、查询和候选一致性核对 |
| `effective_config.json`、`status.json` | 配置与运行阶段 |

原始排名和分块保存在对应的 `results/bm25/batches/`、`results/vector/batches/`；具体路径见总汇总的 `runs` 字段。

当前 44 条测试的 **MAP@10：BM25 0.4596、动态块 0.5650、方法块 0.5239、固定块 0.5466**。详细指标、数据覆盖和限制见 [评测说明](docs/evaluation.md)。这是单项目对比，不是完整 SWE-bench 成绩。

## 4. 单独运行与配置

| 入口 | 默认用途 | 配置 |
| --- | --- | --- |
| `scripts/retrieval/run_bm25.py` | 单条报告 BM25 验证 | `configs/retrieval/bm25.json` |
| `scripts/retrieval/run_bm25_batch.py` | 前 5 条报告 BM25 | `configs/retrieval/bm25_batch.json` |
| `scripts/retrieval/run_vector_batch.py` | 前 5 条报告的三种分块对比 | `configs/retrieval/vector.json` |
| `scripts/retrieval/run_retrieval_comparison.py` | 全部 44 条报告四路对比 | `configs/retrieval/comparison.json` |

批量入口支持 `--limit`（1–50）和可重复的 `--repository`。语义入口支持 `--selection` 绑定已有清单，此时不与数量/仓库过滤混用。模型编码、分块和缓存细节见 [语义检索说明](docs/semantic-retrieval.md)。

RRF 实现保留供后续实验，默认关闭；只有语义入口显式传 `--fusion` 才运行，四路对比入口不会启用它。

## 5. 文件组织与检查

| 目录 | 内容 | 上传 Git |
| --- | --- | --- |
| `configs/` | 检索、ES 和依赖配置 | 是 |
| `src/bug_localization/` | 快照、分块、编码、检索和指标实现 | 是 |
| `scripts/`、`tests/` | 运行入口与单元测试 | 是 |
| `docs/`、`references/blaze/` | 说明文档及原始 BLAZE 代码参考 | 是 |
| `datasets/`、`models/`、`repos/` | 数据、模型、固定提交的源码快照 | 否 |
| `cache/`、`results/` | 可复用缓存和实验输出 | 否 |
| `.venv/`、`.runtime/`、`.idea/` | Python 环境、ES 程序与数据、IDE 设置 | 否 |

逐文件用途见 [项目结构说明](docs/project-layout.md)。

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

当前 35 项单元测试通过。测试不访问 ES 或数据集；未安装 tree-sitter 时，语义分块测试会跳过。实际检索还需要 ES、相应数据和模型就绪。
