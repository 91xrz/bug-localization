# 项目文件说明

项目根目录保留 README、依赖入口和 Git 忽略规则。代码、配置、说明和参考材料按用途存放；本地数据与运行产物不提交 Git。

## 代码与入口

| 文件 | 职责 |
| --- | --- |
| `src/bug_localization/snapshot.py` | 下载指定修复前提交的源码快照，验证并记录文件清单 |
| `src/bug_localization/bm25.py` | 候选读取、查询处理、BM25 建索引、检索及单样本指标 |
| `src/bug_localization/batch.py` | 批量样本选择、固定清单、失败记录和指标汇总 |
| `src/bug_localization/chunking.py` | 动态、方法、固定三种分块，真实结构前缀与 token 限制 |
| `src/bug_localization/ast_chunking.py` | BLagent 风格递归 AST 切块、多语言适配及独立缓存，保留旧 Python 三路行为 |
| `src/bug_localization/languages.py` | 语言扩展名、真实语法节点、解析器及版本注册 |
| `src/bug_localization/chunk_cache.py` | 按源码、参数和实现指纹复用并校验分块 |
| `src/bug_localization/embeddings.py` | 本地模型编码、池化、归一化及 SQLite 向量缓存 |
| `src/bug_localization/vector.py` | ES 向量检索、文件聚合、独立分数验证及可选融合 |
| `src/bug_localization/aggregation.py` | 同一份 Chunk 相似度上的 Max、Top-3 Mean、稳定归一化 LogSumExp |
| `src/bug_localization/comparison.py` | 绑定同一样本清单，对比 BM25 和三条语义路线 |
| `src/bug_localization/__init__.py` | Python 包标识 |
| `scripts/environment/verify_environment.py` | 环境检查；`--semantic` 检查编码依赖与 CUDA；`--check-data` 检查本地数据 |
| `scripts/elasticsearch/setup_elasticsearch.py` | 校验并解压用户下载的 ES ZIP，生成本地配置 |
| `scripts/elasticsearch/start_elasticsearch.ps1` | 启动本地 ES |
| `scripts/elasticsearch/stop_elasticsearch.ps1` | 核实进程归属后停止本地 ES |
| `scripts/retrieval/run_bm25.py` | 单报告 BM25 入口 |
| `scripts/retrieval/run_bm25_batch.py` | 批量 BM25 入口 |
| `scripts/retrieval/run_vector_batch.py` | 四种语义切块与三种聚合入口，可显式开启原有融合控制 |
| `scripts/retrieval/run_retrieval_comparison.py` | BM25 与四种语义切块公平对比入口，关闭融合 |
| `scripts/retrieval/verify_multilingual_ast.py` | 十种语法的真实模型与 ES 功能验证 |

## 配置与依赖

| 文件 | 用途 |
| --- | --- |
| `requirements.txt` | 当前完整环境的直接依赖，包含 CUDA 12.8 版 PyTorch |
| `configs/environment/requirements-base.txt` | 仅数据检查和 BM25 所需的直接依赖，被完整依赖入口引用 |
| `configs/environment/requirements.lock.txt` | 当前环境的全部安装版本，供复现使用 |
| `configs/elasticsearch/distribution.json` | ES 发行版本、官方 ZIP 地址和 SHA512 |
| `configs/elasticsearch/elasticsearch.yml` | 本地 ES 配置模板 |
| `configs/retrieval/bm25.json` | 单条 BM25、候选文件与查询规则 |
| `configs/retrieval/bm25_batch.json` | 默认 BM25 批次、样本范围和 @10 评测 |
| `configs/retrieval/vector.json` | 模型、四种分块、多语言映射、聚合、编码和向量检索配置 |
| `configs/retrieval/comparison.json` | 当前 44 条 BM25 与四种语义切块对比的配置入口 |
| `.gitignore` | 排除数据、模型、源码快照、缓存、结果、环境和 IDE 设置 |

## 文档、参考和测试

- [datasets.md](datasets.md)：三份本地数据的来源、字段限制及 BeetleBox 清洗口径。
- [elasticsearch.md](elasticsearch.md)：手动下载安装包、配置和启停。
- [semantic-retrieval.md](semantic-retrieval.md)：分块、结构前缀、编码、检索、缓存和可选融合。
- [evaluation.md](evaluation.md)：当前 44 条结果、指标定义、验证与限制。
- [file-aggregation.md](file-aggregation.md)：三种文件级聚合、参数、九组对比入口和结果。
- [blagent-ast.md](blagent-ast.md)：新增 AST 路线、多语言配置、对比与功能验证。
- `references/blaze/processor.py`、`convert_to_chunks.py`：用户提供的 BLAZE 原始代码，原样保存供核对，不是运行入口；同目录 README 记录来源与校验值。
- `tests/test_bm25_inputs.py`、`test_bm25_metrics.py`、`test_bm25_batch.py`：词法输入、指标和批次规则测试。
- `tests/test_vector.py`：分块、前缀、缓存、编码输入及融合规则测试，包括对原始动态算法的核对。
- `tests/test_comparison.py`：对比的清单绑定、候选与查询一致性、指标汇总测试。
- `tests/test_aggregation.py`：聚合公式、归一化、数值稳定性、完整分数清单与九组汇总测试。
- `references/blagent/`：用户提供的原始包装代码及校验记录；仅作参考，不执行 Chroma 流程。
- `tests/test_ast_chunking.py`、`tests/fixtures/multilingual/`：真实语法解析、四路多语言分块及缓存测试。

## 仅保存在本机的目录

| 目录 | 用途与清理原则 |
| --- | --- |
| `datasets/` | 已准备好的三份数据，保留 |
| `models/` | Qwen3-Embedding-0.6B 模型和 tokenizer，保留 |
| `repos/` | 按修复前提交固定的源码快照，后续实验继续复用 |
| `cache/chunks/`、`cache/embeddings/` | 有效分块及向量缓存；删除会导致重复计算 |
| `results/comparison/` | 当前 44 条对比的汇总与固定清单 |
| `results/bm25/`、`results/vector/` | 当前对比依赖的原始结果；与汇总一起保留 |
| `results/environment/` | 最近一次环境验证记录 |
| `results/elasticsearch/` | 安装记录及服务日志 |
| `.venv/` | 已安装的 Python 环境 |
| `.runtime/` | ES 程序、生成配置与索引数据 |
| `.idea/` | 本机 PyCharm 工程设置 |

安装用的临时 wheel 缓存、旧单条/5 条试跑结果与源码 `__pycache__` 已清理。运行后新生成的结果仍按以上目录保存。提交仓库时检查 `git status`，无需强制添加被忽略的本地文件。
