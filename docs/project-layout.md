# 项目文件说明

项目根目录保留 README、依赖入口和 Git 忽略规则。代码、配置、说明和参考材料按用途存放；本地数据与运行产物不提交 Git。

```text
configs/
  retrieval/       原有 BM25、语义检索和综合对比配置
    bench4bl/      Bench4BL的BM25、批次、AST96和综合对比配置
  experiments/     AST调参与改进、模块2多视角查询的实验配置
  environment/     基础依赖与环境版本记录
  elasticsearch/   ES 发行信息与配置模板
scripts/
  datasets/        Bench4BL原始包下载、转换与审计
  retrieval/       检索、融合及功能验证入口
  analysis/        已有实验结果的分析与来源审计
  llm/             启动用户已安装的本地 GGUF 推理服务
  environment/     环境检查
  elasticsearch/   ES 安装包处理与启停
src/bug_localization/  可复用算法模块
tests/                单元测试与多语言源码夹具
docs/                 原理、操作说明与实验结论
references/           用户提供的原始参考代码
```

常规对比从 `scripts/retrieval/run_retrieval_comparison.py` 启动；单独语义实验使用 `run_vector_batch.py --config configs/experiments/<配置名>.json`。所有命令从项目根目录运行，JSON 内的数据与模型路径也继续相对于项目根目录。无需在各个子目录间切换。

2026-10-26整理时，三份 `ast_*.json` 实验配置从 `configs/retrieval/` 移至 `configs/experiments/`，两个 `analyze_*_experiment.py` 从 `scripts/retrieval/` 移至 `scripts/analysis/`。历史结果中的配置路径作为运行记录保留；重新运行时请使用新路径。常用检索入口、默认配置、算法模块及缓存位置均未改动。

Bench4BL的四份检索配置现集中于 `configs/retrieval/bench4bl/`，文件名为 `bm25.json`、`bm25_batch.json`、`vector.json`、`comparison.json`。旧的 `bench4bl_*.json` 路径已迁移，模块2配置和文档引用已同步更新。

## 代码与入口

| 文件 | 职责 |
| --- | --- |
| `src/bug_localization/snapshot.py` | 下载指定修复前提交的源码快照，验证并记录文件清单 |
| `src/bug_localization/bench4bl.py` | Bench4BL作者版本分组、标签路径映射及本地发布源码提供器 |
| `src/bug_localization/bm25.py` | 候选读取、查询处理、BM25 建索引、检索及单样本指标 |
| `src/bug_localization/batch.py` | 批量样本选择、固定清单、失败记录和指标汇总 |
| `src/bug_localization/chunking.py` | 动态、方法、固定三种分块，真实结构前缀与 token 限制 |
| `src/bug_localization/ast_chunking.py` | BLagent 风格递归 AST 切块、多语言适配及独立缓存，保留旧 Python 三路行为 |
| `src/bug_localization/languages.py` | 语言扩展名、真实语法节点、解析器及版本注册 |
| `src/bug_localization/chunk_cache.py` | 按源码、参数和实现指纹复用并校验分块 |
| `src/bug_localization/embeddings.py` | 本地模型编码、池化、归一化及 SQLite 向量缓存 |
| `src/bug_localization/vector.py` | ES 向量检索、文件聚合、独立分数验证及可选融合 |
| `src/bug_localization/aggregation.py` | 同一份 Chunk 相似度上的 Max、Top-3 Mean、稳定归一化 LogSumExp |
| `src/bug_localization/comparison.py` | 绑定同一样本清单，对比 BM25 和默认四条语义路线 |
| `src/bug_localization/token_ast.py` | 独立实验：token 预算 AST 贪心与动态规划分块 |
| `src/bug_localization/dependency_context.py` | 独立实验：Python 静态调用与字段上下文、来源记录及缓存 |
| `src/bug_localization/hybrid.py` | 对齐已有结果，补齐 BM25 排名并进行文件级 RRF |
| `src/bug_localization/voting.py` | 按路线投票融合及并列排名的期望指标 |
| `src/bug_localization/report_views.py`、`multiview.py` | 模块2的报告证据抽取、缓存、加权 RRF 和检索调度 |
| `src/bug_localization/__init__.py` | Python 包标识 |
| `scripts/environment/verify_environment.py` | 环境检查；`--semantic` 检查编码依赖与 CUDA；`--check-data` 检查本地数据 |
| `scripts/datasets/download_bench4bl.py`、`prepare_bench4bl.py` | 下载51个原始包，以及将46个现代项目转换为统一HF输入 |
| `scripts/elasticsearch/setup_elasticsearch.py` | 校验并解压用户下载的 ES ZIP，生成本地配置 |
| `scripts/elasticsearch/start_elasticsearch.ps1` | 启动本地 ES |
| `scripts/elasticsearch/stop_elasticsearch.ps1` | 核实进程归属后停止本地 ES |
| `scripts/retrieval/run_bm25.py` | 单报告 BM25 入口 |
| `scripts/retrieval/run_bm25_batch.py` | 批量 BM25 入口 |
| `scripts/retrieval/run_vector_batch.py` | 四种语义切块与三种聚合入口，可显式开启原有融合控制 |
| `scripts/retrieval/run_retrieval_comparison.py` | BM25 与四种语义切块公平对比入口，关闭融合 |
| `scripts/retrieval/verify_multilingual_ast.py` | 十种语法的真实模型与 ES 功能验证 |
| `scripts/retrieval/fuse_rankings.py` | 已有 BM25 与语义批次的文件级 RRF 入口 |
| `scripts/retrieval/vote_rankings.py` | 已有排名的按路线投票实验入口 |
| `scripts/analysis/analyze_chunking_experiment.py` | 配对分块实验的指标、成本与基线一致性分析 |
| `scripts/analysis/analyze_dependency_experiment.py` | 调用依赖实验的源码来源、预算和配对指标审计 |
| `scripts/retrieval/run_multiview.py` | 先准备多视角查询，再运行现有 BM25/AST 检索 |
| `scripts/analysis/analyze_multiview.py` | 查询证据、融合和历史 q0 对照审计 |
| `scripts/analysis/analyze_query_complementarity.py` | 已有查询排名的补救/损失、真实文件覆盖与事后选择上限 |
| `scripts/llm/serve_report_llm.py` | 启动已安装的 llama.cpp，支持 Windows 和 Ubuntu |

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
| `configs/retrieval/bench4bl/*.json` | Bench4BL的BM25、批次、Java AST96和综合对比入口，默认CSV 5条 |
| `configs/experiments/bench4bl_multiview.json` | 模块2的Bench4BL配置，复用既有生成模型和查询流程 |
| `configs/experiments/ast_prefix96_candidate.json` | 原 AST 的96 token前缀候选；不改变默认配置 |
| `configs/experiments/ast_token_experiment.json` | 原动态块、原 AST 与两种 token AST 对照 |
| `configs/experiments/ast_dependency_experiment.json` | 原 AST96 与最多2个函数的依赖增强对照 |
| `configs/experiments/report_multiview.json` | 模块2：生成模型身份、三视角与固定权重 |
| `configs/experiments/report_multiview_crossproject.json` | 同一模块的新44条跨项目样本，固定原参数 |
| `.gitignore` | 排除数据、模型、源码快照、缓存、结果、环境和 IDE 设置 |

## 文档、参考和测试

- [datasets.md](datasets.md)：本地数据来源、Bench4BL适配范围、字段限制及 BeetleBox 清洗口径。
- [bench4bl.md](bench4bl.md)：Bench4BL适配范围、发布版本口径、源码与标签审计、启动命令。
- [elasticsearch.md](elasticsearch.md)：手动下载安装包、配置和启停。
- [semantic-retrieval.md](semantic-retrieval.md)：分块、结构前缀、编码、检索、缓存和可选融合。
- [evaluation.md](evaluation.md)：当前 44 条结果、指标定义、验证与限制。
- [file-aggregation.md](file-aggregation.md)：三种文件级聚合、参数、九组对比入口和结果。
- [blagent-ast.md](blagent-ast.md)：新增 AST 路线、多语言配置、对比与功能验证。
- [ast-token-experiment.md](ast-token-experiment.md)：token AST 实验方案、结果与限制。
- [dependency-ast.md](dependency-ast.md)：调用依赖增强的方案、来源审计和44条结果。
- [hybrid-retrieval.md](hybrid-retrieval.md)：RRF、按路线投票与并列排名评估。
- [report-multiview.md](report-multiview.md)：模块2、本地生成模型、两阶段运行及证据校验。
- `references/blaze/processor.py`、`convert_to_chunks.py`：用户提供的 BLAZE 原始代码，原样保存供核对，不是运行入口；同目录 README 记录来源与校验值。
- `tests/test_bm25_inputs.py`、`test_bm25_metrics.py`、`test_bm25_batch.py`：词法输入、指标和批次规则测试。
- `tests/test_vector.py`：分块、前缀、缓存、编码输入及融合规则测试，包括对原始动态算法的核对。
- `tests/test_comparison.py`：对比的清单绑定、候选与查询一致性、指标汇总测试。
- `tests/test_aggregation.py`：聚合公式、归一化、数值稳定性、完整分数清单与九组汇总测试。
- `references/blagent/`：用户提供的原始包装代码及校验记录；仅作参考，不执行 Chroma 流程。
- `tests/test_ast_chunking.py`、`tests/fixtures/multilingual/`：真实语法解析、四路多语言分块及缓存测试。
- `tests/test_token_ast.py`、`test_dependency_context.py`：实验切块的预算、源码保留、依赖解析与缓存隔离。
- `tests/test_bench4bl.py`：精确发布版本、离线源码、原仓库HEAD保持不变、篡改检测与缺失gold保留。
- `tests/test_hybrid.py`、`test_voting.py`：结果配对、融合、投票与并列指标规则。

## 仅保存在本机的目录

| 目录 | 用途与清理原则 |
| --- | --- |
| `datasets/` | 原有数据、Bench4BL原始包和适配数据，保留 |
| `models/` | Qwen3-Embedding-0.6B 模型和 tokenizer，保留 |
| `repos/` | 按修复前提交固定的源码快照，后续实验继续复用 |
| `cache/chunks/`、`cache/embeddings/` | 有效分块及向量缓存；删除会导致重复计算 |
| `results/comparison/` | 当前 44 条对比的汇总与固定清单 |
| `results/bm25/`、`results/vector/` | 当前对比依赖的原始结果；与汇总一起保留 |
| `results/hybrid/`、`results/voting/` | 融合结果、候选排名及来源记录 |
| `results/multiview/`、`cache/report_queries/` | 模块2查询、证据、指标和可复用的生成响应 |
| `.runtime/llama/`、`results/llm/` | 用户提供的本机推理程序、进程记录与服务日志 |
| `results/maintenance/` | 可恢复的清理归档与清单；旧Web UI备份、无结果的启动失败记录、过期锁及字节码已集中至此 |
| `results/environment/` | 最近一次环境验证记录 |
| `results/elasticsearch/` | 安装记录及服务日志 |
| `.venv/` | 已安装的 Python 环境 |
| `.runtime/` | ES 程序、生成配置与索引数据 |
| `.idea/` | 本机 PyCharm 工程设置 |

本次整理保留数据、模型、已产生检索结果的实验、源码快照、有效缓存及参考代码。未启用的旧Web UI备份、两次ES未启动导致的空尝试、过期锁和字节码采用可恢复归档，没有永久删除。失败算法实验的实现、排名与结论继续保留。运行后新生成的结果仍按以上目录保存。提交仓库时检查 `git status`，无需强制添加被忽略的本地文件。
