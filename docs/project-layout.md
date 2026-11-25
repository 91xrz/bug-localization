# 项目文件说明

项目主线是 [A+B+C 文件级缺陷定位](main-method.md)：原报告语义检索、功能推测语义检索、候选合并与结构初排、代码证据重排。BM25、旧词面/语义融合与切块实验作为对照保留。全部文档见[文档索引](README.md)。

## 目录约定

```text
configs/
  baselines/       已登记的 A+B 初排定义与校验和
  retrieval/       语义检索、BM25 及外部对照的配置
    bench4bl/      Bench4BL 检索配置
    lca/           LCA 多语言配置
    ghrb/          GHRB 检索配置
  experiments/     功能推测、C 重排、消融与验证配置
    selections/    冻结样本清单
  environment/     依赖与环境版本记录
  elasticsearch/   ES 发行信息和配置模板
scripts/
  retrieval/       检索、固定初排重放与外部基线入口
  experiments/     A/B/C 实验调度、重排、训练与扩展验证
  analysis/        已有结果分析、独立审计与统计
  datasets/        数据准备、源码预取及模型准备
  figures/         实验图表生成
  llm/             本地模型服务启动
  environment/     环境检查
  elasticsearch/   ES 安装与启停
src/bug_localization/  可复用方法与数据处理模块
tests/                单元测试与多语言夹具
docs/                 方法、操作指南和实验记录
  figures/            供文档引用的图表
references/           原始参考代码及来源说明
```

所有命令从项目根目录执行，JSON 中相对路径也相对于项目根目录。实验脚本通常依赖指定的本地资产与冻结结果，先阅读对应协议再执行。

## 主线代码与入口

| 模块 | 主要文件 | 作用 |
| --- | --- | --- |
| A：语义检索 | `embeddings.py`、`vector.py`；`scripts/retrieval/run_vector_batch.py` | 编码、向量缓存、代码块检索与文件聚合 |
| B：功能推测 | `report_hypothesis.py`；`scripts/experiments/run_direct_hypothesis.py` | 从报告生成功能描述并检索 |
| A+B：原结构初排 | `lightweight_hypothesis.py`；`scripts/retrieval/run_current_baseline.py` | 冻结 `ab_structure_v1` 的结构分数、候选合并与重放 |
| 多语言结构 | `structural_matching.py`；`scripts/retrieval/run_structural_matching.py` | `structure_ast_v2` 的声明抽取与结构匹配 |
| C：基础重排 | `mixed_recall.py`、`reranking.py` | 双路选块与本地 Reranker 评分 |
| C：联合证据 | `evidence_packets.py`、`conservative_reranking.py` | 文件联合证据构建与排序修正工具；完整主线由下述实验入口固定 |
| 新候选配对 Max C | `scripts/experiments/run_ab_structure_c500.py` | 新旧候选共享相同查询—代码块对分数 |
| 完整 A+B+C | `scripts/experiments/run_ab_structure_newc500.py` | 新候选、逐块/联合评分与初排分数组合；只调整 C 前十 |
| 系数验证 | `coefficient_search.py`；`scripts/experiments/run_multidata_coefficients.py` | 按项目冻结开发/验证划分，多语言评分与系数组合比较 |
| 独立核验 | `scripts/analysis/audit_multidata_coefficients.py` | 校验评分来源、重建排名与指标，检查敏感性分析 |

表中未带目录的模块均位于 `src/bug_localization/`。阶段关系与输入要求见[运行与复现](main-method.md#运行与复现)。

## 公共基础设施与对照

| 文件/入口 | 作用 |
| --- | --- |
| `snapshot.py`、`bench4bl.py`、`lca.py` | 修复前源码、数据适配和标签映射 |
| `batch.py`、`bm25.py`、`paired_statistics.py` | 样本选择、共享指标、失败计分与配对统计；`bm25.py` 同时提供词面对照 |
| `chunking.py`、`ast_chunking.py`、`languages.py` | 基础分块、多语言解析与结构前缀 |
| `chunk_cache.py`、`parallel_chunking.py` | 分块缓存校验和并行处理 |
| `scripts/retrieval/run_bm25.py --batch` | 统一 BM25 批次入口 |
| `scripts/retrieval/run_retrieval_comparison.py` | 历史 BM25＋四种语义切块基础对照，不是完整 A+B+C 入口 |
| `scripts/retrieval/fuse_rankings.py`、`vote_rankings.py` | 历史词面/语义 RRF 与投票融合对照 |
| `scripts/retrieval/run_blaze.py`、`run_rrl.py`、`run_igsil.py`、`run_blagent.py` | 外部方法本地适配；设置差异见[外部基线](external-baselines.md) |
| `scripts/environment/verify_environment.py` | 依赖、CUDA、数据与环境检查 |

## 配置与复现记录

- [requirements.txt](../requirements.txt)：完整环境直接依赖，引用 `configs/environment/requirements-base.txt`。
- `configs/environment/requirements.lock.txt`：原 Windows 环境记录，不是当前服务器的完整锁文件。
- [current.json](../configs/baselines/current.json)：只登记 A+B 初排，不能据此将主线描述为只有 A+B。
- [ab_structure_newc500.json](../configs/experiments/ab_structure_newc500.json)：完整 500 条 A+B+C 组合的来源与 C 参数。
- `configs/experiments/selections/`：固定样本身份与顺序，用于配对评价。
- [方法定义](main-method.md)：当前流程与各实现版本的成绩边界。
- [历史操作指南](reproduction-requests44.md)：本地服务、模型哈希和基础检索验收。

## 本地资产

| 目录 | 内容 |
| --- | --- |
| `datasets/`、`repos/` | 数据集与修复前源码快照 |
| `models/` | Embedding、Reranker 与生成模型权重 |
| `cache/` | 可复用分块、向量与生成缓存 |
| `results/` | 实验输入、逐例排名、指标、审计、日志与运行时协议 |
| `.venv/`、`.runtime/` | 本机环境、推理服务和 ES 运行文件 |

这些目录由 `.gitignore` 排除，需单独准备和备份。文档指向它们的链接仅在资产齐备的本地环境可用。`references/` 内被 Git 跟踪的参考代码和 `docs/figures/` 文档图表随仓库保留。
