# Bug Localization

面向文件级缺陷定位的 **A+B+C 方法**：A 使用原始缺陷报告进行语义检索，B 生成功能推测并独立检索，合并两路候选后，由 C 使用真实代码证据与初排分数完成重排。输入为缺陷报告和修复前源码，输出为候选文件排名。

[方法定义](docs/main-method.md) · [文档索引](docs/README.md) · [项目结构](docs/project-layout.md) · [最新系数验证](docs/multidata-coefficient-results-20261125.md)

## A+B+C 主线

| 阶段 | 职责 | 当前方案 |
| --- | --- | --- |
| A：原报告检索 | 保留报告中的症状、名称与直接线索 | 原报告 → Qwen3-Embedding → AST 代码块检索 → 文件排名 |
| B：功能推测检索 | 补充实现功能线索 | 仅根据报告生成功能推测 → 同一代码表示与检索流程 → 独立文件排名 |
| A+B：候选合并与初排 | 在固定预算内利用两路分数和源码结构 | A Top50 ∪ B Top50；归一化分数融合并加入结构匹配，保留 50 个文件 |
| C：代码证据重排 | 用真实实现核对报告相关性并改善头部排序 | 每文件 A 最优块＋B 补足至最多 3 块；0.6B 逐块与文件联合评分，结合初排分数调整 C 前十 |

当前参照公式：

```text
p = 0.5 N(A) + 0.5 N(B) + 0.5 E
s = 0.75 z + 0.25 u + 4 p
```

`E` 为报告与修复前源码的结构匹配分数，`z` 为逐块评分的最大 logit，`u` 为同文件代码块的联合 logit。先按 `z` 得到 C 初排，再仅按 `s` 调整其前十内部次序。具体归一化、选块、平局规则与版本边界见[方法定义](docs/main-method.md)。这些系数是当前参照设置，尚未证明跨数据集最优。

BM25 作为传统词面对照；BM25＋AST 的 RRF 融合、旧 B40＋A 补足、历史等权 RRF 和原 Max C 均保留为历史对照。AST 分块是基础表示，不单独作为研究创新点。8B 重排、Agent 与 LoRA 结果作为独立实验记录。

## 结果与适用范围

| 实验与范围 | MAP@10 | Top1 | Top10 | 说明 |
| --- | ---: | ---: | ---: | --- |
| A+B 初排，SWE 全量 2,294 条 | 0.621854 | — | 90.76% | `ab_structure_v1`，仅初排，含 500 条开发数据 |
| A+B+C，固定 SWE 500 条 | 0.691051 | 66.0% | 93.2% | 新 AB 候选＋联合证据/初排先验 C；499 成功、1 失败计零 |

完整方法的 500 条结果来自[新 AB＋新版 C 配对实验](docs/ab-structure-newc500.md)。同批新候选＋原 Max C 的 MAP@10 为 0.648499；相对旧候选＋新版 C 的增量为 0.000879，未显示显著差异。500 条已经参与过研究，不能称为独立盲测。

**2026-11-25 多数据集验证：**在 `structure_ast_v2` 下，对 GHRB 134 条及历史 SWE 500 条按项目划分开发/验证集。共享开发系数在 SWE 验证集提高 MAP，在 GHRB 验证集未复现收益，因此维持参照系数，不将开发网格最高点升级为统一默认。多语言结构版本、验证集与上表实验不同，详见[完整结论](docs/multidata-coefficient-results-20261125.md)。

所有结果以预定样本为分母，失败计零；数据集、候选池、结构版本与评分批次不同的结果须分别解释。外部方法的本地适配与原论文设置差异见[对比基线](docs/external-baselines.md)。

## 使用入口

所有命令在项目根目录执行。使用 Python 3.12 和项目依赖；已有环境可直接激活：

```bash
source .venv/bin/activate
python -m pip check
python -B -m unittest discover -s tests -v
```

新环境的依赖入口为 [requirements.txt](requirements.txt)。数据、模型、源码快照、运行结果与缓存不随 Git 分发；语义检索需要本地模型及对应检索服务，历史实验重放还需要指定的原始结果。

| 任务 | 入口与前提 |
| --- | --- |
| 理解/复现完整 A+B+C | [方法与阶段入口](docs/main-method.md#运行与复现)；按实验协议准备 A/B、逐块评分与联合评分来源 |
| 查看固定 500 条完整方法入口 | `python scripts/experiments/run_ab_structure_newc500.py --help`；依赖历史审计产物，不是从空目录一键执行 |
| 检查已登记的 A+B 初排版本 | `python scripts/retrieval/run_current_baseline.py --check-only`；只校验初排配置与代码哈希，不执行 C |
| 多语言结构与跨数据集验证 | [结构匹配](docs/multilingual-structural-matching.md)、[系数实验协议](docs/multidata-coefficient-experiment-20261124.md) |
| 复现历史基础检索对照 | [Requests 44 条操作指南](docs/reproduction-requests44.md)；包含环境、ES、模型哈希与结果验收 |

`configs/baselines/current.json` 登记的是 A+B 初排子阶段。完整 A+B+C 的候选来源、C 参数和输入校验由对应实验协议记录，详见[初排注册说明](docs/current-baseline.md)。

## 数据与文档

| 内容 | 阅读入口 |
| --- | --- |
| 数据集与标签口径 | [SWE 格式化数据](docs/datasets.md)、[Bench4BL](docs/bench4bl.md)、[LCA](docs/lca.md)、[GHRB](docs/ghrb-readiness-20261124.md) |
| 当前方法与机制 | [A+B+C](docs/main-method.md)、[联合证据＋初排先验](docs/c-evidence-prior500.md)、[机制分析](docs/c-evidence-prior500-mechanism.md) |
| 消融与外部对照 | [模块消融](docs/module-ablation-500.md)、[固定 RRF 消融](docs/module-ablation-rrf500.md)、[外部基线](docs/external-baselines.md) |
| 论文组织与研究边界 | [论文大纲](docs/paper-outline.md)、[最新跨数据集结论](docs/multidata-coefficient-results-20261125.md) |
| 全部历史记录 | [文档索引](docs/README.md)、[阶段进展归档](docs/experiment-history.md) |

代码位于 `src/bug_localization/`，运行入口位于 `scripts/`，配置与冻结样本清单位于 `configs/`，测试位于 `tests/`。详细用途见[项目结构](docs/project-layout.md)。

文档中指向 `results/`、`datasets/`、`models/` 等目录的链接为本地实验资产，需要另行取得；这些文件不会因合并代码而出现在 GitHub 中。
