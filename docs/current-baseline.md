# A+B 初排子阶段：冻结基线与注册说明

**项目主线是完整 [A+B+C](main-method.md)。本页仅说明其中 A+B 初排的注册与重放；`run_current_baseline.py` 不执行 C，不能将其输出当作完整方法结果。**

2026-11-22 按用户决定，将**分数融合＋结构匹配**登记为 `ab_structure_v1`。后续召回/初排改进默认与此方案配对比较，不再重新搜索本版本参数。

## 冻结定义

- 原报告语义检索为 A，现有功能推测语义检索为 B。
- 候选范围：A Top50 与 B Top50 的去重并集；最终输出最多50个文件。
- 分数：`0.5*N(A) + 0.5*N(B) + 0.5*E`。
- `N(s)=clip((s-s50)/(s1-s50),0,1)`；不足50时使用末位，全同分时为零。
- E 沿用已验证实现：完整原报告body中的名称与修复前源码的函数/类声明匹配，按并集范围内IDF加权；路径/模块匹配、过滤规则和声明扫描器均冻结。
- 平局按 B 名次、A 名次、文件路径处理。
- 生成或检索依赖失败保留报告并计零，不回退到另一条路线；基础设施和完整性错误停止执行。
- 固定的是 `score_a0.5_e0.5` 组合，不是只在B Top50内改排序的消融版本。

当前注册信息：[current.json](../configs/baselines/current.json)。版本定义与依赖校验和：[ab_structure_v1.json](../configs/baselines/ab_structure_v1.json)。修改算法或参数应另建版本，不能继续使用相同基线名称覆盖已有成绩。

## 默认入口

从项目根目录验证当前基线：

```bash
.venv/bin/python scripts/retrieval/run_current_baseline.py --check-only
```

复算已保存的A/B分数及结构特征，导出标准逐报告结果；输出目录必须不存在：

```bash
.venv/bin/python scripts/retrieval/run_current_baseline.py \
  --output results/baselines/<new-run>
```

入口读取当前注册的固定参数，逐条重新计算最终排名，并要求与已通过独立审计的参考实验完全一致。该入口是已有参考数据的固定重放，不重新生成推测、编码或调用模型，也不执行开发集选参。输出 `cases/*/result.json`、`samples.json`、`summary.json`、`protocol.json` 和 `verification.json`。

参考导出：[results/baselines/20261122_ab_structure_v1](../results/baselines/20261122_ab_structure_v1)。以后新数据应按相同规则生成 A/B 排名与结构特征，并使用固定组合，不能直接把此参考导出的成绩套用于新数据。

## 冻结成绩

| 范围 | 总报告 | MAP@10 | Top10 | Top50 |
|---|---:|---:|---:|---:|
| 开发集 | 500 | 0.620474 | 91.60% | 97.60% |
| 固定参数后验证 | 1,794 | 0.622239 | 90.52% | 97.27% |
| 全量 | 2,294 | 0.621854 | 90.76% | 97.34% |

全量成功2,293、失败1条；失败计入分母。1,794条此前已被观察，不称作全新盲测集。相对旧融合的验证MAP差值为+0.070244，95%配对区间[+0.060016,+0.080623]，完整证据见[轻量改进研究](b-lightweight-improvements.md)与[补充配对检验](../results/b_lightweight/20261122_v1/robustness_vs_previous.json)。

## 与历史实验的关系

L0、S0、H0及旧B40+A补足/RRF60保留为对照。历史 `run_reranker_full.py` 等入口属于已冻结的旧协议，运行名称与旧结果含义保持一致；当前初排基线通过上述独立入口使用。

2026-11-23 已完成[新基线＋C的固定500条同条件对照](ab-structure-c500.md)：新AB+C的MAP@10为0.648499，旧AB+C为0.647302，差值未显著；Top1均为59.2%，Top10净增1条。当前登记的基线范围仍是召回/初排，C组合的此次结果独立保存。历史C3/FULL成绩属于旧候选流程，不能作为新基线+C的成绩。

同日进一步完成[新AB候选池＋新版联合证据/AB先验C](ab-structure-newc500.md)：固定`0.75z+0.25u+4p`、只调整C前十，500条MAP=0.691051、Top1=66.0%、Top10=93.2%。相对旧池新版C的额外差值未显著；相对新池原Max C的改善成立于本批。该完整组合是当前 A+B+C 主线的已完成实验；`configs/baselines/current.json` 的注册范围仍仅覆盖初排，完整 C 参数与来源由对应实验协议固定。
