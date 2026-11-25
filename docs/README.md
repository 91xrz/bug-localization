# 文档索引

项目主线为 **A 原报告语义检索＋B 功能推测语义检索＋C 代码证据重排**。先读[方法定义](main-method.md)，再按任务选择协议与结果；返回[项目首页](../README.md)。

历史记录保留各自实验日期、候选协议与指标，不将 BM25 融合、旧 RRF 或旧 Max C3 作为当前完整方法。文档中的 `results/` 等链接指向本地资产，需要另行准备。

## 主线方法与使用

- [A+B+C：当前文件级缺陷定位方法](main-method.md)
- [A+B 初排子阶段：冻结基线与注册说明](current-baseline.md)
- [项目文件说明](project-layout.md)
- [A+B+C 文件级缺陷定位论文大纲](paper-outline.md)
- [历史 Requests 44 条检索对照复现](reproduction-requests44.md)
- [Elasticsearch 安装与启停](elasticsearch.md)

## 数据集与基础表示

- [本地数据与评测口径](datasets.md)
- [使用 Bench4BL](bench4bl.md)
- [LCA 缺陷定位数据](lca.md)
- [GHRB 下载与当前流程兼容性](ghrb-readiness-20261124.md)
- [语义检索与分块](semantic-retrieval.md)
- [BLagent 风格 AST 分块基线](blagent-ast.md)
- [Chunk 到文件的三种聚合](file-aggregation.md)
- [多语言结构匹配 structure_ast_v2](multilingual-structural-matching.md)
- [BM25 与四种语义切块评测](evaluation.md)

## 当前 A+B 初排与完整 C

- [A 原报告语义检索：分层改进实验](a-semantic-improvements.md)
- [B 的轻量改进：论文依据与 CPU 实验（2026-11-22）](b-lightweight-improvements.md)
- [新 AB 融合＋C：固定500条配对实验（2026-11-23）](ab-structure-c500.md)
- [新AB候选池＋新版C：固定500条完整组合（2026-11-23）](ab-structure-newc500.md)
- [C 联合证据＋初排分数：完整500条验证](c-evidence-prior500.md)
- [C500为什么改善：排名变化、信号消融与代码证据](c-evidence-prior500-mechanism.md)
- [AB 与 C 系数检查：历史数据诊断，外部数据待明确](stage-coefficient-sensitivity-20261124.md)
- [多数据集AB/C系数实验：已完成](multidata-coefficient-experiment-20261124.md)
- [多数据集系数实验结论](multidata-coefficient-results-20261125.md)

## 查询生成、召回与历史融合

- [模块2：面向检索收益的多视角报告查询](report-multiview.md)
- [模块2：新44条跨项目复核](report-multiview-crossproject.md)
- [OB / EB / S2R 报告增强：AST64 探索实验](report-enrichment-ast64.md)
- [补全报告后的功能推测与 AST64 检索](report-hypothesis-ast64.md)
- [功能推测：180 条跨项目固定参数验证](report-hypothesis-crossproject180.md)
- [不补全报告直接生成功能推测](direct-report-hypothesis.md)
- [原文与直接功能推测的离线融合](direct-report-fusion.md)
- [事实压缩还是功能推测：12 条配对先导实验](query-mechanism-pilot12.md)
- [500 条报告的无重排对比：L0 / S0 / H0](recall-baselines-500.md)
- [SWE 全量 L0 / S0 / H0（2026-11-18）](recall-full-swe.md)
- [B 功能推测对原报告语义 Top50 的补充分析](top50-complementarity.md)
- [双路融合小批诊断：45 条 CPU 测试](fusion-pilot45.md)
- [融合退化诊断与 90 条固定验证（2026-11-17）](fusion-regressions-and-repair90.md)
- [BM25 与 AST 的文件级 RRF 融合](hybrid-retrieval.md)

## C 重排、消融与扩展

- [Qwen3 语义检索后重排](semantic-reranking.md)
- [500 条报告扩展实验](semantic-reranking-500.md)
- [C/D 混合召回实验（500 条）](mixed-recall-reranking.md)
- [SWE 全量融合与重排（2026-11-19）](reranker-full-swe.md)
- [固定 C 候选池的代码块选择对照（500 条）](chunk-ablation-reranking.md)
- [A / B / A+B / A+B+C 模块消融（固定 500 条）](module-ablation-500.md)
- [500 条模块消融：固定等权 RRF](module-ablation-rrf500.md)
- [C3 补充实验](c3-supplement-experiments.md)
- [C3 四组补充实验结果](c3-supplement-results.md)
- [C3 离线失效归因与下一轮验证方案](c3-failure-attribution-and-next-validation.md)
- [C3 改进小批实验（2026-11-19）](c3-improvement-pilot.md)
- [C3 故障证据比较重排开发实验（2026-11-19）](c3-evidence-pilot.md)
- [C3 局部源码证据补充实验（2026-11-20）](c3-context-pilot.md)
- [0.6B → 8B联合重排：扩到500条与机制分析](c3-joint500-analysis.md)
- [去掉 0.6B、直接使用 8B 重排：固定 500 条](direct8b-reranking-500.md)
- [C 的主动证据核查：首位分歧实验](c-agent90.md)
- [固定候选内的两步查证重排](c-closed-exploration.md)
- [C 阶段改进：从混合选块转向保守的文件排序纠错](c-conservative-reranking-20261123.md)
- [C 重排器 LoRA：区分报告相关文件与修复文件](c-lora-round1.md)
- [C 重排器第二轮：按基座误判选择训练负例](c-lora-round2.md)

## 外部方法与本地适配

- [外部论文对比基线](external-baselines.md)
- [BLAZE、RRL 与本项目的 500 条对比](external-comparison-500.md)
- [BLAZE 发布编码器基线](blaze-baseline.md)
- [BLAZE 全量 SWE 评测（2026-11-19）](blaze-full-swe.md)
- [RRL 最小对比实验](rrl-baseline.md)
- [RRL 全量 SWE 评测（2026-11-19）](rrl-full-swe.md)
- [IGSIL / CoSIL 文件级实验](igsil-baseline.md)
- [IGSIL 文件级全量评测（2026-11-18）](igsil-full-swe.md)
- [BLAgent 复现与对比](blagent-baseline.md)
- [BLAgent 失败修复与共享模型并发](blagent-recovery.md)
- [GenLoc 本地模型复现可行性评估](genloc-reproduction-assessment.md)
- [GenLoc 本地模型小批实现](genloc-pilot.md)

## 历史汇总与草稿

- [历史实验进展（2026-11-14—23）](experiment-history.md)
- [实验结果汇总](experiment-summary-20261031.md)
- [近期实验汇总（2026 年 11 月 9–14 日）](experiment-summary-20261114.md)
- [切块实验结论](chunking-conclusions.md)
- [历史 C3 论文详细大纲](paper-outline-c3-history.md)
