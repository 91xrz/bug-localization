# 功能推测：180 条跨项目固定参数验证

在 Requests 44 条探索后，将完整的“报告 → OB／EB／S2R → 功能推测 → AST64 语义检索”流程扩展到 9 个其他项目，每项目 20 条，共 180 条。固定推测权重 0.25、原文权重 1、RRF 常数 30，不在新样本上搜索参数。

## 事前范围与排除

项目：astropy/astropy、django/django、matplotlib/matplotlib、pydata/xarray、pylint-dev/pylint、pytest-dev/pytest、scikit-learn/scikit-learn、sphinx-doc/sphinx、sympy/sympy。

每项目独立以 `random.Random('20261110:<repository>')` 无放回抽取 20 条，再按数据集原始顺序执行。排除 Requests、Seaborn、Flask 全部报告及此前跨项目模块2实验的前 11 条 pytest 报告；也排除与这些报告正文完全相同的记录。选择不读取检索排名、评价指标或候选覆盖，不按生成质量或标签覆盖重新选样。

固定清单：`configs/experiments/selections/crossproject_180_seed20261110.json`（随仓库保存）。
清单 SHA256：`ebef98ee78aac9aebae27d7384354b301aa16c02ae93bf106cbfd225806e2824`。

这批报告此前可能用于模块1全量基线；它们未参与本轮模块2的 Requests 参数选择。因此称为固定参数跨项目验证，不称为整个研究从未观察过的全新盲测数据。

## 冻结设置

- 两阶段生成均使用本地 Qwen3-8B、BF16、greedy、关闭 thinking。
- 保留原有补全和功能推测提示词及校验策略，不依据新结果调整。
- 补全输出最多 1536 tokens，推测输出最多 1024 tokens。
- 为覆盖长报告，两阶段输入上限设为 16384 tokens；原文补全输入预检最大 10566 tokens，后续还需容纳补全字段。超长输入明确失败，不静默截断。此上限不改变检索侧 512-token 预算。
- 语义模型 Qwen3-Embedding-0.6B，单 GPU、BF16、batch size 64，AST64（1500 字符块、64-token 前缀、512-token 总输入）、Max 文件聚合。
- 四条路线：原始报告 `q0`；仅补全 `OB_EB_S2R`；三层功能推测 `hypothesis_all`；固定融合 `rrf_w0.25_c30`。
- 新批次不需要历史排名作为对照，四条路线在同一次运行中共用候选、代码块和代码向量。
- 空增强／推测为空排名，仍保留完整评价分母。检索失败不删除样本、不用其他报告替换。

各项目样本数相同，整体指标按 180 条报告汇总；同时输出按项目指标、逐报告配对差异、候选覆盖和生成失败统计。相同报告正文且补全文本一致时复用缓存，保留独立评价样本。

## 一键自动运行

```bash
.venv/bin/python -B scripts/experiments/prepare_hypothesis_validation.py
.venv/bin/python -B -u scripts/experiments/run_report_hypothesis.py \
  --config configs/experiments/report_hypothesis_crossproject180.json --gpu 1
```

自动流程依次执行补全、推测、检索、独立审计与最终汇总。生成与检索使用不同进程，阶段结束释放显存。逐样本日志写入 `results/report_hypothesis/automation/<运行时间>/`，终端只输出最终汇总。`status.json` 记录当前阶段与进程 ID。

`resolved_config.json` 记录自动生成的补全来源目录；原始实验配置不被改写。结果包含 `final_summary.json`、`comparison.csv`、`project_summary.json`、`verification.json` 与 `hypothesis_verification.json`。

大批量入口要求显式设置上限并提供内容校验的固定清单；原有小批次接口默认仍限制 50 条。初始化、固定清单与内容变化检查等共 130 项测试通过。

本次已交由 tmux 会话 `irbl-hypothesis-180` 托管。迁移时保留了已完成的 103 份补全产物及首次响应缓存，新进程复用缓存后继续剩余阶段。最终终端输出同时写入 `results/report_hypothesis/crossproject180-tmux.log`，后台入口信息记录在 `crossproject180-background.json`。通过 `tmux attach -t irbl-hypothesis-180` 可查看任务；聊天无需持续轮询。
