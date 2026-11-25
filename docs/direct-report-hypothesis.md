# 不补全报告直接生成功能推测

对跨项目 180 条固定清单，Qwen3-8B 仅接收原始报告，一次生成模块 context、功能 functionality、逻辑 logic，拼接后检索 AST64 代码块，以 Max 得分排序文件。生成提示词仅修改输入说明以去掉 OB／EB／S2R，其他指令保持一致。推测属于假设，不读取仓库知识、源码、标签或排名。

| 方案 | MAP@10 | MRR@10 | Top-1 | Top-5 | Top-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原文 | 0.4635 | 0.5214 | 35.00% | 74.44% | 84.44% |
| 补全后推测 | 0.5223 | 0.5923 | 44.44% | 82.78% | 87.78% |
| 直接推测 | 0.5525 | 0.6292 | 50.56% | 79.44% | 90.56% |

直接推测相对补全后推测，AP@10 为 49 胜、89 平、42 负；9 项目中 6 个 MAP 提升。四个标注文件不在候选集中，报告保留在评价分母。180 条全部完成，无生成错误；候选、代码块、编码器身份及代码向量缓存与前次对照核对通过。原文 74 条输入被 512-token 限制截断，功能推测没有截断；收益不能全部归因于功能推测内容。

```bash
# 先完成 docs/report-hypothesis-crossproject180.md 中的 180 条对照实验。
.venv/bin/python -B -u scripts/experiments/run_direct_hypothesis_multigpu.py
```

该入口在两张 GPU 上按报告内容散列分片，重复报告落在同一工作进程；生成完成后使用 GPU 1 和先前代码向量做检索。单卡入口为 `CUDA_VISIBLE_DEVICES=1 .venv/bin/python -B -u scripts/experiments/run_direct_hypothesis.py --pipeline`。两者均依赖本机已保存的对照批次和模型缓存；生成与检索产物不纳入 Git。结果位于 `results/direct_report_hypothesis/20261110_113033_098831_retrieval/`。

后续原文与直接推测融合实验见 [融合实验](direct-report-fusion.md)。参数在同一数据上探索，不是独立测试结论。
