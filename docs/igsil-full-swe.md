# IGSIL 文件级全量评测（2026-11-18）

为在 CPU 检索继续运行时使用空闲 GPU 1，本轮启动既定外部基线 IGSIL 的全量评测。直接输入缺陷报告和修复前源码，不依赖 S0/H0 排名。GPU 0 留给用户其他实验。

- 范围：与当前 L0/S0/H0 完全相同的 SWE 格式化版 test 全部 2,294 条、12 项目，原始数据行顺序。
- 配置：`configs/experiments/igsil_full_swe.json`。
- 结果：`results/igsil/full2294-20261118/`；启动协议与日志：`results/igsil/full2294-20261118-launch/`。
- tmux 实验会话：`irbl-igsil-full-20261118`；模型服务会话：`irbl-igsil-full-server-20261118`。

沿用 500 条实验的作者修订、文件级流程、Qwen2.5-Coder-7B-Instruct Q4_K_M 权重、32,768 上下文、每次最多 2,048 输出 token、请求 seed=42 和作者设定的温度。模型服务仅绑定 `CUDA_VISIBLE_DEVICES=1`，监听本机 8098；评测客户端隐藏 CUDA，源码结构解析使用 CPU。启动时核验了服务器默认生成设置、上下文、slot 数、模型别名、类型和路径与旧 500 条服务记录完全一致，`launch_protocol.json` 保留配置、源码、清单和服务程序哈希。

仍使用 `body`、共同 Python 候选及原始全部文件标签，不排除测试文件。最多输出 5 个文件，不补造第 6–10 项；正式方法对比应统一报告 @5，现有运行同时保存 @10 口径供核验。失败样本保留并计零，未完成时不把成功子集当作全量成绩。模型、候选与论文原设置的差别见 `docs/igsil-baseline.md`。

此入口逐条保存结果、模型调用和汇总；运行中的 `summary.json` 给出成功、失败及 pending。它不自动覆盖或续写已有目录；如运行被中断，应核验已保存记录后准备剩余清单，保留原调用记录，不能直接覆盖重跑。

```bash
tail -f results/igsil/full2294-20261118-launch/run.log
cat results/igsil/full2294-20261118/summary.json
tmux attach -t irbl-igsil-full-20261118
```

本轮与当前 CPU 检索共享服务器资源，因此运行耗时不能直接作为不同方法的公平延迟基准。没有重新生成 S0/H0 的 0.6B 向量，也没有修改正在运行的共享快照检索协议。
