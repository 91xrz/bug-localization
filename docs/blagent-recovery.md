# BLAgent 失败修复与共享模型并发

2026-11-18。原实验 `results/blagent/crossproject500-20261118/` 保持原样：112 条已处理，96 成功、16 失败、388 未处理。旧记录不删除，也不把修复后的样本选择性拼回旧结果。

## 诊断与修复

| 旧运行故障 | 数量 | bounded-v2 处理 |
|---|---:|---|
| SAS 得分字典含候选池外路径 | 7 | 最多两次模型纠正；仅提供原报告、合法候选、实际查看的骨架和上一份输出。纠正调用使用 JSON Schema 约束键名为候选路径，再验证非空及分数范围。不会自动映射、删掉或补造路径。 |
| 工具请求越界 | 3 | 返回明确的错误 Observation 和合法候选列表，禁止读取越界文件，让原 ReAct 循环继续。 |
| SAS 用完 15 轮仍无 Final Answer | 1 | 用同一 bounded finalization 收尾；必须已经发生真实骨架检查。 |
| EAR 超过 32K 上下文 | 5 | 能容纳时仍一次重排；超过预算时用两两比较的归并排序，每次使用原 EAR 提示，保留全部前五候选及 SAS 排名尾部。 |

超长 EAR 首先保留作者 `CodePruner` 全部输出；仅当单个文件不适合两文件比较预算时，改用五个完整原始检索块和显式限长的结构骨架。`ear_strategy.json`、`ear_effective_context.json`、`ear_comparisons.json` 保存分批策略、压缩内容、各次比较及排名。候选文件、全部检索证据块和原报告不会静默删除或截断。极端长报告仍会明确失败。

正常 ReAct 仍使用作者 Agent、提示词、15 轮上限与温度；纠正温度为 0。单报告模型调用总预算从 20 调整为 36，覆盖两路查询、最多 15 轮 SAS、2 次收尾以及超长 EAR 的比较／一次格式纠正。修复后的方法是**带有明确失败恢复策略的适配版**，并非原论文默认协议。

普通模型输出失败保留在分母，不因连续三个此类失败终止全批。网络、服务或运行时故障会停止对应工作进程，避免把基础设施问题连续算成数百个模型失败。

## 并发方式

- 同一个本地 Qwen3-8B checkpoint 转为 **BF16 GGUF，无 4/8-bit 权重量化**；norm 等小张量按转换器规则保存为 F32。原始权重、tokenizer、GGUF、转换器和服务程序均记录校验值。
- llama.cpp 固定 commit `b2e5e9b28b2484fbf94b543432ece638996a8b97`，共享模型，2 个并发 slot，每 slot 32768 tokens，F16 KV cache，continuous batching，prompt cache，无 context shift。监听本机 `127.0.0.1:8099`。
- 两个报告工作进程按固定清单奇偶位置分区；调用种子由模型／配置身份和请求内容确定，独立于 worker、报告在分区中的位置和调度顺序。
- 客户端沿用本地 HF tokenizer 和关闭 thinking 的 chat template，直接传 token IDs；已验证 HF 与 GGUF tokenizer ID 一致。
- 两进程共享检索缓存，通过文件锁串行完成短时检索及缓存写入，防止同时写同一临时文件；模型推理请求仍可并发。
- 合并时验证每个报告只出现一次，模型、嵌入、实现身份相同，按原始清单位置重排。失败计 0；未处理样本存在时主指标为 null。

推理后端改变可能带来数值与输出差异，所以修订版重新运行固定 500 条，不能与旧 Transformers 112 条混合作为同一正式实验。GPU 加速比尚待实测，不能仅据“2 slots”认定翻倍。

## 验证

22 项单元测试通过，覆盖恢复边界、非法路径不读文件、模型纠正而非自动过滤、约束解码进入请求与断点身份、上下文预算、所有候选保留、断点恢复、分区合并及逐请求随机种子。

对旧 5 条 EAR 溢出样本使用真实上下文和实际 tokenizer 做预算检查：原先 31,998–40,223 输入 tokens，分批比较后各样本最大输入分别为 17,823、15,414、16,347、20,142、21,636 tokens，全部保留五个文件。该项使用合成排列测试算法，不是新的 LLM 推理或准确率结果，见 `results/blagent/recovery-20261118/overflow-preflight/summary.json`。

CPU 服务另运行两条旧 SAS 无效输出（原位置 2、110）的真实模型纠正，验证服务、tokenizer 和两个请求 slot；不读取 gold，不评估准确率，不并入正式结果。第一轮仅提示词纠正仍有一条失败，保留在 `results/blagent/recovery-20261118/cpu-probe/`。加入候选键名约束后，两条都在一次真实模型纠正后通过，见 `results/blagent/recovery-20261118/cpu-probe-v2/summary.json`。这只验证 SAS 纠正步骤，完整 19 条流水线仍待 GPU 回归。

GPU 回归清单固定为**全部 16 条旧失败 + 前三条成功对照**（19 条），仅用于检验实现和恢复行为，不能拿其指标当无偏精度结论。通过回归及审计后，以同一新配置重跑原来的 500 条清单。

## 自动继续

本次检测到另一项 `recall_full/20261118_swe2294_v2` 实验使用两张 GPU，每卡仅剩约 2–3 GB。首次 GPU 服务加载因此失败；没有停止或修改该实验。

资源队列每 30 秒检查一次，要求两卡各有至少 18000 MiB 空余、连续三次满足，才启动服务。等待期间保留冻结配置／代码哈希；如果排队期间实现被修改，拒绝悄悄执行新版本。队列先跑 GPU regression19，若全部成功且审计通过，自动跑 full500；否则留下诊断状态。结束后只关闭本队列创建并校验身份的服务。

已于 2026-11-18 14:55（北京时间）在 tmux `irbl-blagent-recovery-v2` 启动队列。启动状态为 `waiting_for_gpu`，两卡分别空闲 2173、1324 MiB；尚未开始 GPU 回归或 full500。CPU 验证服务已关闭，队列要求新建自己的 GPU 服务，避免误用 CPU 服务或关闭其他进程。

### GPU0 单卡继续（2026-11-18 15:32）

用户通知 GPU0 空闲后，实测 GPU0 空余 19652 MiB、GPU1 空余 210 MiB。已停止仍在等待的旧双卡队列，状态记为 `superseded_by_gpu0`；新队列在 tmux `irbl-blagent-gpu0-v2` 中运行，输出为 `results/blagent/recovery-20261118/gpu0/`。

服务及两个报告工作进程都设置 `CUDA_VISIBLE_DEVICES=0`。由于单卡剩余显存不足以同时容纳 BF16 全模型、两个 32K F16 KV cache 和检索编码器，固定将 18 层模型卸载到 GPU，其余在 CPU 执行；仍保留两个并发请求、每请求 32768 tokens 和未量化权重。两个检索编码器改为单卡 BF16，并使用新的缓存身份，避免把旧双卡缓存伪装成同一运行配置。具体配置为 `configs/experiments/blagent_recovery19_gpu0_v2.json` 与 `configs/experiments/blagent500_gpu0_v2.json`；请求超时放宽到 1800 秒以适应部分 CPU 计算。

GPU0 队列只检查 GPU0 的容量，先执行 19 条回归，通过审计后自动执行固定 500 条。该运行独立保存硬件分配和新配置，不与旧结果混合。实际启动、进度及失败状态以 `gpu0/queue_status.json` 和阶段的 `automation_status.json` 为准。

15:34 GPU0 服务加载成功，显存占用实测 13148 MiB，两个 slot 均为 32768 tokens，19 条回归任务已启动。已核对服务及两个工作进程的实际环境均为 `CUDA_VISIBLE_DEVICES=0`。22 项单元测试、GPU 索引选择检查，以及 19/500 条配置预检通过。

```bash
.venv/bin/python -u scripts/experiments/queue_blagent_recovery.py \
  --output results/blagent/recovery-20261118/gpu0 \
  --visible-gpus 0 --gpu-layers 18 --minimum-free-mib 18500 \
  --regression-config configs/experiments/blagent_recovery19_gpu0_v2.json \
  --full-config configs/experiments/blagent500_gpu0_v2.json
```

```bash
# 转换／校验 BF16 模型；源模型与固定 llama.cpp 工具已在本机
.venv/bin/python scripts/datasets/prepare_blagent_server.py

# 资源队列（建议在 tmux 中运行）
.venv/bin/python -u scripts/experiments/queue_blagent_recovery.py \
  --output results/blagent/recovery-20261118/gpu
```

若已手动启动通过身份校验的 GPU 服务，也可直接运行：

```bash
.venv/bin/python scripts/experiments/run_blagent_parallel.py \
  --config configs/experiments/blagent_recovery19_v2.json \
  --output results/blagent/recovery-20261118/gpu/regression19

.venv/bin/python scripts/experiments/run_blagent_parallel.py \
  --config configs/experiments/blagent500_v2.json \
  --gate results/blagent/recovery-20261118/gpu/regression19 \
  --output results/blagent/recovery-20261118/gpu/full500
```

状态入口为 `gpu/queue_status.json`；启动后各阶段还有 `automation_status.json`、`summary.json`、逐 worker 日志。回归与全量结果独立；完整 500 最终生成 `verification.json` 和与项目已有方法的 `comparison.csv`。

## 双卡全量 SWE 2,294 条（dual-v3）

用户随后要求运行 SWE 全量，并明确 GPU1 的既有约 9.9 GB 服务必须保留。单卡回归于 17:11 完成 19/19，18 成功、1 失败；失败为 Pylint #4421 未读取骨架就提交评分，故原队列没有进入 500 条，正常关闭了自建服务。

修复在作者 Agent 的输出解析器外增加检查：未发生成功骨架读取时，将提前的 Final Answer 转为错误 Observation，交还同一 ReAct 循环。模型自行选择候选、调用工具、重新评分；不自动填充访问记录、不按标签选文件、不增加原 15 轮上限。两项真实 LangChain 循环测试覆盖恢复成功和持续提前作答的预算终止，总计 24 项测试通过。

全量清单冻结在 `configs/experiments/selections/swe_full2294_20261118.json`，直接沿用本地全量召回实验的 2,294 行、12 项目及原始顺序；不筛选失败，不重复拼接旧结果。身份 SHA-256 为 `59d45872c142bbf250a95c9b4fb2ecbdcdd53ac6dca865e5ce05f8c33d52e539`。完成后统一审计全部样本，并报告 all2294、development500、remaining1794 和各项目；remaining1794 不宣称为独立盲测。

新配置 `configs/experiments/blagent_full2294_dual_v3.json` 保留 BF16 权重、两个 32K F16 KV 槽位，全部模型计算层卸载到 GPU（`-ngl 99`），两卡分配比例 `2,1`。检索编码器配置为 `gpu_count=1`，两个工作进程均仅在逻辑 GPU0 加载编码器。启动容量门槛为 GPU0 至少 21500 MiB、GPU1 至少 13000 MiB，保留其他服务。比例是模型分配参数，不是硬性显存配额，需同时核对实际进程占用。

先执行三个固定操作回归：Pylint #4421（提前作答）、Matplotlib #23203（旧 EAR 溢出）、Astropy #12880（正常对照），通过后自动开始 full2294。正式运行与回归均全新推理，旧单卡 19 条不混入全量。后台会话 `irbl-blagent-full2294-v3`，状态 `results/blagent/recovery-20261118/dual-v3/queue_status.json`。

### EAR 路径约束补充（dual-v4，当前运行）

dual-v3 的三个回归完成，2 成功、1 失败：Pylint #4421 已成功调用骨架工具，但 EAR 连续输出候选之外的 `pylint_quotes/checker.py`。因此保留 v3 完整失败记录，并为 EAR 增加候选排列约束。约束仅允许输入 1–5 个文件的全部排列（最多 120 种），不指定正确顺序；模型仍根据原报告和证据选择顺序，不能新增、遗漏或重复文件。解码 schema 连同请求落盘，审计重新核验排列集合和候选来源。

当前配置改为 `configs/experiments/blagent_full2294_dual_v4.json`、`configs/experiments/blagent_recovery3_dual_v4.json`，后台会话 `irbl-blagent-full2294-v4`，输出 `results/blagent/recovery-20261118/dual-v4/`。24 项测试再次通过。模型／双卡比例／完整数据清单不变，仍先三个操作回归后进入全量。

全 GPU 服务已实测启动成功：生成服务 GPU0 16630 MiB、GPU1 9076 MiB；GPU1 当时剩余 5268 MiB，另两个既有进程保持运行。编码器加载后 GPU0 剩余约 3.6 GiB，GPU1 剩余约 5 GiB。实际生成一次约 45 tokens/s，但不能把单次生成速度当作全量样本吞吐。显存记录见 `dual-v3/gpu_allocation_startup.json`。

**2026-11-18 17:28（北京时间）：dual-v4 三条回归全部成功，独立审计通过；全量 2,294 条已自动启动。** 正式状态为 `dual-v4/full2294/automation_status.json`，队列为 `running_full2294`。本次真实解码约束探针也通过，见 `results/blagent/recovery-20261118/ear-schema-probe.json`。该记录表示全量已启动，不表示已完成；最终成绩以完整运行和审计为准。
