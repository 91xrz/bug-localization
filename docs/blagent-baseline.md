# BLAgent 复现与对比

后续进展：[失败修复与共享模型并发](blagent-recovery.md)。初版运行在 112/500 时因连续失败停止；修订版新增有界纠正、超长上下文分批重排和双请求服务，使用新目录重新评测，不能将初版启动记录理解为已完成。

更新：2026-11-18。结论：**适合作为近期强基线，但必须区分作者已发布结果核验、原配置重新推理和本项目同协议适配实验。** 当前新增完整 Agent 流程，原有 `blagent_ast` 仍只是分块路线。

论文：[BLAgent: Agentic RAG for File-Level Bug Localization](https://arxiv.org/html/2605.17965v2)，作者注明已被 TOSEM 接收，DOI `10.1145/3830238`。[官方代码](https://github.com/afifaniks/BLAgent)固定在 `9f91f63e50aec8ad87a11e11148b086dfcd5e66d`。不能把“论文报告优于所比较基线”写成截至当前所有方法的全局 SOTA。

## 已完成的论文结果核验

下载作者 `results/rq1-1/blagent_gpt-oss.json`，按 Git blob `767c22721cc55882cecfdb16e81d1932e6ca07df` 校验，独立复算全部 300 条；索引恰好覆盖 0–299，没有重复。

| 作者 GPT-OSS-120B 预测 | MRR | Top-1 | Top-3 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|
| 独立复算 | 0.851407 | 78.6667% | 92.3333% | 93.3333% | 94.3333% |

与 README 四位小数结果全部一致。验证文件在 `results/blagent/published-verification.json`，入口为 `scripts/analysis/verify_blagent_published.py`。这是**已有预测的完整性及计分核验，不是重新调用 GPT-OSS-120B 的实验**。本地没有该生成模型、Nomic 嵌入模型及官方 Lite 数据快照，因此尚未完成原配置重跑。

## 作者材料存在的实际问题

- `pipelines/query_transformation.py` 使用未定义的 `instance_id`。
- `pipelines/agentic_pipeline.py` 定义 `--results_path`，却读取 `args.results_file`。
- 论文文字说合并后候选限制为 15；`SWEBenchRankerAgent.run()` 实际是两路各取 15，按 T0→T1 顺序去重，最多 30。适配版遵循代码。
- 论文及工具描述提及 docstring；发布的 `extract_signature_tree()` 实际只输出简化类、函数和参数名，不输出 docstring。适配直接调用该函数，不自行增强。
- `extract_patch_file_path()` 只返回补丁中第一个文件，作者评测使用单个 `patch_file`。发布预测的排名长度也不是固定 10，实测为 7–16。不能把其 MRR 与本项目全部修改文件口径的 MAP/MRR 直接横向比较。
- 作者 EAR 有超长上下文 tournament 分支；当前适配为正常单次 EAR，超过本地实际 tokenizer 预算则明确失败。

入口适配绕过上面两处 CLI 错误，作者源码保持不变；每次运行核验 commit、源码 diff 和 SHA-256。作者没有完全锁定所有依赖，适配运行依赖另行记录。

## 完整最小流程

1. 从冻结清单读取原报告正文、修复前 commit 和共同候选文件。推理函数仅接收报告与候选源码，不传 gold、patch、PR 标题或修复后源码。
2. 调用作者 `QueryAugmentor` 的结构 T0、行为 T1 提示词，现场生成两条查询。
3. 使用共同 Qwen AST64 分块、结构前缀、512-token 嵌入预算及本地向量缓存。逐查询对全部块做精确余弦评分，按作者策略取前 1000 块、以每文件最高分聚合，再取各路前 15。
4. 直接调用作者 LangChain ReAct Agent 与 `ReadFileSkeleton`，由模型决定读取文件。工具被约束为只可读取召回候选，并记录实际访问，不能用人工静态排序冒充 Agent。
5. 保留作者 SAS 得分排序的同分顺序；以 T0 检索每个前五文件的五个证据块，调用作者 `CodePruner`，再执行作者 EAR 提示词。EAR 只重排前五，保留其后排名。
6. 最后才用全部修改文件计算统一 MAP@10、MRR@10、Top-1/5/10、Recall@10。失败按 0 计入完整分母。

当前接口按作者代码接收非空、合法的候选得分字典，允许排名长于或短于 10，最终统一截断到 @10 计分。EAR 要求是给定前五的完整排列；不自动补充、不移除无效路径、不偷偷回退到原始检索。另要求至少发生一次实际骨架读取。它比原代码的宽松 JSON 接收更严格，失败率须一并报告。每条最多 20 次模型调用，SAS 最多 15 步；连续三条失败会停止，避免系统问题浪费全量计算。

## 与原配置的差异

| 条件 | 作者配置 | 本项目适配 |
|---|---|---|
| 数据 | SWE-bench Lite 300 | 本地 SWE 格式化版；小批 Requests 3，正式固定跨项目 500 |
| 生成模型 | GPT-OSS-120B；另测 Qwen3-32B、Claude | Qwen3-8B BF16，关闭 thinking，双卡分配 |
| 推理后端 | Ollama／Anthropic | 本地 Transformers，逐调用固定随机种子、保存原始响应 |
| 检索模型 | Nomic embed text v1，8192 上下文 | 共同 Qwen3-Embedding-0.6B，512 tokens |
| 切块 | LlamaIndex CodeSplitter，路径包装 | 共同 AST64，1500 字符预算、64-token 结构前缀 |
| 向量库 | Chroma 默认距离 | NumPy 精确余弦；Qwen 向量 L2 归一化 |
| 候选 | Python，过滤测试文件 | 共同 Python 候选，包含测试文件，不按标签筛选 |
| 生成预算 | 原模型大上下文与默认输出上限 | 32768 总上下文、每次最多 2048 新 tokens |
| 温度 | 查询 0.7、SAS 0.5，EAR 模型默认 | 查询 0.7、SAS 0.5、EAR 0；top-p 0.9、top-k 40 |
| 标签／评价 | 首个补丁文件，MRR、Top-k | 全部修改文件，统一 @10；失败计 0 |

因此正式结果应命名 **BLAgent author-agent adaptation (Qwen3-8B + shared Qwen AST64)**。原项目功能推测也用 Qwen3-8B，故适合检验不同流程在共同模型资源下的效果；不能用该小模型适配失败或较低分声称打败论文完整 120B 系统。主论文比较建议同时列明原始文献结果的独立表格和同协议适配结果表格。

## 运行

先准备固定源码与隔离依赖，复用已有 `.runtime/rrl-deps`、`.runtime/igsil-deps` 及主 `.venv`，不替换主环境。权重、源码快照和数据使用当前本地资产。

```bash
.venv/bin/python scripts/datasets/prepare_blagent.py
.venv/bin/python scripts/retrieval/run_blagent.py \
  --output results/blagent/requests3-smoke --check-only
.venv/bin/python -u scripts/retrieval/run_blagent.py \
  --output results/blagent/requests3-smoke
.venv/bin/python scripts/analysis/audit_blagent.py \
  --run results/blagent/requests3-smoke
```

大规模入口要求至少三条真实样本全部成功、独立审计通过，且科学配置和实现指纹与小批一致；使用原有固定 500 条清单，不重新选样、不根据指标筛选。

```bash
.venv/bin/python -u scripts/experiments/run_blagent500.py \
  --smoke results/blagent/requests3-smoke \
  --output results/blagent/crossproject500
```

最终自动核对共同候选、原始正文和标签，重算指标，并给出合计 500、原有 180、新增 320 与原报告＋重排、功能推测＋重排、C3 和 A+B+C RRF 的配对对比，并输出 `comparison.csv`。运行未完成时主指标为 null，不能将当前成功子集当最终成绩。

同输出目录支持断点恢复，拒绝配置、源码、模型、清单身份变化；已记录的失败不自动重试覆盖，修复后另开运行目录。每个运行目录有独占锁，避免重复进程写坏记录。

主要产物：`provenance.json`、`effective_config.json`、`selection.json`、`summary.json`、`samples.json`；逐条 `inputs.json`、`call_*.json`、`retrieval.json`、`tool_calls.json`、`phase1.json`、`phase2_context.json`、`result.json`／`error.json`；最终 `verification.json`。500 调度状态另见 `automation_status.json`。

## 实测状态

- 作者已发布的 300 条预测结果核验通过。
- 8 项单元测试通过：候选合并、得分／排列有效性、精确源文保留、检索聚合与证据来源、两阶段排名尾部保留、工具越界、失败分母、完成后的断点恢复及结果篡改拒绝。
- 第一次小批在模型加载时缺失 `psutil`，三条均失败，未启动大规模。已在隔离目录补齐，记录保存在 `results/blagent/requests3-smoke-20261118/`。
- 第二次小批 `results/blagent/requests3-smoke-v2-20261118/` 的前两条完整成功，第三条被接入层多余的“恰好 10 个得分”校验拒绝。已按作者代码改为允许可变长度；这属于接入修正，不根据定位指标修改模型输出。
- 修正校验后的小批 `results/blagent/requests3-smoke-v3-20261118/` 逐字复用第二次运行全部已完成的模型响应，重新执行工具与解析，第三条 EAR 现场补跑；`replay_source.json` 记录响应来源及哈希，审计校验未改动。完成情况以该目录 `summary.json`、`verification.json` 为准。


### 已验证的小批结果与正式运行

修正后的 Requests #1142、#1327、#1339 全部成功，3/3 通过独立审计。累计 30 次真实生成调用（输入 70,079 tokens、输出 2,134 tokens），18 次骨架工具调用；Top-1 为 3/3、MAP@10 为 0.916667。样本很小，仅证明流程可执行，不作为效果结论。v3 使用 29 次已有响应和 1 次新 EAR 调用，来源逐项哈希校验，不把回放耗时当真实模型延迟。

冻结实现回归检查在 `results/blagent/requests3-final-20261118/`，复用上述全部 30 次未经修改的响应，只重执行检索、工具、解析和计分；其后自动执行固定 500 条新推理。正式 500 不复用这三条 Requests 预测，清单是此前九项目固定样本，与小批项目分离。

正式运行目录：`results/blagent/crossproject500-20261118/`。后台会话：`tmux attach -t irbl-blagent-500`。调度日志：`results/blagent/crossproject500-20261118.launcher.log`。该会话先完成冻结实现回归，再调用带审计门槛的大规模入口。

500 条完成状态以 `automation_status.json` 和 `summary.json` 为准；未完成时不报告全量成绩。全部结束后自动生成 `verification.json` 和 `comparison.csv`，若连续三条失败则停止并保留诊断。使用现有两卡的空余显存，未停止原有缓存任务；共享资源计时不能当公平延迟基准。

2026-11-18 11:44（北京时间）：冻结实现回归 3/3 成功、独立审计通过；500 条正式实验已启动，首条为 Astropy #12880。此记录表示启动状态，不代表 500 条已完成。
