# GenLoc 本地模型小批实现

2026-11-20：已接入 GenLoc 的完整检索与五工具探索流程，正在执行六项目小批。完成状态以 [summary.json](../results/genloc/pilot6-v3-20261120/summary.json) 和独立审计 [verification.json](../results/genloc/pilot6-v3-20261120/verification.json) 为准。本轮为单种子开发冒烟实验，不能作为完整基准或原论文分数的复现。

前置材料核查见 [GenLoc 复现评估](genloc-reproduction-assessment.md)。作者代码固定为 `92e4dd4aea0b35f31b80f160e1f59e643a81a71d`，直接调用作者方法解析、五个探索函数和路径恢复算法；作者源码保持原样，通过 SHA256 核对。

本地方案为 **GenLoc-local：Qwen3-8B BF16 + CodeXEmbed-400M**。检索保留方法体拼接、cl100k_base 300-token 零重叠分块、块前路径、Top-300 块去重后至多 50 个文件。模型可以搜索整个共同候选池、读取方法签名及方法体、根据结果继续调用工具；调用不限定在初始候选内。

| 设置 | 本轮配置 |
|---|---|
| 数据 | SWE 格式化版中六个预先指定项目，各取原始数据顺序第一条 |
| 报告 | 仅原始 body，不使用独立 PR title、补丁或标签 |
| 源码与文件池 | 各样本 before_fix_sha；共同 Python 文件，包含测试文件与无函数文件 |
| 生成模型 | Qwen3-8B BF16，GPU0，llama.cpp 原生工具调用，关闭 thinking |
| 上下文与输出 | 32768 总 tokens；每次最多 4096 输出 tokens；超限明确失败 |
| 探索预算 | 遵从发布代码：第 1 次 required，第 2–8 次 auto，第 9 次 none |
| 采样 | temperature=1.0，不额外做 top-k/top-p 截断；基准 seed=42，逐请求派生种子 |
| 嵌入 | Salesforce/SFR-Embedding-Code-400M_R，GPU1，float32，CLS + L2，batch8 |
| 向量排序 | NumPy 全量精确余弦；稳定按块顺序处理同分 |
| 输出与计分 | 作者路径恢复后再次去重，最多十个真实文件；失败计零 |
| 重复次数 | 1；原论文三次均值尚未执行 |

无函数文件保留在共同工具池中，可以按文件名搜索并返回空方法列表；不会为这些文件额外发明作者没有的嵌入表示。测试文件参与检索。二者均属于共同候选协议适配，与作者默认过滤不同。

生成阶段只接收报告与源码。标签仅供末端评价使用。保存所有调用请求、原始模型响应、参数、工具返回、最终原始文本、路径纠正及精确排序；断点恢复核对配置、源码、模型和选择清单身份，不覆盖已记录失败。

**已解决的运行问题**

- llama.cpp 同时接收工具调用与严格 JSON schema 时，schema 会覆盖工具约束，导致 `required` 仍直接返回答案。真实探针确认后，探索轮使用原生工具协议，提示词用两字段 JSON 数据示例说明最终输出结构，强制最终轮才启用 schema。提前结束也必须通过严格解析，不能静默改成普通聊天。
- CodeXEmbed 作者自定义代码依赖 Transformers 4 的接口，当前主环境为 Transformers 5；配套的旧 Hugging Face Hub 又与主环境 Datasets 5 冲突。已在 `.runtime/genloc-deps` 中隔离固定兼容版本，主环境恢复原状。
- 对校正后的文件路径再次去重，防止作者后处理的重复路径造成 AP 重复计分。统一评测以完整冻结清单作分母。
- 用每个精确历史快照重建文件池，避免作者增量过滤测试目录的不一致。向量缓存按模型身份和文本内容保存，读取时核验校验和。

初版运行 [pilot6-20261120](../results/genloc/pilot6-20261120) 在两条报告上均完成实际工具调用，但把 JSON schema 外壳写进最终答案，严格校验失败后自动停止，剩四条未运行。已在修改前完成独立审计并保存原始源码快照。v2 改用 JSON 数据示例后，首轮出现超过 3500 tokens 的长生成，尚未返回任何完整响应时中止，记录在 [pilot6-v2-20261120/interruption.json](../results/genloc/pilot6-v2-20261120/interruption.json)，没有完成案例或分数。本地 Qwen 模板对 required/auto 生成相同的提示，required 只在语法层阻止提前结束；v3 因此在报告后补充固定首轮协议，明确先发原生工具调用并等待结果，不指定工具、参数或候选。v3 使用相同六条清单重新推理；各版记录独立保留。

CodeXEmbed 固定模型 revision 为 `cb950dc80d677c6fdc00f56c8ddd20ca2642c59e`，自定义模型代码 `Alibaba-NLP/new-impl` 固定为 `40ced75c3017eb27626c9d4ea981bde21a2662f4`。论文只称 CodeXEmbed 400M，未公开确切 checkpoint 标识，因此不宣称已核实与论文使用完全相同的权重。实际加载不联网获取模型代码。

隔离环境的主要版本为 Transformers 4.57.3、Tokenizers 0.22.1、Hugging Face Hub 0.36.0、Datasets 4.4.1、tree-sitter 0.25.0、tree-sitter-language-pack 0.9.0。完整记录见 [.runtime/genloc-deps/environment.json](../.runtime/genloc-deps/environment.json)，每次运行 provenance 同时保存该记录。

**验证与运行**

22 项行为检查通过，覆盖候选外搜索、方法体证据、路径限制、去重、失败分母、原生工具消息、上下文溢出和输出截断。CodeXEmbed 官方模型卡示例在 GPU1 的相似度为 74.847473、65.392662，与公布数值最大差约 0.000023；缓存重读、归一化和明确拒绝超长输入通过。Qwen 真实探针完成 required 工具调用、工具结果回填与最终 JSON 输出。

准备和运行入口如下。所有路径相对于项目根目录；模型服务端口已占用时，准备脚本会停止而不替换已有进程。

```bash
.venv/bin/python scripts/datasets/prepare_genloc.py
.venv/bin/python scripts/datasets/prepare_genloc_server.py --start
.venv/bin/python scripts/retrieval/run_genloc.py \
  --config configs/experiments/genloc_pilot6_v3.json \
  --output results/genloc/pilot6-v3-20261120 --check-only
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python -u scripts/retrieval/run_genloc.py \
  --config configs/experiments/genloc_pilot6_v3.json \
  --output results/genloc/pilot6-v3-20261120
.venv/bin/python scripts/analysis/audit_genloc.py \
  --run results/genloc/pilot6-v3-20261120
```

审计不加载大模型，也不调用推理服务：重新读取原始报告及历史源码，重建作者方法和分块，从带校验的向量缓存重算全量余弦排名，逐次重放工具结果和模型输入，复算所有指标。推理失败但初始检索已完成时，纯检索基线仍按实际排名计分，不跟随代理失败被错误计零。

冻结清单：[genloc_pilot6_20261120.json](../configs/experiments/selections/genloc_pilot6_20261120.json)。配置：[genloc_pilot6_v3.json](../configs/experiments/genloc_pilot6_v3.json)。真实运行产物：[pilot6-v3-20261120](../results/genloc/pilot6-v3-20261120)。模型和接口验证：[setup-20261120](../results/genloc/setup-20261120)。

本轮只用于验证可执行性、过程完整性和发现失效模式。六条都属于开发样本；单次随机采样的效果不应外推到 500/2294 条。正式扩展前需冻结独立评价清单与随机重复策略，并继续明确本地模型、精确检索、共同候选与作者 GPT/Chroma 配置的差异。

**若改用论文数据集评价现有方法**

本轮另做只读核查，未启动新的数据集实验。本地尚未发现官方 SWE-bench Lite 原始快照；已有 RRL 作者发布的 `swe_dataset.csv` 保存 300 个唯一实例及原始字段。以下映射以该参考 CSV 为依据，尚需与官方冻结快照独立核验：

| 对齐项目 | 核查结果 |
|---|---:|
| 通过 PR 编号匹配本地 SWE 格式化版 | 300/300，一对一 |
| base_commit 与 before_fix_sha | 300/300 一致 |
| 原 patch 的文件集合与 updated_files | 300/300 一致 |
| gold 含 test_patch 文件 | 0/300 |
| problem_statement 与本地 body 逐字一致 | 39/300 |
| 忽略大小写和空白后正文一致 | 299/300 |
| 与旧开发 500 条重叠 | 61 条 |
| 本地源码快照及 manifest 已存在 | 300/300，共 297 个唯一快照 |

正文规范化后仍不同的是 `mwaskom__seaborn-2848`。因此这条路线可以复用现成源码和兼容的代码向量缓存，但应按官方原始 `problem_statement` 重新生成查询和运行方法，不能直接从旧 2294 条结果中抽取排名冒充原始 Lite 协议。gold 仅取 `patch`，不加入 `test_patch`。快照存在性已核对，此次映射检查未重新校验全部 297 个源码清单。

现有 [reserve_swe_official_round3.py](../scripts/datasets/reserve_swe_official_round3.py) 记录官方 Lite revision `b0dde1093fe417d83b7184254edf8199c1f0dff5`，可作获取版本的参考；它为训练排除而做的多条件并集匹配不能直接当作严格的一对一评测清单。参考 CSV 在 `.runtime/external-baselines/rrl/bug_localization/datasets/swe_dataset.csv`。

建议后续先冻结官方 Lite 300 条，统一报告、源码、gold 和候选过滤后运行现有方法。与 GenLoc 原论文成绩作文献对照时仍需列出本地模型与 GPT-4o-mini、候选过滤及输出规则的差异；这 300 条也不能称为此前未接触的新盲测集。
