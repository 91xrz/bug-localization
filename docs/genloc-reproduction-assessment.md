# GenLoc 本地模型复现可行性评估

核查日期：2026-11-20。结论：**建议接入，工程难度中等；可以完成全本地推理，但需要明确标为本地模型适配版。** 本页记录最初的论文、官方源码、本机资源与评估协议审查，以及两个纯 CPU 边界复核。后续本地实现、运行状态与结果见 [GenLoc 六项目小批](genloc-pilot.md)。

论文为 *Towards Explorative IRBL: Combining Semantic Retrieval with LLM-driven Iterative Code Exploration*，方法名 **GenLoc**，[ISSTA 2026 正式会议页面](https://conf.researchr.org/details/issta-2026/issta-2026-research-papers/60/Towards-Explorative-IRBL-Combining-Semantic-Retrieval-with-LLM-driven-Iterative-Code)已列出该工作。本次采用 [arXiv v3（2026-04-21）](https://arxiv.org/html/2508.00253v3)，其 Python 实验与模型替换实验不能用旧版描述代替。官方仓库审查固定在 [`92e4dd4aea0b35f31b80f160e1f59e643a81a71d`](https://github.com/seal-hub/Towards-Explorative-IRBL/tree/92e4dd4aea0b35f31b80f160e1f59e643a81a71d)。

适合加入近期强基线的原因是：它同样以报告和源码为输入、输出文件排名，而且能通过实际读取方法体来扩展初始检索范围。这能检验当前项目的功能推测、双路检索和重排相对于主动探索的效果。论文在 SWE-bench Lite 300 条上比较了 CoSIL、LocAgent，使用相同 GPT-4o-mini；这支持其作为相关强基线，但不能据此认定其优于所有后续方法。[论文 §4.2、§5.3](https://arxiv.org/html/2508.00253v3#S5.SS3)

**需要保留的完整流程**

1. 解析方法体，构造带文件路径的代码块，进行语义检索。
2. 向模型提供初始候选获取、文件名搜索、方法名搜索、方法签名读取、方法体读取五个工具。
3. 保留模型逐轮选择工具、观察结果、更新候选的过程。搜索范围是允许访问的整个源码快照，不能限制在初始 Top-50。
4. 保留错误参数反馈、近似方法签名恢复，以及最终路径校正；最终输出带理由的文件排名。

无需训练 GenLoc 专用权重，也无需构建测试执行环境。官方已经提供 Java、Python 实现及探索提示词，主要工作集中在模型接口、历史源码接入、输出审计和评估。具体流程见[官方 Python 源码](https://github.com/seal-hub/Towards-Explorative-IRBL/tree/92e4dd4aea0b35f31b80f160e1f59e643a81a71d/source-code/GenLoc-Python)。

| 条件 | 官方默认设置 | 本地处理建议 |
|---|---|---|
| 生成模型 | GPT-4o-mini；另测 GPT-5.1 | Qwen3-8B BF16，沿用当前项目模型，单独验证工具调用 |
| 嵌入模型 | text-embedding-3-small；论文另测 CodeXEmbed 400M | 优先考察 CodeXEmbed 400M；共同 Qwen embedding 可作为另列的受控版本 |
| 切块 | 方法体拼接后切分，300 tokens，零重叠，再添加路径前缀 | 保留作者表示与切块；固定 tokenizer，不能直接复用 AST64 向量 |
| 初始候选 | Top-300 代码块去重后取至多 50 文件 | 保留此规则，不能误写成全库精确 Top-50 文件 |
| 探索轮数 | 提示词最多 10 轮；发布代码第 9 次请求强制回答 | 记录差异，预先冻结采用的规则 |
| 输出 | Top-10 文件及理由 | 校正后按最终路径去重，使用共同评分器 |
| 随机性 | 温度默认 1.0，运行三次取平均 | 先检查工具稳定性，再冻结采样设置；正式运行保留三次独立种子 |

300-token、Top-50、温度和三次均值来自[官方配置说明](https://github.com/seal-hub/Towards-Explorative-IRBL/blob/92e4dd4aea0b35f31b80f160e1f59e643a81a71d/docs/implementation-and-default-configurations.md)；实现细节分别见 `utils.py:16`、`file_processor.py:43`、`collection_handler.py:73`、`file_data_processor.py:19`、`bug_report_processor.py:191`。轮数差异中，代码以零起始计数 8 执行 `tool_choice="none"`，正常返回时即第 9 次调用结束；一轮可以包含多个工具调用，不能把轮数当作工具总数。

**全本地模型的具体选择**

建议优先准备 `GenLoc-local (Qwen3-8B + CodeXEmbed-400M)`，保持作者探索流程和代码表示；如需单独控制检索模型差异，再运行 `GenLoc-local (Qwen3-8B + Qwen3-Embedding-0.6B)`。若进一步改为共同 AST64，名称应注明 shared-AST64，并单列为检索替换实验。

CodeXEmbed 有论文依据：v3 §5.6/Table 13 已比较 400M 版本，但只在 GHRB 上进行，且生成模型仍为 GPT。可部署候选是 [Salesforce/SFR-Embedding-Code-400M_R 官方权重](https://huggingface.co/Salesforce/SFR-Embedding-Code-400M_R)，其模型卡提供本地加载代码。**GenLoc 论文未给确切 checkpoint ID/revision，因此该选择是依据官方模型家族作出的对应，尚未确认就是论文实验的同一权重版本。** 当前 GenLoc 发布代码只有 OpenAI/GTE 等分支，没有 CodeXEmbed 接入；GTE 分支是 `Alibaba-NLP/gte-modernbert-base`，不能混称为 CodeXEmbed。[论文 RQ6](https://arxiv.org/html/2508.00253v3#S5.SS6)、[发布的 embedding_handler.py](https://github.com/seal-hub/Towards-Explorative-IRBL/blob/92e4dd4aea0b35f31b80f160e1f59e643a81a71d/source-code/GenLoc-Python/embedding_handler.py)

本机现场核查为两块 RTX 4090，各 24,564 MiB。已经有 Qwen3-8B 原始权重、BF16 GGUF 及 Qwen3-Embedding-0.6B；未见上述 CodeXEmbed 目录。现有 BLAgent 已采用 Qwen3-8B、32K 上下文和本地 llama.cpp，证明该模型规模有部署基础。配置见 [blagent_full2294_dual_v4.json](../configs/experiments/blagent_full2294_dual_v4.json)，运行经验见 [blagent-recovery.md](blagent-recovery.md)。GenLoc 的长方法体、多轮历史及不同并发设置仍需单独测显存和上下文峰值。

已有 [blagent_runtime.py](../src/bug_localization/blagent_runtime.py) 可复用模型身份核验、日志、token 与耗时记录，但其文本生成接口不等于 GenLoc 的原生工具协议。GenLoc 同时使用 `tools`、`tool_choice`、结构化 `tool_calls` 和严格 JSON schema；接入本地模型时要验证工具模板、参数解析、工具结果回填和最终排名格式，不能仅替换 API 地址就认定已跑通。[作者调用代码](https://github.com/seal-hub/Towards-Explorative-IRBL/blob/92e4dd4aea0b35f31b80f160e1f59e643a81a71d/source-code/GenLoc-Python/bug_report_processor.py#L191)

**已经确认的复现问题**

| 问题 | 证据及影响 | 接入处理 |
|---|---|---|
| Python 依赖未完整列出 | `file_parser.py:1` 导入 `tree_sitter_language_pack`，requirements 未列该包 | 隔离安装并冻结兼容版本；不覆盖主环境 |
| 云端配置写死 | `main.py:25` 选 OpenAI embedding，`bug_report_processor.py:203` 写死 GPT，两个客户端都读取 API key 文件 | 生成和嵌入分别接本地后端，检查没有云端调用 |
| 路径校正后可能重复计分 | `post_processor.py:68–99` 只对原始路径去重；`evaluation_metric_calculator.py:85–90` 对重复相关路径累计 AP | 保留校正记录，最终路径去重后统一计分 |
| 候选池存在差异及历史依赖 | `file_processor.py:26–61` 初始扫描跳过任意层 test* 目录及无函数文件；`:85–89` 增量过滤只检查整个路径开头 | 按每个修复前快照冻结共同文件清单，明确共同候选适配差异 |
| 长报告检索只用首段 | 报告按 8191 tokens 切分，但 `file_data_processor.py:22` 只读取第一段检索结果 | 记录原规则；改变报告覆盖范围必须列为差异 |
| 同签名方法覆盖 | `file_parser.py:11–31` 用不含类名的函数签名作字典键 | 保留作者行为并披露，或另列修正版；不得静默增强 |
| 失败样本可能消失 | `bug_localizer.py:38–39` 不消费线程结果；后处理异常跳过，评估分母为读入结果数 | 使用冻结样本清单，所有失败计零；支持断点和逐例状态 |

以上为对固定版本的源码审查；[后处理源码](https://github.com/seal-hub/Towards-Explorative-IRBL/blob/92e4dd4aea0b35f31b80f160e1f59e643a81a71d/source-code/GenLoc-Python/post_processor.py)、[评分源码](https://github.com/seal-hub/Towards-Explorative-IRBL/blob/92e4dd4aea0b35f31b80f160e1f59e643a81a71d/source-code/GenLoc-Python/evaluation_metric_calculator.py)、[文件处理源码](https://github.com/seal-hub/Towards-Explorative-IRBL/blob/92e4dd4aea0b35f31b80f160e1f59e643a81a71d/source-code/GenLoc-Python/file_processor.py)可逐项核对。

两项纯 CPU 构造复核：模型原始排名 `a.py, pkg/a.py`，候选与正确文件只有 `pkg/a.py`，作者后处理返回两次 `pkg/a.py`，随后作者评分打印 MAP@10=2.0000；`is_test_path("pkg/tests/test_demo.py")` 返回 False。这证明边界实现存在问题，**不代表已经证明论文发布成绩受到这些问题影响**；本轮未审计所有作者预测。

**本项目的公平比较协议与建议顺序**

论文的 Accuracy@1/5/10、MRR@10、MAP@10 定义与项目主指标基本一致，AP 分母同为全部相关文件数，便于统一计分。数据清单不同：作者使用 SWE-bench Lite 300 条，本项目使用 SWE 格式化版 2,294 条及其固定子集，不能混用原论文数字。[论文 §4.4](https://arxiv.org/html/2508.00253v3#S4.SS4)、[本项目评价](evaluation.md)

1. 固定作者代码版本，接入本地生成和嵌入，保留五工具、方法体证据、候选外搜索、恢复及最终排名。先用数条跨项目开发样本验证真实工具调用和稳定终止。
2. 小批记录召回覆盖、工具参数有效率、最终排名有效率、首次正确文件是否来自初始候选外、调用数、token、上下文峰值和耗时。效果之外也验证是否实际运行了探索流程；不以定位成绩挑选成功案例。
3. 将配置冻结后接当前共同清单：只用报告 `body` 和修复前源码，不读取 PR 标题、补丁或 gold；搜索限于共同允许的源码集合。方法为空的文件仍应在共同清单中有明确状态，不按正确答案排除。
4. 每次独立运行用全部冻结样本作分母，失败计零，先对校正后路径去重，再计算 Top-k、MRR@10、MAP@10、Recall@10；保存原始响应与全部工具轨迹。三次分别计分后求均值和标准差，不能取三次命中并集当平均成绩。
5. 主表标明 `GenLoc-local` 的生成模型、嵌入、表示、候选和预算差异。原论文结果单列作参考。现有 500 条已用于方法开发，全量 2,294 条也不能自动宣称为独立盲测；需要独立结论时另用预先冻结、未参与调参的样本。

可复用的工程组件包括 [snapshot.py](../src/bug_localization/snapshot.py) 的历史快照、[embeddings.py](../src/bug_localization/embeddings.py) 的本地编码与缓存机制、[batch.py](../src/bug_localization/batch.py) 的结果组织，以及 [bm25.py](../src/bug_localization/bm25.py) 的共同评价函数。代码表示和 tokenizer 不同时必须重建向量，不能把现有 AST64 缓存当作作者 300-token 检索结果。

工作量主要是本地工具协议适配、作者实现差异审计和重新建立向量缓存，不涉及训练。作者报告平均 47.7 秒/bug，来自其云端实验，**不能用来预测本机总时长**；先测本地小批，再分别估算索引构建和三次全量推理。来源：[官方时间与费用说明](https://github.com/seal-hub/Towards-Explorative-IRBL/blob/92e4dd4aea0b35f31b80f160e1f59e643a81a71d/docs/time-cost-analysis.md)。
