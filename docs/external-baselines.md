# 外部论文对比基线

更新日期：2026 年 11 月 15 日。本页整理本轮筛选的 9 篇论文，记录其公开材料、数据集、适配成本和推荐顺序，供后续实验选型与论文引用使用。

**实际接入三篇：BLAZE、RRL 和 IGSIL；其余论文用于同数据集相关工作与论文结果参考。** 三篇适配版已在相同固定 500 条清单上完成运行，BLAZE 三路线各成功 500 条，RRL 成功 455 条、IGSIL 文件级成功 498 条；与项目方法的结果分别见 [BLAZE/RRL 对比](external-comparison-500.md)与 [IGSIL 对比](igsil-baseline.md)。本页的“复现难度”是材料审查后的判断，不代表已复现论文分数。

BLAZE 的公开权重是编码器，训练程序另有两套残差适配器，发布权重中没有这些参数。因此当前实验命名为 **BLAZE released encoder / author-demo adaptation**，沿用作者公开演示的推理方式；不能称训练全流程或原论文数字的完整复现。RRL 最小测试使用本地 Qwen2.5-Coder-7B Q4_K_M 与 llama.cpp，已完成作者代理流程；与论文 30B/32B、Ollama 配置的差异见 [RRL 接入记录](rrl-baseline.md)。

## 1 筛选标准与推荐顺序

“有 GitHub 仓库”不等于已经满足完整开源与可复现要求。这里分别核对源码、许可证、模型、数据和运行流程；公开代码但未确认许可证，或者默认依赖闭源模型的工作，保留为候选，不能直接写成“完全开源基线”。代码、模型及数据可能使用不同许可，应按各自发布页记录。

| 优先级 | 论文 | 公开材料与当前结论 | 主要适用范围 | 复现难度 |
| --- | --- | --- | --- | --- |
| 实测第一项 | BLAZE，TSE 2025 | 源码和发布编码器可用；Zenodo 包 CC BY 4.0、HF 模型 MIT；训练适配器未随权重发布 | SWE 格式化版、BeetleBox；Ye 需核对版本与划分 | 中：先完成发布编码器适配；已固定模型并验证新旧依赖下编码一致 |
| 实测第二项 | RRL，BoatSE 2026 | Apache-2.0 源码；使用可本地部署的 Qwen 模型与 Ollama；本机已跑通 7B 适配版单条测试 | SWE-bench Lite、LCA，与报告改写设计直接相关 | 中高：30B/32B 模型和多轮代理带来显存、时延与过程复现成本 |
| SWE 相关工作 | BLAgent，2026，TOSEM 已接收 | 作者源码与本地模型配置公开；未确认仓库 LICENSE | SWE-bench Lite，双路检索与代理重排 | 高：完整代理流程和大模型成本；许可需补核 |
| SWE 相关工作 | LocAgent，ACL 2025 | Apache-2.0 源码，专用模型权重公开；训练材料完整性及 32B 权重许可待核 | SWE-bench Lite、LocBench | 中高：需建代码图、运行代理、对齐文件级输出 |
| Bench4BL 相关工作 | BRaIn，ICPC 2025 | 源码、步骤和本地模型方案公开；未确认许可证 | Bench4BL 的作者筛选子集 | 中：Java 数据接近，但反馈、提示词选择与时间划分需复核 |
| Java 经典参考 | BLUiR，ASE 2013 | Bench4BL 提供修改后的源码和可执行文件；明确源码许可未确认 | Java Bench4BL | 中：Eclipse JDT、Indri 5.6 和旧 Java 环境；Python 需额外适配 |
| SWE 相关工作 | Agentless，FSE 2025 | MIT 源码及运行产物公开；官方 GPT/Claude 配置不能算全开放模型方案 | SWE-bench Lite、Verified 的定位阶段 | 中高：需抽取定位阶段并明确模型替换的实验身份 |
| 实测第三项 | IGSIL / CoSIL，ASE 2025 | 作者源码公开；许可证待核；本地 7B 文件级适配在 500 条上成功 498 条 | SWE-bench Lite、Verified，报告文件及函数级指标 | 中高：文件级 Top-5 适配已完成，函数级阶段未运行 |
| LCA 相关工作 | Retrieval-Oriented Code Representations，2026 预印本 | Apache-2.0 源码；发布材料尚未下载试跑 | LCA、SWE-bench Verified | 中高：代码摘要生成、索引和检索/融合重排的完整成本待实测 |

其余论文优先按数据集对齐：SWE 侧引用 BLAgent、LocAgent、Agentless、CoSIL；LCA 侧补充 Retrieval-Oriented Code Representations；Bench4BL/Java 侧补充 BRaIn、BLUiR。这里不把“未开源”“源码可见但许可未核实”“原配置使用闭源模型”混为一谈。参考论文的原始分数单列，只有样本、源码版本和指标均对齐后，才能进入同一实测表。

## 2 论文与公开材料

### 2.1 BLAZE

- **论文：** *Cross-Language and Cross-Project Bug Localization via Dynamic Chunking and Hard Example Learning*，IEEE Transactions on Software Engineering，2025。[DOI](https://doi.org/10.1109/TSE.2025.3579574)。
- **作者材料：** [Zenodo 发布包](https://zenodo.org/records/15122980)包含源码、数据、训练和推理程序及约 440 MB 的权重；发布包标 CC BY 4.0。[Hugging Face 模型](https://huggingface.co/bug-localization/BLAZE)标 MIT。两处许可分别按对应资产记录，不推断为全部第三方依赖的许可。
- **论文数据：** BeetleBox、SWE-bench 和 Ye 数据。当前本机使用 [BLAZE 格式化 SWE-bench](https://huggingface.co/datasets/Partha117/swe_bench_formatted)；BeetleBox 已执行本地完全重复记录去重；Ye 的版本、筛选和提交映射仍有未解决差异，见 [数据说明](datasets.md)。
- **可比性：** 同为报告到文件排名的语义检索，适合检验当前双路语义召回和证据重排相对外部检索系统的效果。作者实现实际使用 CodeSage-small 相关模型与旧依赖，不能用 Qwen 替换编码器后仍称完整 BLAZE。
- **难度与接入：** 中。需要核对动态分块、作者模型、训练或推理权重、块到文件聚合及数据过滤。已有 `references/blaze/` 只保存用户提供的分块/转换参考代码；当前项目的动态块适配不能代替完整 BLAZE。第一项接入的具体状态见 [BLAZE 基线接入](blaze-baseline.md)。

### 2.2 RRL

- **论文：** *Reformulate, Retrieve, Localize: Agents for Repository-Level Bug Localization*，BoatSE 2026（ICSE workshop）。[论文](https://arxiv.org/abs/2512.07022)。应按 workshop 论文引用，不写为 ICSE 主会论文。
- **作者源码：** [gencau/boatse-query-reformulation](https://github.com/gencau/boatse-query-reformulation)，Apache-2.0。作者配置使用 Qwen3-Coder-30B / Qwen2.5-Coder-32B 和 Ollama；具体权重变体、量化与模型标签应按冻结的配置记录，不把同一系列的不同模型视为相同权重。
- **论文数据：** SWE-bench Lite、LCA 150 条。Lite 与本地 SWE 格式化版 2,294 条并非同一评价清单；LCA 需按语言、行 ID、issue 与提交核对。当前本机 LCA 为 Java、Python、Kotlin 各 50 条测试报告，见 [LCA 数据说明](lca.md)。
- **可比性：** 报告改写、检索再定位，与本项目功能假设生成和后续重排形成近期方法对照；检索表示与代理检查方式和当前双路 dense 检索不同。
- **难度：** 中高。除数据适配外，还要固定大模型、提示词、迭代停止条件和调用预算。用更小模型运行应单列为模型替换变体，不能直接当作原论文设置。
- **可执行材料与下一步：** 已核查作者 [README](https://github.com/gencau/boatse-query-reformulation/blob/50903acf3d6124514ff4f4c179cfd67a3e8ea414/README.md) 和 Apache-2.0 LICENSE，冻结源码版本 `50903acf3d6124514ff4f4c179cfd67a3e8ea414`。作者测试环境为 Python 3.11、Ollama 0.11.10，BM25 部分依赖 Pyserini；仓库提供 CSV 数据、模型配置、运行日志和指标脚本。查询抽取后 BM25 的入口为 `extract_lca.py` / `extract_swe.py` 与 `run_bm25_extracted*.py`，代理入口为 `agent_workflow_defined_set.py`。先按本地 LCA 的实例和 base SHA 校对作者 CSV，再恢复完整对应流程；不能只跑其中一个普通 BM25 就称 RRL 完整系统。现已在当前 SWE 格式化版 Requests #1142 跑通 7B 适配版：12 次模型调用、6 个实际查看文件、1 次格式纠正，自检后正确文件排名第 2；详见 [RRL 接入记录](rrl-baseline.md)。这不代表作者 30B/32B 配置或完整基准的复现。

### 2.3 BLAgent

- **论文：** *Agentic RAG for File-Level Bug Localization*，2026，TOSEM 已接收。[论文 v2](https://arxiv.org/html/2605.17965v2)、[DOI](https://doi.org/10.1145/3830238)。引用时保留实际采用的论文版本。
- **作者源码：** [afifaniks/BLAgent](https://github.com/afifaniks/BLAgent)。已确认源码公开，尚未确认仓库 LICENSE，暂不归为许可已核实的完全开源基线。
- **模型与数据：** 作者本地方案包括 GPT-OSS-120B / Qwen3-32B 与 nomic 嵌入；确切组合由具体实验配置决定。论文使用 SWE-bench Lite 300 条。
- **可比性：** 将报告转换为结构与行为查询，双路 dense 检索后合并候选并代理重排，与当前双路设计最接近。需要比较完整查询、召回和重排流程。
- **难度：** 高。大模型部署与多轮代理成本明显高于当前小型重排器；还需补核许可证。仓库内 `references/blagent/` 及 `blagent_ast` 仅为 AST 切块参考和适配，不是该论文完整系统的实现或成绩。

### 2.4 LocAgent

- **论文：** *LocAgent: Graph-Guided LLM Agents for Code Localization*，ACL 2025。[正式论文](https://aclanthology.org/2025.acl-long.426/)。
- **作者源码与权重：** [gersteinlab/LocAgent](https://github.com/gersteinlab/LocAgent)，Apache-2.0；[Qwen2.5-Coder-7B-CL](https://huggingface.co/czlll/Qwen2.5-Coder-7B-CL)、[Qwen2.5-Coder-32B-CL](https://huggingface.co/czlll/Qwen2.5-Coder-32B-CL)权重公开。7B 模型卡标 MIT；32B 模型卡的明确许可及完整训练材料尚待核实，不能由代码许可代推模型许可。
- **论文数据：** SWE-bench Lite、LocBench。LocBench 的任务构造和标注不能直接视为本地 SWE、BeetleBox 或 LCA 的另一份同分布测试集。
- **可比性：** 提供图结构辅助的代理定位对照，可评价当前轻量检索与重排方案的效果和成本。需使用文件级输出或预先冻结从更细粒度输出到文件的转换规则。
- **难度：** 中高。需要构建作者代码图、恢复代理环境并部署专用模型。公开推理权重不自动证明训练全流程可重现；先区分“使用作者权重评测”和“重新训练复现”。

### 2.5 Agentless

- **论文：** *Agentless: Demystifying LLM-based Software Engineering Agents*，FSE 2025。[DOI](https://doi.org/10.1145/3715754)。
- **作者源码与产物：** [OpenAutoCoder/Agentless](https://github.com/OpenAutoCoder/Agentless)，MIT；作者发布 SWE-bench Lite / Verified 的完整运行产物。官方 GPT/Claude 配置依赖闭源模型，代码开放不等于原实验全开放。
- **论文数据与任务：** SWE-bench Lite、SWE-bench Verified；完整系统同时包含定位和修复。当前项目只研究文件定位，应提取其定位阶段，不能把修复成功率与 MAP@10 并列作为相同任务指标。
- **可比性：** 可补充层级定位流程对照。若为满足本地开放模型要求替换后端，应明确记为 `Agentless + 指定模型`，记录与作者默认配置的差异。
- **难度：** 中高。代码与产物较完整，但文件排名导出、定位阶段边界以及模型替换都需要核对。现有运行产物可以审查流程，不能代替同一测试清单上的重新运行。

### 2.6 IGSIL / CoSIL

- **论文：** *Issue Localization via LLM-Driven Iterative Code Graph Searching*，ASE 2025 Research Papers。[会议页面](https://conf.researchr.org/details/ase-2025/ase-2025-papers/110/Issue-Localization-via-LLM-Driven-Iterative-Code-Graph-Searching)称方法 IGSIL；[论文预印本 v3](https://arxiv.org/html/2503.22424v3)和代码仓库称 CoSIL。应固定所用版本与引用，不把两种名称当作两篇独立论文。
- **作者源码：** [ZhonghaoJiang/CoSIL](https://github.com/ZhonghaoJiang/CoSIL)为本轮核查的新版本；[ZJU-CTAG/CoSIL](https://github.com/ZJU-CTAG/CoSIL)明确为旧版本。应冻结实际使用的仓库和 commit，避免混用两版脚本及结果。源码公开，许可证尚未确认。
- **模型与数据：** 预印本实验包括 Qwen2.5-Coder-7B、14B、32B，本地 vLLM 部署；SWE-bench Lite、SWE-bench Verified。本机已有 Qwen2.5-Coder-7B-Instruct 原始权重和 Q4_K_M 量化版本；模型版本、量化与后端应单独记录，量化运行不是原 vLLM 精度配置。32B 权重当前未见于本机模型目录。Lite / Verified 不能直接当作本地 SWE 格式化版 500 条。
- **可比性：** 代码图迭代搜索可作为另一类代理定位方法；论文报告文件级和函数级结果。需核验作者文件级排名的导出长度、文件去重和候选限制；若只能取得函数级排名，需要预先固定函数到文件的转换方式，并记录与作者评价的区别。
- **难度：** 中高。需要恢复新版图构建和搜索流程、控制迭代成本，并验证文件排名与路径。7B 是论文报告过的模型规模，但模型配置、推理参数及代码版本对齐仍需核验；当前未实际试跑。仓库未发现明确 LICENSE，不能只凭源码公开就称许可已确认的完全开源。

### 2.10 2025 顶会补充选型（2026-11-16）

新增与当前 500 条共同清单对比的 2025 年顶会 LLM 工作，已完成 **IGSIL / CoSIL 的 Qwen2.5-Coder-7B 文件级适配**的 500 条运行；见 [IGSIL 实验记录](igsil-baseline.md)。严格要求源码与模型许可明确时，备选 **LocAgent（ACL 2025）**。

| 方法 | 原论文／资料 | 适配本项目的判断 |
| --- | --- | --- |
| IGSIL / CoSIL | [ASE 2025 会议页面](https://conf.researchr.org/details/ase-2025/ase-2025-papers/110/Issue-Localization-via-LLM-Driven-Iterative-Code-Graph-Searching)、[新版作者仓库](https://github.com/ZhonghaoJiang/CoSIL)、[v3 预印本](https://arxiv.org/html/2503.22424v3) | SWE Lite/Verified 与本地 SWE 任务接近；论文也报告 7B，已有本地同系列权重。源码许可、实际运行和论文对应代码 commit 仍待核查。 |
| KEPT | [FSE 2025 会议页面](https://conf.researchr.org/details/fse-2025/fse-2025-research-papers/27/A-Knowledge-Enhanced-Large-Language-Model-for-Bug-Localization)、[公开材料仓库](https://github.com/keptmodel/KEPT) | 论文使用项目文档与历史代码构建知识图谱，七个项目、6,000 多条报告；仓库 README 指向 Google Drive 的模型与数据包，许可及下载可用性未核实。迁移到 SWE 当前快照需重建时间安全的项目知识图谱、核验训练和评价划分，工程量高。仓库 README 标题与最终论文标题不同，需进一步核验版本对应。 |
| OrcaLoca | [ICML 2025 正式论文](https://proceedings.mlr.press/v267/yu25x.html)、[作者仓库，MIT](https://github.com/fishmingyu/OrcaLoca) | 使用 SWE-bench Lite、Python 代码图及 LLM 代理；论文主要实验使用 Claude 3.5 Sonnet，仓库说明 Docker 和模型 API。改用本地开放模型须说明替换，与原论文条件不同；文件级输出与统一 @10 排名需核验。 |
| LocAgent | [ACL 2025 正式论文](https://aclanthology.org/2025.acl-long.426/)、[作者仓库，Apache-2.0](https://github.com/gersteinlab/LocAgent)、[作者 7B 权重](https://huggingface.co/czlll/Qwen2.5-Coder-7B-CL) | 有作者发布的 7B 专用权重和许可证明确的源码，适合在严格开放许可要求下优先审查；建图和代理运行仍未试跑，专用权重尚需下载、核验。7B 与论文某些 32B 主结果要区分。 |

以上排序针对本项目的 SWE Python 文件级任务及本机资源，不是实测速度或准确率排序。新方法应在固定同一份报告、修复前 SHA、候选文件与允许的报告字段后，输出可检查的文件排名；无法提供完整前十排名时不得把未排序集合伪装成 MAP@10。500 条此前参与方法开发，不能称为独立盲测。

### 2.7 Retrieval-Oriented Code Representations

- **论文：** *Retrieval-Oriented Code Representations in Agentic Bug Localization*，2026 年预印本。[论文](https://arxiv.org/abs/2607.11046)。本轮未核实正式会议或期刊录用信息，引用时标为预印本。
- **作者源码：** [gencau/code-representations](https://github.com/gencau/code-representations)，Apache-2.0。发布材料尚未下载试跑；具体生成模型、可重用摘要或索引以及完整运行依赖，需在接入前继续检查。
- **论文数据：** LCA、SWE-bench Verified。需核对所用语言、子集和实例 ID，再与本地 LCA 或 SWE 清单对应。
- **可比性：** 研究面向检索的代码表示、角色感知摘要及检索/融合重排，与当前代码证据选择有关，但其代码摘要生成属于额外的离线计算成本。
- **难度：** 中高。需把摘要生成、索引构建和每报告定位成本分开记录；只有仓库源码许可已经核实，不能据此断言所有模型与生成产物都已具备完整开放复现条件。

### 2.8 BRaIn

- **论文：** *Improved IR-Based Bug Localization with Intelligent Relevance Feedback*，ICPC 2025。[论文](https://arxiv.org/abs/2501.10542)。
- **作者源码：** [asifsamir/BRaIn](https://github.com/asifsamir/BRaIn)。源码与运行步骤公开，尚未确认明确 LICENSE。作者使用本地 LLaMA 3 8B、Mistral v0.3 7B、Qwen1.5 7B；各模型需分别核实原始许可，不能统一称为 MIT 或 Apache-2.0。
- **论文数据：** 从 Bench4BL 筛选 4,683 条报告，覆盖 42 个系统、684 个版本，按时间约 80:20 划分，测试集 956 条。这与本项目 46 个现代项目、9,459 条评价报告不是同一范围。
- **可比性：** 智能相关性反馈可以对照当前报告生成与重排设计，尤其适合 Java 实验。
- **难度：** 中。主要工作是重建时间划分、反馈流程和数据筛选。作者提示词优化使用了实验集中的 20 条报告；本项目应把提示词选择限于独立开发集，固定后再测测试集，并把与作者协议的差异记录清楚。

### 2.9 BLUiR

- **论文：** *Improving Bug Localization using Structured Information Retrieval*，ASE 2013。[作者论文 PDF](https://mattlease.com/papers/saha-ase13.pdf)、[DOI](https://doi.org/10.1109/ASE.2013.6693093)。
- **代码来源：** [Bench4BL](https://github.com/exatoa/Bench4BL)提供其修改后的 BLUiR 源码及运行包，并说明用统一格式导出各工具结果。这是 Bench4BL 的适配版本；本轮未确认明确源码许可，不能直接标为作者原样、许可已核实的完整开源实现。该方法不需要神经模型权重。
- **论文数据：** AspectJ、Eclipse、SWT、ZXing，约 3,400 条缺陷；后续 Bench4BL 扩展到更多 Java 项目。原论文数据、完整 Bench4BL 与本地适配子集应分别记录。
- **可比性：** 经典结构化词法检索基线，基础版本仅需源码和报告；利用相似历史缺陷的增强版本应另行标注。原实现通过 Eclipse JDT 提取类、方法、变量和注释字段，使用 Indri 检索。
- **难度：** Java 为中，Python 为高。Bench4BL 说明使用 Java 8 和 Indri 5.6；Python 需更换语法解析及结构字段提取并记录适配，不能直接宣称原实现支持 Python。适合在明确许可后作为 Java 侧经典补充。

## 3 数据集与评测边界

| 本项目资产 | 当前范围 | 与外部论文对比前必须核对 |
| --- | --- | --- |
| SWE-bench 格式化版 | 2,294 条；另有固定 500 条开发分析清单 | 不是 Lite 300 或 Verified 500；按仓库、issue、修复前 SHA 对应；查询仅用允许的报告字段 |
| BeetleBox | train 10,191；本地清洗后 test 8,955 | 与原始测试集条数不同；报告完全去重与空正文填充需保持一致；训练与测试还需审计交叉 |
| Ye 派生数据 | 本地 22,709 条，存储名为 train | 存储名不代表论文训练划分；缺少修复前 SHA，与 BLAZE 报告的 16,314 条测试规模不同，不能未经核对直接复现 |
| Bench4BL 本地适配 | 46 个现代项目、9,459 条 | 使用作者发布版本分组；不是 SWE 式逐缺陷修复前提交，也不同于 BRaIn 的 42 系统/956 条测试集 |
| LCA 测试集 | Java、Python、Kotlin 各 50 条 | 使用 PR base SHA 与 issue 原文；原始行 ID 不等同 GitHub issue 编号；检查跨数据集重复 |

数据来源、修订和标签覆盖以 [datasets.md](datasets.md)、[bench4bl.md](bench4bl.md)和 [lca.md](lca.md)为准。本地 500 条数据已经用于方法选择与开发分析，不能重新命名为最终独立测试集。

## 4 统一评测与记录要求

1. **区分作者设置与本项目对比。** 作者原数据、模型及评价脚本用于检查原方法；本项目比较使用共同样本、源码、候选与标签。改变模型、语言解析、候选过滤、函数到文件映射或评分方法时，明确标为适配/变体，不把它冒称为完整原样复现。
2. **固定输入。** 所有方法使用同一允许范围的缺陷报告。SWE 格式化版只用 `body`，不使用可能来自修复 PR 的独立 `title`。修复标签、补丁和未来提交不进入查询、生成或候选选择。需要历史信息的变体只使用当时已可获得的信息，并单独说明输入条件。
3. **固定源码与候选。** 对 SWE、BeetleBox、LCA 使用各数据协议规定的修复前版本；Bench4BL 保留发布版本语义。统一语言、测试文件、路径、大小及二进制过滤规则，记录候选清单指纹。标签缺失或映射不明确时保留审计，不根据正确答案筛选候选。
4. **统一输出与指标。** 输出去重的仓库相对文件路径和稳定排名，使用本项目 MAP@10、MRR@10、Top-1/5/10、MeanRecall@10，定义见 [evaluation.md](evaluation.md)。AP@10 的分母为该报告全部相关文件数；失败样本在主指标中计 0，另报成功样本指标和失败原因。作者的全排名 MAP/MRR、另一种 AP@K 分母或修复成功率不能直接混入同一表格。
5. **冻结参数与随机性。** 先确定开发集和独立测试集，再选择提示词、融合配额、图搜索轮数和重排参数。记录代码 commit、模型 revision/文件校验、依赖、量化、解码参数、随机种子、查询截断及重试策略。沿用作者模型的结果与更换为当前 Qwen 模型的控制实验分别命名。
6. **同时评估效果与成本。** 保留各方法完整流程，记录源码预处理、摘要/图/索引构建、单报告模型调用、token、时延和失败率。需要等预算实验时单独设置候选数、证据量或调用上限；不能未经说明删掉外部方法关键阶段后仍使用原方法名称。
7. **隔离论文分数与实测分数。** 原论文数字只能进入带来源与原协议的参考表；正式对比表必须来自同一冻结测试清单上的本机产物。只运行少量样本时标为功能检查，未跑的方法保持“未运行”，不填写估计精度。

本页只记录文献与可复现性筛选，不包含任何新增基线的实测优劣结论。BLAZE 的具体接入和验证记录见 [blaze-baseline.md](blaze-baseline.md)，后续基线在实际完成验证后再补充独立实验记录。
