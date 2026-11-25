# C3 离线失效归因与下一轮验证方案

复用开发集500条与LCA47条结果，未运行新组合、未重新生成推测或调用重排器。完成候选覆盖及指标复算、28例证据包和助手初步分析；人工确认案例数仍为0。统计事实与语义判断分开报告。

## 1. 可由已有结果直接确认的分布

按“报告—正确文件”计数，不是独立文件数，也不是报告失败率。下表只比较两方法都成功的相同报告：开发集499条、821个文件标签；LCA47条、123个文件标签。开发集另1条C3推测失败仍在原500条主指标中计零，不混入选块/重排诊断。

| 阶段 | 开发集 A | 开发集 C3 | LCA A | LCA C3 |
|---|---:|---:|---:|---:|
| 修复前可用源码/候选范围中不存在 | 24 | 24 | 0 | 0 |
| 源码中存在，但未进入候选50 | 99 | 83 | 22 | 13 |
| 进入候选50，但排在Top10之后 | 107 | 118 | 23 | 34 |
| 进入Top10 | 591 | 596 | 78 | 76 |

历史A结果包含未重排的候选池外尾部；本次以实际 scored_chunks 中出现的文件识别候选50，不能把完整ranking误当成候选池。源码缺口包括新增文件或过滤范围，应与语义召回不足分开。Top10内仍可能有AP排序损失，不能视为全部定位问题已解决。

开发集C3相对A新召回26个正确文件，其中只有7个进入Top10，19个仍排后；另有10个原本召回的正确文件被挤出候选。LCA新召回11个正确文件，只有2个进入Top10，9个仍排后；另丢失2个原候选正确文件。召回改善没有充分转化为最终定位收益。

**不能自动把候选内排后归为重排器失败。** 开发集C3的118个、LCA的34个此类文件仍需看代码，区分没有有效线索与有线索但被低估。目前不存在人工确认的全量“选块不足占比/重排不足占比”，不据28例推算总体比例。

## 2. 改善、退化与普通案例

固定抽样包含开发集5个最大改善、5个最大退化、10个普通样本；LCA为2、2、4。普通样本覆盖同分未满分、同分满分、小幅变化。属于诊断抽样，不是机制发生率估计。

| 案例 | ΔAP | 证据与初步归因 |
|---|---:|---|
| [external47_001](../results/c3_failure_attribution/20261112/review_packets/external47_001.md) | 1.0000 | 报告只给数值、链接，未明说3/5倍数求和；推测补出算术级数和整数除法方向，与保存代码及补丁将/改为//一致。A未召回目标，C3召回并排1。这是功能补充有帮助的代表，但不能由单例区分推理与已有知识。 |
| [development500_222](../results/c3_failure_attribution/20261112/review_packets/development500_222.md) | 0.6667 | 报告明确点名_check_list_display_item；A只选到外围_check_list_display与editable块，C3选到目标函数，文件3→1。推测称hasattr检查被移除，但保存代码仍含hasattr(obj.model,item)，不能当成可靠的实现事实。 |
| [development500_464](../results/c3_failure_attribution/20261112/review_packets/development500_464.md) | 0.7500 | C3 新增 ColorizeFormatter.format，直接涉及日志颜色，文件4→1；A 的块主要为过滤器和setup。推测把构建控制台警告说成HTML渲染，场景不准确，却仍选到有用的日志着色块。 |
| [development500_068](../results/c3_failure_attribution/20261112/review_packets/development500_068.md) | -0.5000 | 报告只有groupby应支持name=None。A选中明确的if getattr(group,"name",None) is None: raise ValueError，得5.12；C3丢掉此块，换成无组/bins错误与状态赋值，文件1→2。可直接看到有效判定分支丢失；推测本身并不明显错误。 |
| [development500_133](../results/c3_failure_attribution/20261112/review_packets/development500_133.md) | -0.5000 | C3选到_parallel_pairwise中的np.empty(...dtype=X.dtype)与线程写回，恰与整数输入导致小数被截断的现象一致；该块仅3.94分。A泛化的pairwise_distances参数说明块7.88被替换，文件1→2。这支持核查重排器对执行逻辑的低估，不支持把退化简单归因于错误推测。 |
| [development500_295](../results/c3_failure_attribution/20261112/review_packets/development500_295.md) | -0.7500 | 报告抱怨输出过长、希望约40行截断；推测反说threshold太低、表示不完整，方向相反。C3保留short_numpy_repr，却把高分的截断工具块换成按字符宽度平铺的format_array_flat，文件1→4。错误推测与选块变化同时出现，单例不证明因果。 |
| [external47_022](../results/c3_failure_attribution/20261112/review_packets/external47_022.md) | -0.3000 | 两组都选到补丁增加str转换的位置：deserialize_properties直接构造FileAsset等；该块两组均为-3.00。C3用serialize_property块替换另一较高分块，文件2→5。有效故障代码已在输入中仍被低估，不能认定主要是漏选它。 |
| [external47_015](../results/c3_failure_attribution/20261112/review_packets/external47_015.md) | 0.2500 | 报告已给完整traceback、函数与缺失header。两组gold选块完全相同，包含补丁所修改的header.index表达式，但模型块分数9.56→9.75，文件2→1；不能把增益记作gold选块改善。不同候选/输入批次可能改变BF16分数及竞争顺序。 |
| [external47_019](../results/c3_failure_attribution/20261112/review_packets/external47_019.md) | 0.0000 | 普通并列失败：全部3个gold文件都进候选；exceptions.py仍在15/16。补丁修改首行from sanic.http import STATUS_CODES，但其已选块为异常类、abort和孤立装饰器，不含该导入，支持选块不足待人工确认。推测正确指出模块遮蔽，但把response/exceptions的http用途说成cookies，与补丁的状态码用途不符。 |
| [external47_016](../results/c3_failure_attribution/20261112/review_packets/external47_016.md) | 0.0000 | 普通并列失败：A漏4/6个gold，C3漏3/6并把新增tests/test_widget排46；await_remove/message_pump/_context仍未召回。推测停留在示例的populate/click和事件循环，没有具体补出这些内部同步组件。 |
| [development500_267](../results/c3_failure_attribution/20261112/review_packets/development500_267.md) | 0.0000 | 普通并列样本包含大量新增下划线模块标签，修复前候选清单中不存在。两组均有8个此类路径缺口；不能把这些算作语义召回或选块失败。C3召回了更多旧模块，但Top10 AP不变。 |
| [external47_003](../results/c3_failure_attribution/20261112/review_packets/external47_003.md) | 0.0000 | 两组均第1，选块相同，包含subprocess.run；补丁改用check_call兼容Python2。推测的版本兼容说明正确，但原traceback与Python2.7信息已足够，准确率无增益。 |

这些例子同时反驳了两种过强解释：“改善一定意味着补出了正确根因”和“退化一定源于错误推测”。有正确机制却丢掉有效块的例子，有更相关的执行代码被重排低估的例子，也有错误细节仍带来改善的例子。

在开发集共同成功样本中，8个改善和4个退化报告的全部正确文件选块集合没有变化；竞争文件和BF16批次评分变化也必须纳入解释。LCA OpenBB例的gold块完全相同但分数与名次变化，不能记作gold选块成功。

## 3. 下一步优先处理什么

先让人工复核选块与相关性评分的边界：groupby丢掉明确name=None判定分支；pairwise与Pulumi已经选到与故障直接相关的执行块却低分。这两类优先于继续扩大混合召回组合。对新增召回但仍在Top10外的文件，逐例核对有效块是否进入输入。保留原文对照A，不在现有500/LCA47上选择新的策略。

## 4. 下一轮Python外部验证：方法与规则冻结，数据尚未开跑

按用户指定的最小实际增益：**MAP@10绝对提升0.02**。唯一主要对比C3−A；双侧α=0.05、规划功效80%；失败计零，A不依赖推测成功，禁止失败回退/样本替换。次要指标单列并做Holm校正。主推断按项目成组，报告项目差异和耗时，完成全部样本后一次分析，不中途挑选策略。

历史配对差值标准差：开发集0.1384、LCA47为0.1651，采用较大值规划。独立报告近似需535条；每项目最多5条、假设组内相关0.10时设计效应1.4，近似需749条，预留至**800条**。这是规划近似，不是80%功效保证：若组内相关0.20则需963条；若标准差升至0.20则需1099条。

审计LCA原始Python train划分（不拼接与其重叠的dev），排除全部历史项目、历史报告、重复正文/issue、非Python主语言和无Python标签后，得到**3,286条、411个未使用项目**。按每项目最多5条可供1,166条，已固定一个**800条、411项目的源码获取预选清单**。这不是已通过质量检查的最终测试集，也不是实验结果。

**当前未运行新验证：800条对应项目均无本地源码包。** 需要核验精确base提交并完成不看模型输出的issue—PR—标签质量复核。这批原始train标签不能冒充此前人工核验的LCA test标签。源码与质量门槛通过后，记录全部排除原因、冻结最终清单，才启动A/C3。若无法达到样本量，报告不足，不悄悄复用旧项目。

本轮不做Bench4BL Java迁移，不做小模型/短生成/按需推测优化。新验证未确认收益前，不以成本优化替代效果验证。

## 产物与复核入口

- [全量归因数据](../results/c3_failure_attribution/20261112/attribution.json)：547条报告，保留失败。
- [配对阶段转移](../results/c3_failure_attribution/20261112/paired_stage_diagnostics.json)：同分母比较。
- [28例索引](../results/c3_failure_attribution/20261112/case_index.csv)、[助手阅读笔记](../results/c3_failure_attribution/20261112/assistant_review_notes.json)、[人工核查模板](../results/c3_failure_attribution/20261112/human_review_template.json)。
- [下一轮协议](../configs/experiments/c3_external_python_next.json)、[样本量敏感性](../results/c3_failure_attribution/20261112/next_validation/power_analysis.json)、[源码获取预选清单](../results/c3_failure_attribution/20261112/next_validation/acquisition_reserve.json)。
- 原始报告、推测文本、真正送入重排器的查询、gold代码块及竞争文件块、可用的回顾性补丁，均保存在各案例证据包。补丁仅用于事后分析，从未送入检索或重排。

## 全部诊断案例

| 案例 | 抽样层 | ΔAP | 初步观察 |
|---|---|---:|---|
| [development500_264](../results/c3_failure_attribution/20261112/review_packets/development500_264.md) | large_gain | 0.8000 | 原报告已给出 rotation/linespacing 和 setter 要求。C3 保留 setter 块，新增 Text.__init__ 参数块，最高分升至4.81、文件5→1。推测没有补出关键缺失功能，更多是重述加字典/集合猜测；不能只凭提升认定根因推断正确。 |
| [development500_464](../results/c3_failure_attribution/20261112/review_packets/development500_464.md) | large_gain | 0.7500 | C3 新增 ColorizeFormatter.format，直接涉及日志颜色，文件4→1；A 的块主要为过滤器和setup。推测把构建控制台警告说成HTML渲染，场景不准确，却仍选到有用的日志着色块。 |
| [development500_047](../results/c3_failure_attribution/20261112/review_packets/development500_047.md) | large_gain | 0.6667 | A 已选 MultiCursor.onmove/_update，C3 新增构造器块获7.38分，文件3→1。事件/绘制描述相关，但timer及idle原因未经补丁验证，不能把分数提升等同于补中根因。 |
| [development500_222](../results/c3_failure_attribution/20261112/review_packets/development500_222.md) | large_gain | 0.6667 | 报告明确点名_check_list_display_item；A只选到外围_check_list_display与editable块，C3选到目标函数，文件3→1。推测称hasattr检查被移除，但保存代码仍含hasattr(obj.model,item)，不能当成可靠的实现事实。 |
| [development500_033](../results/c3_failure_attribution/20261112/review_packets/development500_033.md) | large_gain | 0.5000 | 报告已包含ModelChoiceField错误模板；A已有抛ValidationError的to_python块，C3再选到缺少value占位符的模板，最高分6.75，文件2→1。属于证据补充，不是发现报告中没有的功能。 |
| [development500_234](../results/c3_failure_attribution/20261112/review_packets/development500_234.md) | large_loss | -0.7500 | 推测指向GROUP BY别名；C3选到set_group_by，语义上与报告直接相关，但分数为-2.75/-2.00；A的get_aggregation说明块7.56被移除，文件1→4。更相关不一定更高分；缺少补丁核对，不能判定推测错误。 |
| [development500_295](../results/c3_failure_attribution/20261112/review_packets/development500_295.md) | large_loss | -0.7500 | 报告抱怨输出过长、希望约40行截断；推测反说threshold太低、表示不完整，方向相反。C3保留short_numpy_repr，却把高分的截断工具块换成按字符宽度平铺的format_array_flat，文件1→4。错误推测与选块变化同时出现，单例不证明因果。 |
| [development500_455](../results/c3_failure_attribution/20261112/review_packets/development500_455.md) | large_loss | -0.7500 | 报告明确相对路径按rst文件解析；推测却说固定基目录而非源文件相对，部分不符。C3用低分Figure块替换A最高分模块块，文件1→4；两组CSVTable块主要是标题/源码位置处理，尚未证实所选块含路径解析根因，需要扩展上下文人工核查。 |
| [development500_068](../results/c3_failure_attribution/20261112/review_packets/development500_068.md) | large_loss | -0.5000 | 报告只有groupby应支持name=None。A选中明确的if getattr(group,"name",None) is None: raise ValueError，得5.12；C3丢掉此块，换成无组/bins错误与状态赋值，文件1→2。可直接看到有效判定分支丢失；推测本身并不明显错误。 |
| [development500_133](../results/c3_failure_attribution/20261112/review_packets/development500_133.md) | large_loss | -0.5000 | C3选到_parallel_pairwise中的np.empty(...dtype=X.dtype)与线程写回，恰与整数输入导致小数被截断的现象一致；该块仅3.94分。A泛化的pairwise_distances参数说明块7.88被替换，文件1→2。这支持核查重排器对执行逻辑的低估，不支持把退化简单归因于错误推测。 |
| [development500_374](../results/c3_failure_attribution/20261112/review_packets/development500_374.md) | ordinary_tied_imperfect | 0.0000 | 普通并列样本：两组deprecated.py块相同，文件均第3。推测重述文档迁移需求，没有产生额外区分力；仍需看警告定义与文档标签口径。 |
| [development500_356](../results/c3_failure_attribution/20261112/review_packets/development500_356.md) | ordinary_tied_imperfect | 0.0000 | 普通并列样本：报告给出目标文件并要求新增glob支持。推测猜测已有split/glob逻辑无法递归，未经证实。C3换块但最高分仍来自共有块，正确文件保持第2。 |
| [development500_267](../results/c3_failure_attribution/20261112/review_packets/development500_267.md) | ordinary_tied_imperfect | 0.0000 | 普通并列样本包含大量新增下划线模块标签，修复前候选清单中不存在。两组均有8个此类路径缺口；不能把这些算作语义召回或选块失败。C3召回了更多旧模块，但Top10 AP不变。 |
| [development500_236](../results/c3_failure_attribution/20261112/review_packets/development500_236.md) | ordinary_tied_imperfect | 0.0000 | 两组均选择_check_indexes，最高分共有块4.44且文件均第3；C3新增_check_field_name_clashes没有改善。适合作为普通低排名对照，不能仅凭文件标签宣布选块已充分。 |
| [development500_240](../results/c3_failure_attribution/20261112/review_packets/development500_240.md) | ordinary_tied_perfect | 0.0000 | 两组均保留删除方法块，均排第1；原报告已明确with块外访问rowcount。推测正确地复述这一点，但此样本没有额外准确率收益。 |
| [development500_022](../results/c3_failure_attribution/20261112/review_packets/development500_022.md) | ordinary_tied_perfect | 0.0000 | 报告已有确切文件、save调用和数据库说明。两组均选_rename、均第1；新增推测没有带来AP改善。 |
| [development500_271](../results/c3_failure_attribution/20261112/review_packets/development500_271.md) | ordinary_tied_perfect | 0.0000 | 两组均以共有style.use/fix_style块获得最高分，均第1；版本变化的推测未进一步提高已满分结果。 |
| [development500_207](../results/c3_failure_attribution/20261112/review_packets/development500_207.md) | ordinary_small_change | 0.0125 | 报告请求新增常量版本context manager；推测把它描述成现有context manager恢复状态失败，属于无依据的现状假设。常量utils.py标签在修复前不存在，两组gold代码块实际相同，微小收益来自其他文件相对顺序，不能归因于修复了该假设。 |
| [development500_152](../results/c3_failure_attribution/20261112/review_packets/development500_152.md) | ordinary_small_change | -0.0333 | 两组均保留_resolve_ref_xref且得分0.75不变；C3替换的另一个块得分更低，gold文件5→6。应核查竞争文件而非直接指责gold块不足。 |
| [development500_094](../results/c3_failure_attribution/20261112/review_packets/development500_094.md) | ordinary_small_change | -0.0250 | C3新增_preprocess_options，符合CLI选项问题，但得-4.56；最高分仍为共有扩展加载块-1.00。文件8→10；需要完整函数上下文确认相关分支，暂不下选块失败结论。 |
| [external47_001](../results/c3_failure_attribution/20261112/review_packets/external47_001.md) | large_gain | 1.0000 | 报告只给数值、链接，未明说3/5倍数求和；推测补出算术级数和整数除法方向，与保存代码及补丁将/改为//一致。A未召回目标，C3召回并排1。这是功能补充有帮助的代表，但不能由单例区分推理与已有知识。 |
| [external47_015](../results/c3_failure_attribution/20261112/review_packets/external47_015.md) | large_gain | 0.2500 | 报告已给完整traceback、函数与缺失header。两组gold选块完全相同，包含补丁所修改的header.index表达式，但模型块分数9.56→9.75，文件2→1；不能把增益记作gold选块改善。不同候选/输入批次可能改变BF16分数及竞争顺序。 |
| [external47_022](../results/c3_failure_attribution/20261112/review_packets/external47_022.md) | large_loss | -0.3000 | 两组都选到补丁增加str转换的位置：deserialize_properties直接构造FileAsset等；该块两组均为-3.00。C3用serialize_property块替换另一较高分块，文件2→5。有效故障代码已在输入中仍被低估，不能认定主要是漏选它。 |
| [external47_027](../results/c3_failure_attribution/20261112/review_packets/external47_027.md) | large_loss | -0.2583 | 实现文件两组3块完全相同，含补丁修复的_inner_schema分支，文件3→4且该块分数5.62→5.75。推测偏向validator改变值造成Literal不匹配，补丁实际处理validator包装schema。实现文件退化不能归为漏选块；测试文件选块则有变化。 |
| [external47_019](../results/c3_failure_attribution/20261112/review_packets/external47_019.md) | ordinary_tied_imperfect | 0.0000 | 普通并列失败：全部3个gold文件都进候选；exceptions.py仍在15/16。补丁修改首行from sanic.http import STATUS_CODES，但其已选块为异常类、abort和孤立装饰器，不含该导入，支持选块不足待人工确认。推测正确指出模块遮蔽，但把response/exceptions的http用途说成cookies，与补丁的状态码用途不符。 |
| [external47_016](../results/c3_failure_attribution/20261112/review_packets/external47_016.md) | ordinary_tied_imperfect | 0.0000 | 普通并列失败：A漏4/6个gold，C3漏3/6并把新增tests/test_widget排46；await_remove/message_pump/_context仍未召回。推测停留在示例的populate/click和事件循环，没有具体补出这些内部同步组件。 |
| [external47_003](../results/c3_failure_attribution/20261112/review_packets/external47_003.md) | ordinary_tied_perfect | 0.0000 | 两组均第1，选块相同，包含subprocess.run；补丁改用check_call兼容Python2。推测的版本兼容说明正确，但原traceback与Python2.7信息已足够，准确率无增益。 |
| [external47_040](../results/c3_failure_attribution/20261112/review_packets/external47_040.md) | ordinary_small_change | 0.0208 | 普通微小改善：推测复述安装目录/状态文件缺失，未识别补丁新增的shims/executable判定。C3丢失一个测试标签候选，helpers仍第18，ruby由4→3带来小幅AP提升。不能据此认为故障机制解释改善。 |
