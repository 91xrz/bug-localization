"""Render offline attribution and review packets without changing experiment outcomes."""
from collections import Counter
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src'))
from bug_localization.reranking import read_json
from bug_localization.batch import write_json


def main():
    out=PROJECT/'results/c3_failure_attribution/20261112'
    data=read_json(out/'attribution.json');notes=read_json(out/'assistant_review_notes.json')['cases']
    note_by_id={n['case_id']:n for n in notes};selection=read_json(out/'review_selection.json')['samples']
    diagnostics={}
    for cohort in data['cohorts']:
        rows=[r for r in data['rows'] if r['cohort']==cohort and all(v['status']=='ok' for v in r['methods'].values())]
        files={m:Counter() for m in ('A','C3')};flow=Counter();same_evidence=Counter()
        for r in rows:
            a={g['file_path']:g for g in r['methods']['A']['gold_files']};c={g['file_path']:g for g in r['methods']['C3']['gold_files']}
            if a.keys()!=c.keys():raise ValueError('Different gold labels')
            for f in a:
                flow[(a[f]['stage'],c[f]['stage'])]+=1
                files['A'][a[f]['stage']]+=1;files['C3'][c[f]['stage']]+=1
            sign='gain' if r['delta_ap']>1e-12 else 'loss' if r['delta_ap']<-1e-12 else 'tie'
            if all(set(a[f]['selected_chunk_ids'])==set(c[f]['selected_chunk_ids']) for f in a):same_evidence[sign]+=1
        diagnostics[cohort]={'common_success_reports':len(rows),'gold_file_occurrences':{m:dict(c) for m,c in files.items()},
            'transitions':[{'A':a,'C3':c,'count':n} for (a,c),n in flow.items()],
            'same_gold_chunk_sets':dict(same_evidence)}
    write_json(out/'paired_stage_diagnostics.json',diagnostics)
    # Full packets remain available; markdown makes the manual review directly usable.
    for r in selection:
        key=f"{r['cohort']}_{r['position']:03d}";p=read_json(out/'review_packets'/(key+'.json'));note=note_by_id[key]
        lines=[f"# {key}: {r['repository']}",'',f"抽样层：{r['sampling_stratum']}；ΔAP={r['delta_ap']:.6f}。",'',
            '**助手初步分析，尚未人工确认。** '+note['notes_zh'],'',f"报告来源：{p['issue_url']}",'','## 原报告','',p['report_body'],'','## 功能推测','',p['hypothesis_query'],'','## 候选和排名','',
            '| 方法 | 正确文件 | 候选内排名 | 自动归类 |','|---|---|---:|---|']
        for m,s in p['methods'].items():
            for g in s['gold_files']:lines.append(f"| {m} | {g['file_path']} | {g['rank']} | {g['stage']} |")
        for m,e in p['selected_evidence'].items():
            lines+=['',f'## {m} 实际重排查询','', '````text',e['query_used'],'````','',f'## {m} 正确文件的已选代码块','']
            for c in e['gold_chunks']:
                lines += [f"### {c['chunk_id']}",'',f"得分：{c['score']}；来源：{c.get('selected_by','original')}",'','````python',c['embedding_text'],'````','']
            lines += [f"## {m} 排名最高的非标签文件",'',str(e['top_non_gold_path']),'']
            for c in e['top_non_gold_chunks']:lines+=['````python',c['embedding_text'],'````','']
        if 'retrospective_reference_patch' in p:lines+=['## 回顾性补丁证据（未送入模型）','','````diff',p['retrospective_reference_patch'],'````','']
        lines+=['## 人工复核项目','','- 推测补充了报告缺少的功能，还是改写已有信息？','- 推测中的具体机制与代码/补丁是否矛盾？','- 已选块能否提供直接或间接定位线索？记录chunk_id及原文证据。','- 若相关块已选，是否被低分或竞争文件挤后？','- 新增/重命名文件、测试和多文件修复标签是否限制了判断？','- 归因尚不确定时保留 unresolved，不按改善/退化倒推答案。']
        (out/'review_packets'/(key+'.md')).write_text('\n'.join(lines)+'\n')
    lines=['# C3 离线失效归因与下一轮验证方案','',
        '复用开发集500条与LCA47条结果，未运行新组合、未重新生成推测或调用重排器。完成候选覆盖及指标复算、28例证据包和助手初步分析；人工确认案例数仍为0。统计事实与语义判断分开报告。','',
        '## 1. 可由已有结果直接确认的分布','',
        '按“报告—正确文件”计数，不是独立文件数，也不是报告失败率。下表只比较两方法都成功的相同报告：开发集499条、821个文件标签；LCA47条、123个文件标签。开发集另1条C3推测失败仍在原500条主指标中计零，不混入选块/重排诊断。','',
        '| 阶段 | 开发集 A | 开发集 C3 | LCA A | LCA C3 |','|---|---:|---:|---:|---:|']
    for key,label in [('source_inventory_gap','修复前可用源码/候选范围中不存在'),('recall_gap','源码中存在，但未进入候选50'),('selection_or_reranking_unresolved','进入候选50，但排在Top10之后'),('retrieved_at_k','进入Top10')]:
        counts=[diagnostics[c]['gold_file_occurrences'][m].get(key,0) for c in ('development500','external47') for m in ('A','C3')]
        lines.append('| '+label+' | '+' | '.join(map(str,counts))+' |')
    lines+=['','历史A结果包含未重排的候选池外尾部；本次以实际 scored_chunks 中出现的文件识别候选50，不能把完整ranking误当成候选池。源码缺口包括新增文件或过滤范围，应与语义召回不足分开。Top10内仍可能有AP排序损失，不能视为全部定位问题已解决。','',
        '开发集C3相对A新召回26个正确文件，其中只有7个进入Top10，19个仍排后；另有10个原本召回的正确文件被挤出候选。LCA新召回11个正确文件，只有2个进入Top10，9个仍排后；另丢失2个原候选正确文件。召回改善没有充分转化为最终定位收益。','',
        '**不能自动把候选内排后归为重排器失败。** 开发集C3的118个、LCA的34个此类文件仍需看代码，区分没有有效线索与有线索但被低估。目前不存在人工确认的全量“选块不足占比/重排不足占比”，不据28例推算总体比例。','',
        '## 2. 改善、退化与普通案例','',
        '固定抽样包含开发集5个最大改善、5个最大退化、10个普通样本；LCA为2、2、4。普通样本覆盖同分未满分、同分满分、小幅变化。属于诊断抽样，不是机制发生率估计。','',
        '| 案例 | ΔAP | 证据与初步归因 |','|---|---:|---|']
    featured=['external47_001','development500_222','development500_464','development500_068','development500_133','development500_295','external47_022','external47_015','external47_019','external47_016','development500_267','external47_003']
    by_key={f"{r['cohort']}_{r['position']:03d}":r for r in selection}
    for key in featured:
        r=by_key[key];n=note_by_id[key]
        lines.append(f"| [{key}](../results/c3_failure_attribution/20261112/review_packets/{key}.md) | {r['delta_ap']:.4f} | {n['notes_zh']} |")
    lines+=['','这些例子同时反驳了两种过强解释：“改善一定意味着补出了正确根因”和“退化一定源于错误推测”。有正确机制却丢掉有效块的例子，有更相关的执行代码被重排低估的例子，也有错误细节仍带来改善的例子。','',
        '在开发集共同成功样本中，8个改善和4个退化报告的全部正确文件选块集合没有变化；竞争文件和BF16批次评分变化也必须纳入解释。LCA OpenBB例的gold块完全相同但分数与名次变化，不能记作gold选块成功。','',
        '## 3. 下一步优先处理什么','',
        '先让人工复核选块与相关性评分的边界：groupby丢掉明确name=None判定分支；pairwise与Pulumi已经选到与故障直接相关的执行块却低分。这两类优先于继续扩大混合召回组合。对新增召回但仍在Top10外的文件，逐例核对有效块是否进入输入。保留原文对照A，不在现有500/LCA47上选择新的策略。','',
        '## 4. 下一轮Python外部验证：方法与规则冻结，数据尚未开跑','',
        '按用户指定的最小实际增益：**MAP@10绝对提升0.02**。唯一主要对比C3−A；双侧α=0.05、规划功效80%；失败计零，A不依赖推测成功，禁止失败回退/样本替换。次要指标单列并做Holm校正。主推断按项目成组，报告项目差异和耗时，完成全部样本后一次分析，不中途挑选策略。','',
        '历史配对差值标准差：开发集0.1384、LCA47为0.1651，采用较大值规划。独立报告近似需535条；每项目最多5条、假设组内相关0.10时设计效应1.4，近似需749条，预留至**800条**。这是规划近似，不是80%功效保证：若组内相关0.20则需963条；若标准差升至0.20则需1099条。','',
        '审计LCA原始Python train划分（不拼接与其重叠的dev），排除全部历史项目、历史报告、重复正文/issue、非Python主语言和无Python标签后，得到**3,286条、411个未使用项目**。按每项目最多5条可供1,166条，已固定一个**800条、411项目的源码获取预选清单**。这不是已通过质量检查的最终测试集，也不是实验结果。','',
        '**当前未运行新验证：800条对应项目均无本地源码包。** 需要核验精确base提交并完成不看模型输出的issue—PR—标签质量复核。这批原始train标签不能冒充此前人工核验的LCA test标签。源码与质量门槛通过后，记录全部排除原因、冻结最终清单，才启动A/C3。若无法达到样本量，报告不足，不悄悄复用旧项目。','',
        '本轮不做Bench4BL Java迁移，不做小模型/短生成/按需推测优化。新验证未确认收益前，不以成本优化替代效果验证。','',
        '## 产物与复核入口','',
        '- [全量归因数据](../results/c3_failure_attribution/20261112/attribution.json)：547条报告，保留失败。',
        '- [配对阶段转移](../results/c3_failure_attribution/20261112/paired_stage_diagnostics.json)：同分母比较。',
        '- [28例索引](../results/c3_failure_attribution/20261112/case_index.csv)、[助手阅读笔记](../results/c3_failure_attribution/20261112/assistant_review_notes.json)、[人工核查模板](../results/c3_failure_attribution/20261112/human_review_template.json)。',
        '- [下一轮协议](../configs/experiments/c3_external_python_next.json)、[样本量敏感性](../results/c3_failure_attribution/20261112/next_validation/power_analysis.json)、[源码获取预选清单](../results/c3_failure_attribution/20261112/next_validation/acquisition_reserve.json)。',
        '- 原始报告、推测文本、真正送入重排器的查询、gold代码块及竞争文件块、可用的回顾性补丁，均保存在各案例证据包。补丁仅用于事后分析，从未送入检索或重排。','',
        '## 全部诊断案例','', '| 案例 | 抽样层 | ΔAP | 初步观察 |','|---|---|---:|---|']
    for key,r in by_key.items():
        lines.append(f"| [{key}](../results/c3_failure_attribution/20261112/review_packets/{key}.md) | {r['sampling_stratum']} | {r['delta_ap']:.4f} | {note_by_id[key]['notes_zh']} |")
    path=PROJECT/'docs/c3-failure-attribution-and-next-validation.md';path.write_text('\n'.join(lines)+'\n');print(path)

if __name__=='__main__':main()
