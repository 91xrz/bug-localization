"""Summarize the completed fixed500 paired initial-ranking/C experiment."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/ab_structure_c/20261123_fixed500'


def read(name):
    return json.loads((RUN/name).read_bytes())


def main():
    summary=read('summary.json');stats=read('paired_statistics.json');verified=read('verification.json')
    if verified['status']!='passed':raise ValueError('Incomplete experiment')
    cases=[json.loads(p.read_bytes()) for p in sorted((RUN/'cases').glob('*.json'))]
    good=[r for r in cases if r['status']=='ok'];protocol=read('protocol.json')
    diagnostics={}
    for left,right in [('new_ABC','old_ABC'),('new_ABC','new_AB'),('old_ABC','old_ABC_historical')]:
        key=left+'_vs_'+right
        diagnostics[key]={}
        for k in ('top_1','top_10'):
            diagnostics[key][k]={'rescued':sum(r['metrics'][left][k]>r['metrics'][right][k] for r in good),
                                'lost':sum(r['metrics'][left][k]<r['metrics'][right][k] for r in good)}
    same_pool=[r for r in good if set(r['rankings']['old_AB'])==set(r['rankings']['new_AB'])]
    diagnostics['same_pool_reports']=len(same_pool)
    diagnostics['same_pool_changed_final_order']=sum(r['rankings']['old_ABC']!=r['rankings']['new_ABC'] for r in same_pool)
    diagnostics['same_pool_changed_ap']=sum(abs(r['metrics']['old_ABC']['average_precision_at_k']-r['metrics']['new_ABC']['average_precision_at_k'])>1e-12 for r in same_pool)
    diagnostics['mean_common_candidates']=sum(len(set(r['rankings']['old_AB'])&set(r['rankings']['new_AB'])) for r in good)/len(good)
    diagnostics['historical_identical_scores']=sum(r['historical_score_check']['identical_pairs'] for r in good)
    diagnostics['historical_total_scores']=sum(r['historical_score_check']['pairs'] for r in good)
    diagnostics['historical_max_absolute_delta']=verified['historical_max_score_delta']
    diagnostics['wall_seconds_including_preparation']=read('status.json')['finished_unix']-protocol['started_unix']
    diagnostics['sum_worker_case_seconds']=sum(r['elapsed_seconds'] for r in good)
    changes=[]
    for r in good:
        d=r['metrics']['new_ABC']['average_precision_at_k']-r['metrics']['old_ABC']['average_precision_at_k']
        if abs(d)<1e-12:continue
        changes.append({k:r[k] for k in ('position','repository','issue_id','ground_truth')}|{
            'delta_ap':d,'old_top10':r['rankings']['old_ABC'][:10], 'new_top10':r['rankings']['new_ABC'][:10],
            'added_candidates':sorted(set(r['rankings']['new_AB'])-set(r['rankings']['old_AB'])),
            'dropped_candidates':sorted(set(r['rankings']['old_AB'])-set(r['rankings']['new_AB']))})
    diagnostics['largest_gains']=sorted(changes,key=lambda x:-x['delta_ap'])[:10]
    diagnostics['largest_losses']=sorted(changes,key=lambda x:x['delta_ap'])[:10]
    (RUN/'diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+'\n')
    names={'old_AB':'旧AB：B40+A补足、RRF60','new_AB':'新AB：固定分数融合＋结构匹配',
           'old_ABC':'旧AB＋C，本轮重算','new_ABC':'新AB＋C，本轮重算','old_ABC_historical':'旧AB＋C，历史缓存对照'}
    d=stats['new_ABC_vs_old_ABC']['average_precision_at_k'];lo,hi=d['ci95_project_stratified_bootstrap']
    lines=['# 新 AB 融合＋C：固定500条配对实验（2026-11-23）','',
           f"**结论：本轮新AB+C与旧AB+C基本持平。MAP由{summary['old_ABC']['metrics']['MAP@10']:.6f}变为{summary['new_ABC']['metrics']['MAP@10']:.6f}，差值{d['delta']:+.6f}，95%配对区间[{lo:+.6f},{hi:+.6f}]，未显示稳定改善。Top1同为59.2%，Top10由93.0%升至93.2%，净增1条。**",'',
           '沿用原跨项目500条冻结清单和顺序，含旧180及新增320，未再选参。新AB使用已登记的 `ab_structure_v1`，原报告/推测分数权重各0.5、结构匹配权重0.5、最终50文件。500条是已观察且用于AB开发的数据，本轮不是独立盲测。两组统一读取2026-11-18全量检索缓存中对应的500条，旧C参考来自相同缓存的全量重排结果。','',
           '## 同条件对照','',
           'C保持Qwen3-Reranker-0.6B、BF16、batch16、max_tokens1536。查询精确复用旧C使用的原报告文本，即S0语义编码实际使用的512-token预算内文本，去掉编码指令包装。功能推测和结构匹配分不输入C提示词。',
           '每个文件仍选择A最高分块，再用B最高分的不同块补足最多3块；文件分数取C块分数最大值，同分按该方案的AB名次。',
           '对旧、新候选池所需代码块取并集，在两张4090上重新评分。一个报告内相同查询—代码块对只算一次，两组共用这次分数；旧、新组都有本轮C结果，不直接用不同批次的历史分数作主要配对对照。历史成绩另列以监测BF16批次差异。',
           'A/B检索、推测和代码向量复用，C分数全部重新计算。1条已有B生成失败保留计零，不替换、不回退；完整性和基础设施错误停止运行并可续跑。','',
           '## 结果','',
           '| 方案 | 成功/总数 | MAP@10 | MRR@10 | Top1 | Top5 | Top10 | 平均标签召回@10 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in ('old_AB','new_AB','old_ABC','new_ABC','old_ABC_historical'):
        r=summary[key];m=r['metrics']
        lines.append(f"| {names[key]} | {r['successful_samples']}/500 | {m['MAP@10']:.6f} | {m['MRR@10']:.6f} | {m['Top-1']:.2%} | {m['Top-5']:.2%} | {m['Top-10']:.2%} | {m['MeanRecall@10']:.6f} |")
    lines+=['','## 配对差异','', '| 对比 | ΔMAP | MAP差值95%区间 | AP胜/平/负 | Holm校正p |', '|---|---:|---|---|---:|']
    for key,label in [('new_ABC_vs_old_ABC','新AB+C − 旧AB+C'),('new_ABC_vs_new_AB','新AB+C − 新AB')]:
        x=stats[key]['average_precision_at_k'];lo,hi=x['ci95_project_stratified_bootstrap']
        lines.append(f"| {label} | {x['delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {x['wins']}/{x['ties']}/{x['losses']} | {x['p_holm_10_tests']:.6g} |")
    lines+=['','统计为5,000次项目内分层配对bootstrap和配对随机化。Holm校正覆盖两项对比的MAP、MRR、Top1、Top10、MeanRecall10，共10项。','',
            '新AB+C相对新AB的MAP点估计增加0.028025，Top1净增9条、Top10净增8条；但MAP配对区间仍包含零，多项检验校正后也未通过。因此应记录为本批观察到的改善，不能表述为C在新基线上已被证实稳定增益。','',
            '| 对比 | 首位纠正/改坏 | Top10补回/丢失 |','|---|---:|---:|']
    for key,label in [('new_ABC_vs_old_ABC','新AB+C 对旧AB+C'),('new_ABC_vs_new_AB','新AB+C 对新AB')]:
        d=diagnostics[key]
        lines.append(f"| {label} | {d['top_1']['rescued']}/{d['top_1']['lost']} | {d['top_10']['rescued']}/{d['top_10']['lost']} |")
    lines+=['','## 候选与排序机制','',
            f"共同成功报告中，两组最终候选平均重合 {diagnostics['mean_common_candidates']:.2f}/50 个；{len(same_pool)} 条候选集合完全相同。在相同候选集合内，{diagnostics['same_pool_changed_final_order']} 条最终完整排序不同，{diagnostics['same_pool_changed_ap']} 条AP不同；由于代码块分数共用，这些变化仅由C同分时采用不同AB先验次序产生。",'',
            '新AB的结构匹配分没有直接进入C的最终打分，只影响候选与同分顺序。因此，AB初排的增益不能直接累加到C结果。新旧候选共同文件的C输入和评分严格相同，便于区分候选选择与重排作用。','',
            '## 核验与成本','',
            f"共重新评分 {verified['fresh_scored_pairs']:,} 个查询—代码块对。GPU0/1分别250份报告，{verified['success']}份成功、失败位置{verified['failed_positions']}。两卡模型身份与旧C一致。样本身份、冻结AB输出、旧证据选择、所有输入/结果校验和、文件Max聚合和同分规则、逐报告指标均已核验。",'',
            f"本轮含输入准备和统计的墙钟约 {diagnostics['wall_seconds_including_preparation']/60:.2f} 分钟；各卡逐报告时间总和约 {diagnostics['sum_worker_case_seconds']/60:.2f} 分钟。不包括历史推测生成、源码编码或语义检索，且共享机器环境不作为公平速度基准。",'',
            f"与历史C分数比较，{diagnostics['historical_identical_scores']:,}/{diagnostics['historical_total_scores']:,} 个共同对分数逐位相同，最大绝对分差 {diagnostics['historical_max_absolute_delta']:.6f}。主要结论使用本轮新旧配对结果，历史行仅作参考。",'',
            '## 产物','',
            '- [完整比较CSV](../results/ab_structure_c/20261123_fixed500/comparison.csv)。',
            '- [配对统计](../results/ab_structure_c/20261123_fixed500/paired_statistics.json)。',
            '- [逐项目结果](../results/ab_structure_c/20261123_fixed500/project_summary.json)。',
            '- [改善/退化案例与机制统计](../results/ab_structure_c/20261123_fixed500/diagnostics.json)。',
            '- [核验](../results/ab_structure_c/20261123_fixed500/verification.json)。',
            '- 配置：`configs/experiments/ab_structure_c500.json`；入口：`scripts/experiments/run_ab_structure_c500.py`。',
            '- `prepare`准备新目录，`run`双卡执行并自动汇总；同目录再次`run`会校验并跳过完整检查点。','']
    (ROOT/'docs/ab-structure-c500.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in diagnostics.items() if k not in ('largest_gains','largest_losses')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
