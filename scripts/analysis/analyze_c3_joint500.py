"""Paired 500-report cascade analysis and observable ranking-change attribution.

No model calls; explanations are saved as qualitative evidence, not causal proof.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.paired_statistics import align, analyze, values
from bug_localization.snapshot import sha256_file

C3 = ROOT / 'results/chunk_ablation_reranking/20261112_C2_C3_500/C3'
DIRECT = ROOT / 'results/direct8b_reranking/20261121_rrf500_joint10'
RRF = ROOT / 'results/module_ablation/20261116_rrf500'
BL = ROOT / 'results/blagent/recovery-20261118/dual-v4/full2294'
PILOT = ROOT / 'results/c3_improvement_pilot/20261119'


def read(path):
    return json.loads(path.read_text())


def checked(folder, record):
    if record['status'] != 'ok':
        return None
    path = folder / record['result_file']
    if sha256_file(path) != record['result_sha256']:
        raise ValueError('Result changed: ' + str(path))
    result = read(path)
    ranking = [r['file_path'] if isinstance(r, dict) else r for r in result['ranking']]
    gold = result.get('ground_truth', result.get('gold_files'))
    if ranking_metrics(ranking, gold, 10) != record['metrics']:
        raise ValueError('Independent metric mismatch')
    return {**result, 'paths': ranking, 'gold': gold}


def test_path(path):
    parts = Path(path.lower()).parts
    return any(p in ('test', 'tests', 'testing') for p in parts) or Path(path).name.startswith('test_')


def metric(record, name):
    return record['metrics'][name] if record['status'] == 'ok' else 0.


def table(summary):
    names = {'C3_06B': '仅0.6B（原C3）', 'direct_8B': '直接8B（RRF前十）',
             'cascade_06B_8B': '0.6B → 8B联合重排', 'BLAgent': '本地BLAgent 8B适配',
             'RRF_06B': 'RRF顺序＋0.6B'}
    lines = ['| 方法 | 成功/总数 | MAP@10 | MRR@10 | Top-1 | Top-5 | Top-10 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, row in summary.items():
        m = row['metrics']
        lines.append(f'| {names[name]} | {row["successful_samples"]}/{row["selected_samples"]} | '
                     f'{m["MAP@10"]:.6f} | {m["MRR@10"]:.6f} | {m["Top-1"]:.2%} | '
                     f'{m["Top-5"]:.2%} | {m["Top-10"]:.2%} |')
    return '\n'.join(lines)


def run(folder):
    audit = read(folder / 'audit/analysis.json')
    if audit['status'] != 'complete' or audit['runs']['joint10']['audit_status'] != 'passed':
        raise ValueError('Requires complete independent audit')
    joint = folder / 'joint10'
    selection = read(joint / 'selection.json')['samples']
    if len(selection) != 500:
        raise ValueError('Require the fixed500')
    baseline = read(C3 / 'samples.json'); direct = read(DIRECT / 'samples.json'); cascade = read(joint / 'samples.json')
    methods = {'C3_06B': baseline, 'direct_8B': direct, 'cascade_06B_8B': cascade}
    align(methods)
    bl_index = {r['dataset_row']: r for r in read(BL / 'samples.json')}
    methods['BLAgent'] = [{**bl_index[s['dataset_row']], **{k: s[k] for k in
                           ('position', 'query_sha256', 'labels_sha256')}} for s in selection]
    methods['RRF_06B'] = read(RRF / 'samples.json')['A+B+C']
    align(methods)
    explored = {s['position'] for name in ('development45.json', 'confirmation90.json') for s in read(PILOT / name)['samples']}
    groups = {'all500': set(range(1, 501)), 'previous180': set(range(1, 181)), 'new320': set(range(181, 501)),
              'excluding_pilot135': set(range(1, 501)) - explored}
    summary = {label: {name: aggregate_results([r for r in rows if r['position'] in positions], 10)
                       for name, rows in methods.items()} for label, positions in groups.items()}
    write_json(folder / 'comparison.json', summary)
    write_json(folder / 'project_summary.json', {repo: {name: aggregate_results([r for r in rows if r['repository'] == repo], 10)
               for name, rows in methods.items()} for repo in sorted({r['repository'] for r in cascade})})
    stats = analyze(methods, {'cascade_minus_06B': {'cascade_06B_8B': 1, 'C3_06B': -1},
                             'cascade_minus_direct8B': {'cascade_06B_8B': 1, 'direct_8B': -1}}, repetitions=5000)
    write_json(folder / 'paired_statistics.json', stats)
    details = []; packets = {}; calls = []; query_groups = defaultdict(list); transitions = Counter()
    promoted_ranks = Counter(); direct_groups = defaultdict(list); input_checks = Counter(); test_transitions = Counter()
    for old, dr, new in zip(baseline, direct, cascade):
        pos = new['position']; case = joint / 'cases' / f'{pos:04d}'
        a, d, c = checked(C3, old), checked(DIRECT, dr), checked(joint, new)
        calls.extend(read(p) for p in sorted(case.glob('call_*.json')))
        delta = metric(new, 'average_precision_at_k') - metric(old, 'average_precision_at_k')
        direct_delta = metric(new, 'average_precision_at_k') - metric(dr, 'average_precision_at_k')
        oldhit, newhit = bool(metric(old, 'top_1')), bool(metric(new, 'top_1'))
        transition = ('kept_correct' if newhit else 'broken') if oldhit else ('corrected' if newhit else 'still_wrong')
        transitions[transition] += 1
        row = {k: new[k] for k in ('position', 'dataset_row', 'repository', 'issue_id')}
        row.update(status=new['status'], error=new.get('error'), delta_ap=delta, delta_ap_vs_direct=direct_delta,
                   top1_transition=transition)
        if a:
            inp = read(case / 'inference_input.json') if (case / 'inference_input.json').exists() else None
            if inp is None:
                raise ValueError('Missing joint inference inputs')
            group = 'same_report' if inp['query'] == a['query'] else 'extended_report'
            if group == 'extended_report' and not inp['query'].startswith(a['query']):
                raise ValueError('Query changed beyond restoring full report')
            query_groups[group].append(row)
            row.update(query_group=group, old_top1=a['paths'][0], new_top1=c['paths'][0] if c else None,
                       old_first_gold_rank=next((i for i,p in enumerate(a['paths'],1) if p in a['gold']), None),
                       gold=a['gold'], old_top10=a['paths'][:10], new_top10=c['paths'][:10] if c else [])
            if transition == 'corrected':
                promoted_ranks[row['old_first_gold_rank']] += 1
                test_transitions['corrected_from_test_path' if test_path(a['paths'][0]) else 'corrected_from_other_path'] += 1
            if c and test_path(a['paths'][0]) and not test_path(c['paths'][0]):
                test_transitions['test_top1_to_non_test'] += 1
            if c and not test_path(a['paths'][0]) and test_path(c['paths'][0]):
                test_transitions['non_test_top1_to_test'] += 1
            dp = DIRECT / 'cases' / f'{pos:04d}' / 'inference_input.json'
            di = read(dp)
            input_checks['same_full_report' if di['query'] == inp['query'] else 'different_full_report'] += 1
            common = set(inp['candidates']) & set(di['candidates'])
            same_evidence = all(inp['evidence'][p] == di['evidence'][p] for p in common)
            input_checks['same_evidence_for_common_files' if same_evidence else 'different_common_evidence'] += 1
            row['candidate_overlap_with_direct'] = len(common)
            gold = set(a['gold']); cg = gold & set(inp['candidates']); dg = gold & set(di['candidates'])
            row.update(cascade_head_gold=sorted(cg), direct_head_gold=sorted(dg))
            if c and d:
                membership = 'same_gold_membership' if cg == dg else 'different_gold_membership'
                direct_groups['both_success_' + membership].append(row)
            else:
                direct_groups['cascade_' + new['status'] + '_direct_' + dr['status']].append(row)
            packets[pos] = {'report': inp['query'], 'reasoning': c['reasoning'] if c else new.get('error'),
                            'evidence': inp['evidence'], 'row': row}
        details.append(row)
    def group_summary(rows, delta_key='delta_ap'):
        return {'n': len(rows), 'delta_map_within_group': sum(r[delta_key] for r in rows)/len(rows),
                'contribution_to_all500_delta_map': sum(r[delta_key] for r in rows)/500,
                'wins': sum(r[delta_key] > 1e-12 for r in rows),
                'ties': sum(abs(r[delta_key]) <= 1e-12 for r in rows),
                'losses': sum(r[delta_key] < -1e-12 for r in rows)}
    attribution = {'selected': 500, 'top1_transitions': dict(transitions),
                   'first_gold_original_rank_for_corrected_top1': dict(sorted(promoted_ranks.items())),
                   'report_groups': {k: group_summary(v) for k,v in query_groups.items()},
                   'test_path_diagnostic': dict(test_transitions), 'direct_vs_cascade_input_checks': dict(input_checks),
                   'direct_comparison_decomposition': {k: group_summary(v, 'delta_ap_vs_direct') for k,v in direct_groups.items()},
                   'limitations': ['Descriptive groups are not causal effect estimates', 'Test path classification uses path names only',
                                  'Model explanations may contain errors even when the ranked file is correct'], 'samples': details}
    write_json(folder / 'attribution.json', attribution)
    cost = {'calls': len(calls), 'input_tokens': sum(c.get('input_tokens',0) for c in calls),
            'output_tokens': sum(c.get('output_tokens',0) for c in calls),
            'incremental_wall_seconds': read(joint / 'summary.json')['wall_seconds'],
            'failures': dict(Counter(r.get('error') for r in cascade if r['status'] != 'ok')),
            'note': 'Incremental 8B inference over cached C3; not end-to-end cost'}
    write_json(folder / 'cost.json', cost)
    available = [r for r in details if r['position'] in packets]
    gains = sorted([r for r in available if r['delta_ap'] > 1e-12], key=lambda r:(-r['delta_ap'],r['position']))[:8]
    losses = sorted([r for r in available if r['delta_ap'] < -1e-12], key=lambda r:(r['delta_ap'],r['position']))[:8]
    unchanged = sorted([r for r in available if abs(r['delta_ap']) <= 1e-12],
                       key=lambda r: hashlib.sha256(str((r['repository'],r['issue_id'])).encode()).hexdigest())[:4]
    review = folder / 'review'; review.mkdir(exist_ok=True)
    manifest = []
    for kind, rows in [('largest_gains',gains), ('largest_losses',losses), ('unchanged_identity_hash',unchanged)]:
        for row in rows:
            p = packets[row['position']]; name = f'{row["position"]:04d}.md'
            text = [f'# {row["repository"]} #{row["issue_id"]}', '', f'抽样类别：{kind}；ΔAP={row["delta_ap"]:+.6f}', '',
                    '## 排名变化', '', '```json', json.dumps(row,ensure_ascii=False,indent=2), '```', '',
                    '## 原始报告', '', p['report'], '', '## 8B输出解释（模型自述，需核对）', '', p['reasoning'], '', '## 实际输入证据', '']
            for path, chunks in p['evidence'].items():
                text.extend(['### ' + path, ''])
                for chunk in chunks:
                    text.extend(['```python', chunk['text'], '```', ''])
            (review / name).write_text('\n'.join(text) + '\n')
            manifest.append({'kind':kind, 'position':row['position'], 'path':str(review/name), 'delta_ap':row['delta_ap']})
    write_json(review / 'selection.json', manifest)
    with (folder / 'comparison.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['cohort','method','successful','MAP@10','MRR@10','Top1','Top5','Top10'])
        for group, rows in summary.items():
            for name, item in rows.items():
                m=item['metrics'];writer.writerow([group,name,item['successful_samples'],m['MAP@10'],m['MRR@10'],m['Top-1'],m['Top-5'],m['Top-10']])
    paragraphs=['# 0.6B → 8B联合重排：500条及排名变化分析','',
                '固定历史开发500条，双卡、双并发，复用0.6B阶段后全新运行8B；全部失败计零。', '', table(summary['all500']), '']
    for label, metrics in stats['contrasts'].items():
        s=metrics['average_precision_at_k'];ci=s['ci95_project_stratified_bootstrap']
        paragraphs.extend([f'{label}：ΔMAP={s["delta"]:+.6f}，95%项目内分层配对区间[{ci[0]:+.6f},{ci[1]:+.6f}]，'
                           f'AP胜/平/负={s["wins"]}/{s["ties"]}/{s["losses"]}，12项Holm校正p={s["p_holm_all_contrasts_and_metrics"]:.6f}。',''])
    paragraphs.extend(['## 8B改变了什么','',
                       f'相对原C3，首位从错变对{transitions["corrected"]}条，从对变错{transitions["broken"]}条，'
                       f'保持正确{transitions["kept_correct"]}条，仍未命中{transitions["still_wrong"]}条。', '',
                       f'被纠正样本的正确文件原先排名分布：{dict(sorted(promoted_ranks.items()))}。', '',
                       '每条成功输出均保持原C3前十集合及其后尾部不变。因此8B带来的是已召回文件的头部排序变化，并未新增召回；输出失败会降低总体Top10覆盖。', '',
                       '报告长度分组只表示相关性：', '', '```json', json.dumps(attribution['report_groups'],ensure_ascii=False,indent=2), '```', '',
                       '若原报告未扩展的样本也改善，可以排除“全部增益都来自恢复报告长度”；仍不能拆开模型大小、提示词与联合比较的各自贡献。', '',
                       f'文件路径启发式统计：{dict(test_transitions)}。这不是人工确认的错误机制发生率。', '',
                       '## 与直接8B的差异','',f'输入一致性检查：{dict(input_checks)}。', '',
                       '```json',json.dumps(attribution['direct_comparison_decomposition'],ensure_ascii=False,indent=2),'```','',
                       '原C3与RRF的0.6B同分排序规则有区别，完整表单列两种0.6B基线。直接8B使用RRF前十，级联使用0.6B前十；不同候选会同时改变覆盖及竞争关系。',''])
    for group in ('previous180','new320','excluding_pilot135'):
        paragraphs.extend(['## ' + group,'',table(summary[group]),''])
    paragraphs.extend(['## 成本、失败及限制','',
                       f'新增8B阶段墙钟{cost["incremental_wall_seconds"]/60:.2f}分钟，{cost["calls"]}次调用；'
                       f'输入{cost["input_tokens"]:,}、输出{cost["output_tokens"]:,} tokens。','',
                       f'失败：{cost["failures"]}。','',
                       '代码及逐例输入、实际请求、模型响应、排列、完整分母指标均经独立分析器核验。20例证据包见review/；按最大增益、最大损失和身份哈希抽样，不能代表整体机制频率。', '',
                       '这是历史开发数据上的描述性分析，不是全新盲测；8B解释可能包含错误，不应由文件排对反推根因解释正确。'])
    (folder / 'report.md').write_text('\n'.join(paragraphs)+'\n')
    write_json(folder / 'analysis_manifest.json', {'script_sha256':sha256_file(Path(__file__)),
               'audit_sha256':sha256_file(folder/'audit/analysis.json'), 'selected':500, 'review_cases':len(manifest)})
    print(table(summary['all500']))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run(args.run.resolve())
