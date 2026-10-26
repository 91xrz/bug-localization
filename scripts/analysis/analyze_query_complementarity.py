"""Offline rescue/loss analysis and label-informed diagnostic selector ceilings."""

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.bm25 import ranking_metrics
from bug_localization.batch import write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(folder):
    names = ('q0', 'q1', 'q2', 'q0_bm25', 'original_rrf', 'multiview_rrf')
    selection = read(folder / 'selection.json')['samples']
    records = {name: read(folder / name / 'samples.json') for name in names}
    if not selection or any(len(rows) != len(selection) for rows in records.values()):
        raise ValueError('Selection and channel sample counts must match')
    cases, flat = [], []
    for i, sample in enumerate(selection):
        results, metrics = {}, {}
        for name in names:
            row = records[name][i]
            if row['status'] != 'ok' or any(row[k] != sample[k] for k in ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha')):
                raise ValueError('All inputs must be successful and paired')
            result = read(folder / row['result_file'])
            paths = [r['file_path'] for r in result['ranking']]
            if len(set(paths)) != len(paths) or [r['rank'] for r in result['ranking']] != list(range(1, len(paths) + 1)):
                raise ValueError('Invalid input ranking')
            metrics[name] = ranking_metrics(paths, result['ground_truth'], 10)
            if metrics[name] != row['metrics'] or metrics[name] != result['metrics']:
                raise ValueError('Stored metrics differ from rankings')
            results[name] = result
        gold = set(results['q0']['ground_truth'])
        if any(set(r['ground_truth']) != gold or r['candidate_fingerprint'] != results['q0']['candidate_fingerprint'] for r in results.values()):
            raise ValueError('Candidate or gold inventory mismatch')
        gold_ranks = {name: {r['file_path']: r['rank'] for r in result['ranking'] if r['file_path'] in gold} for name, result in results.items()}
        cases.append({'issue_id': sample['issue_id'], 'metrics': metrics, 'gold_ranks': gold_ranks})
        flat.append({'issue_id': sample['issue_id'], 'repository': sample['repository'],
                     **{f'{name}_{key}': value for name, values in metrics.items() for key, value in values.items()},
                     **{name + '_gold_ranks': json.dumps(ranks, sort_keys=True) for name, ranks in gold_ranks.items()}})
    n = len(cases)
    comparisons = {}
    for label, bases, alternatives in (
        ('q1_vs_q0', ('q0',), ('q1',)), ('q2_vs_q0', ('q0',), ('q2',)),
        ('q1_or_q2_vs_q0', ('q0',), ('q1', 'q2')),
        ('q1_vs_original_bm25', ('q0_bm25',), ('q1',)),
        ('views_vs_original_channels', ('q0', 'q0_bm25'), ('q1', 'q2')),
        ('views_vs_original_rrf', ('original_rrf',), ('q1', 'q2'))):
        comparisons[label] = {}
        for k in (1, 5, 10):
            rescue, lost, both, neither = [], [], [], []
            extra_gold = []
            for case in cases:
                base = any(case['metrics'][name]['top_' + str(k)] for name in bases)
                alt = any(case['metrics'][name]['top_' + str(k)] for name in alternatives)
                (both if base and alt else rescue if alt else lost if base else neither).append(case['issue_id'])
                base_gold = {p for name in bases for p, rank in case['gold_ranks'][name].items() if rank <= k}
                alt_gold = {p for name in alternatives for p, rank in case['gold_ranks'][name].items() if rank <= k}
                extra_gold.extend({'issue_id': case['issue_id'], 'file_path': p} for p in sorted(alt_gold - base_gold))
            comparisons[label][str(k)] = {'rescued': rescue, 'lost': lost, 'both_hit': both, 'both_miss': neither,
                                           'extra_gold_report_file_pairs': extra_gold}
    metrics_keys = ('average_precision_at_k', 'reciprocal_rank_at_k', 'top_1', 'top_5', 'top_10', 'recall_at_k')
    ceilings = {}
    for label, routes in (('q0_only', ('q0',)), ('q0_q1', ('q0', 'q1')), ('q0_q2', ('q0', 'q2')),
                          ('all_views', ('q0', 'q1', 'q2')), ('original_channels', ('q0', 'q0_bm25')),
                          ('original_plus_views', ('q0', 'q0_bm25', 'q1', 'q2'))):
        choices = [max(routes, key=lambda r: c['metrics'][r]['average_precision_at_k']) for c in cases]
        ceilings[label] = {
            'independent_metric_ceiling': {key: sum(max(c['metrics'][r][key] for r in routes) for c in cases) / n for key in metrics_keys},
            'AP_selected_coherent_metrics': {key: sum(c['metrics'][r][key] for c, r in zip(cases, choices)) / n for key in metrics_keys},
            'AP_selected_routes': {r: choices.count(r) for r in routes}}
    paired = {}
    for name in ('q1', 'q2', 'multiview_rrf'):
        paired[name] = {}
        for key in ('average_precision_at_k', 'reciprocal_rank_at_k'):
            deltas = [c['metrics'][name][key] - c['metrics']['q0'][key] for c in cases]
            paired[name][key] = {'wins': [c['issue_id'] for c, d in zip(cases, deltas) if d > 1e-12],
                                'losses': [c['issue_id'] for c, d in zip(cases, deltas) if d < -1e-12],
                                'ties': sum(abs(d) <= 1e-12 for d in deltas), 'mean_delta': sum(deltas) / n}
    output = {'samples': n, 'comparisons': comparisons, 'oracle_ceilings': ceilings, 'paired_vs_q0': paired,
              'warning': 'Oracle uses ground truth per report; NOT achievable or validated selector performance. Independent metric maxima may choose different routes. OR hit counts do not define a fused ranking; top-K unions may contain more than K candidates.',
              'cases': cases}
    write_json(folder / 'complementarity.json', output)
    with (folder / 'complementarity_samples.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    output = analyze(PROJECT / args.run)
    print(json.dumps({k: v for k, v in output.items() if k != 'cases'}, ensure_ascii=False, indent=2))
