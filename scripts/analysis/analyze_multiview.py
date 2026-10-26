"""Audit multi-view rankings and compare unchanged q0 with an existing AST baseline."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.report_views import filter_evidence, build_views, weighted_rrf
from bug_localization.bm25 import ranking_metrics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(folder, reference=None):
    summary = read(folder / 'summary.json')
    selection = read(folder / 'selection.json')
    config = read(folder / 'effective_config.json')['multiview']
    records = {name: read(folder / name / 'samples.json') for name in summary['routes']}
    if any(any(row['status'] != 'ok' for row in rows) for rows in records.values()):
        raise ValueError('Resolve failed cases before claiming a complete paired experiment')
    reference_rows = None
    if reference:
        if read(reference / 'selection.json')['samples'] != selection['samples']:
            raise ValueError('Reference selection differs')
        reference_rows = read(reference / config['semantic_route'] / 'samples.json')
    audit = {'samples': len(selection['samples']), 'all_rankings_and_metrics_verified': True,
             'q0_historical_full_ranking_matches': 0, 'q0_historical_metric_matches': 0,
             'accepted_evidence': 0, 'rejected_evidence': 0, 'empty_q1': 0, 'empty_q2': 0,
             'generation_usage': {'prompt_tokens': 0, 'completion_tokens': 0}, 'effects': {}}
    for name, rows in records.items():
        if len(rows) != len(selection['samples']) or aggregate_results(rows, 10) != summary['routes'][name]:
            raise ValueError('Summary or sample count differs: ' + name)
    for i, selected in enumerate(selection['samples']):
        results = {name: read(folder / rows[i]['result_file']) for name, rows in records.items()}
        case = (folder / records['q0'][i]['result_file']).parent
        generated = read(case / 'queries.json')
        accepted, rejected = filter_evidence(generated['extraction'], generated['report'])
        views, evidence = build_views(generated['report'], accepted)
        if views != generated['views'] or evidence != generated['evidence'] or rejected != generated['rejected_evidence']:
            raise ValueError('Query evidence differs')
        audit['accepted_evidence'] += sum(len(v) for v in evidence.values())
        audit['rejected_evidence'] += len(rejected)
        for q in ('q1', 'q2'):
            audit['empty_' + q] += not views[q]
        for key in audit['generation_usage']:
            audit['generation_usage'][key] += (generated.get('usage') or {}).get(key, 0)
        q0 = results['q0']
        for name, result in results.items():
            if any(result[k] != selected[k] for k in ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha')):
                raise ValueError('Report pairing differs: ' + name)
            if result['candidate_fingerprint'] != q0['candidate_fingerprint'] or result['ground_truth'] != q0['ground_truth']:
                raise ValueError('Candidate or gold inventory differs')
            paths = [r['file_path'] for r in result['ranking']]
            if len(paths) != len(set(paths)) or result['metrics'] != ranking_metrics(paths, result['ground_truth'], 10):
                raise ValueError('Ranking metrics differ')
        recomputed = weighted_rrf({q: results[q]['ranking'] for q in ('q0', 'q1', 'q2')}, config['weights'], config['rrf_constant'])
        if recomputed != results['multiview_rrf']['ranking']:
            raise ValueError('Weighted RRF differs')
        original = weighted_rrf({'q0': q0['ranking'], 'bm25': results['q0_bm25']['ranking']}, {'q0': 1, 'bm25': 1}, config['rrf_constant'])
        if original != results['original_rrf']['ranking']:
            raise ValueError('Original RRF differs')
        for q in ('q0', 'q2'):
            if views[q]:
                verification = read(case / (q + '_retrieval.json'))['verification']
                if not verification['all_chunk_scores_verified'] or not verification['all_file_scores_verified']:
                    raise ValueError('ES/local scores not verified')
        if views['q2'] and read(case / 'q0_retrieval.json')['index'] != read(case / 'q2_retrieval.json')['index']:
            raise ValueError('Query views used different source indices')
        if reference_rows is not None:
            previous_path = reference / reference_rows[i]['result_file']
            previous = read(previous_path)
            if (read(previous_path.parent / 'query.json')['body'] != views['q0']
                    or previous['candidate_fingerprint'] != q0['candidate_fingerprint']):
                raise ValueError('Original query or candidate inventory changed')
            if ([r['file_path'] for r in previous['ranking']] != [r['file_path'] for r in q0['ranking']]
                    or previous['metrics'] != q0['metrics']):
                raise ValueError('Original AST baseline changed')
            audit['q0_historical_full_ranking_matches'] += 1
            audit['q0_historical_metric_matches'] += 1
    for comparator in ('q0', 'original_rrf'):
        delta = [a['metrics']['average_precision_at_k'] - b['metrics']['average_precision_at_k']
                 for a, b in zip(records['multiview_rrf'], records[comparator])]
        audit['effects'][comparator] = {'MAP@10_delta': sum(delta) / len(delta),
            'wins': sum(v > 1e-12 for v in delta), 'ties': sum(abs(v) <= 1e-12 for v in delta),
            'losses': sum(v < -1e-12 for v in delta)}
    projects = sorted({sample['repository'] for sample in selection['samples']})
    project_summary = {project: {name: aggregate_results([r for r in rows if r['repository'] == project], 10)
                                for name, rows in records.items()} for project in projects}
    write_json(folder / 'project_summary.json', project_summary)
    write_json(folder / 'verification.json', audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(PROJECT / args.run, PROJECT / args.reference if args.reference else None), indent=2))
