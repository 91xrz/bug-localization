"""Audit and compare BLAZE, RRL and project methods on the frozen 500 reports."""

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.snapshot import sha256_file

R = PROJECT / 'results'
FULL_SELECTION = PROJECT / 'configs/experiments/selections/crossproject_500_seed20261111.json'
NEW_SELECTION = PROJECT / 'configs/experiments/selections/crossproject_new320_seed20261111.json'
BLAZE = R / 'blaze-repair-20261116/merged500'
BM25 = R / 'recall_baselines/20261112_L0_S0_H0_500'
SEMANTIC = R / 'hypothesis_reranking/20261111_153819_040009_500'
C3 = R / 'chunk_ablation_reranking/20261112_C2_C3_500/C3'
RRL_CPU = R / 'rrl/crossproject500-cpu-retry-20261116'
RRL_GPU = R / 'rrl/crossproject500-gpu1-remaining-20261116'
IDENTITY = ('dataset_row', 'repository', 'issue_id', 'before_fix_sha')
FIELDS = {'MAP@10': 'average_precision_at_k', 'MRR@10': 'reciprocal_rank_at_k',
          'Top-1': 'top_1', 'Top-5': 'top_5', 'Top-10': 'top_10', 'MeanRecall@10': 'recall_at_k'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def key(entry):
    return tuple(entry[x] for x in IDENTITY)


def compare_counts(left, right, keys):
    counts = Counter()
    for k in keys:
        a = left[k].get('metrics', {}).get('average_precision_at_k', 0) if left[k]['status'] == 'ok' else 0
        b = right[k].get('metrics', {}).get('average_precision_at_k', 0) if right[k]['status'] == 'ok' else 0
        diff = a - b
        counts['win' if diff > 1e-12 else 'loss' if diff < -1e-12 else 'tie'] += 1
    return {name: counts[name] for name in ('win', 'tie', 'loss')}


def means(mapping, keys):
    records = [mapping[k] for k in keys]
    successful = [r for r in records if r['status'] == 'ok']
    return {'samples': len(records), 'successful': len(successful), 'failed': len(records) - len(successful),
            'metrics': {name: sum(r['metrics'][field] for r in successful) / len(records)
                        for name, field in FIELDS.items()}}


def run(output):
    selection, new = read(FULL_SELECTION), read(NEW_SELECTION)
    entries = selection['samples']
    if len(entries) != 500 or len({key(x) for x in entries}) != 500:
        raise ValueError('Expected 500 distinct frozen reports')
    keys = [key(x) for x in entries]
    sample_by_key = {key(x): x for x in entries}
    if read(BLAZE / 'selection.json')['selection_sha256'] != selection['selection_sha256']:
        raise ValueError('BLAZE selection differs')
    for folder in (BM25, SEMANTIC):
        if read(folder / 'selection.json')['selection_sha256'] != selection['selection_sha256']:
            raise ValueError('Project selection differs: ' + str(folder))
    original = read(BM25 / 'samples.json')
    semantic = read(SEMANTIC / 'paired_samples.json')['cohorts']['all500']
    methods = {'BLAZE BM25': read(BLAZE / 'blaze_bm25/samples.json'),
               'BLAZE dense': read(BLAZE / 'blaze_dense/samples.json'),
               'BLAZE RRF': read(BLAZE / 'blaze_rrf/samples.json'),
               'Project BM25 L0': original['L0'], 'Project semantic S0': original['S0'],
               'Project hypothesis H0': original['H0'], 'Project original rerank A': semantic['original_reranked'],
               'Project C3': read(C3 / 'samples.json'),
               'RRL 7B adaptation': read(RRL_CPU / 'samples.json') + read(RRL_GPU / 'samples.json')}
    maps = {}
    for name, rows in methods.items():
        mapping = {key(row): row for row in rows}
        if len(rows) != 500 or len(mapping) != 500 or set(mapping) != set(keys):
            raise ValueError('Report identities do not match: ' + name)
        for k, row in mapping.items():
            for field in ('query_sha256', 'labels_sha256'):
                if field in row and row[field] != sample_by_key[k][field]:
                    raise ValueError(f'{name} {field} mismatch at {k}')
            if row['status'] not in ('ok', 'failed'):
                raise ValueError('Unexpected status: ' + name)
            if row['status'] == 'ok' and row['metrics']['k'] != 10:
                raise ValueError('Wrong evaluation cutoff: ' + name)
        maps[name] = mapping
    # Audit identical report text, labels and full candidate inventory, not only aggregate counts.
    checked_rrl, absent_rrl = 0, []
    for i, entry in enumerate(entries, 1):
        k = key(entry)
        l0 = read(BM25 / f'L0/{i:03d}/result.json')
        blaze_record = maps['BLAZE RRF'][k]
        if blaze_record['status'] != 'ok':
            raise ValueError('Merged BLAZE is not complete')
        blaze = read(Path(blaze_record['result_file']))
        if (l0['query_text'] != blaze['query'] or l0['query_fields'] != blaze['query_fields'] or
                l0['candidate_files'] != blaze['candidates'] or l0['ground_truth'] != blaze['gold_files'] or
                l0['ground_truth_missing_from_candidates'] != blaze['missing_gold']):
            raise ValueError(f'BLAZE versus L0 inputs differ at original position {i}')
        for route in ('BLAZE BM25', 'BLAZE dense', 'BLAZE RRF'):
            rec = maps[route][k]
            if rec['metrics'] != ranking_metrics([x['file_path'] for x in
                    read(Path(rec['result_file']))['routes'][{'BLAZE BM25':'blaze_bm25',
                    'BLAZE dense':'blaze_dense', 'BLAZE RRF':'blaze_rrf'}[route]]['ranking']], blaze['gold_files'], 10):
                raise ValueError(f'BLAZE metrics mismatch at {i}: {route}')
        rec = maps['RRL 7B adaptation'][k]
        # RRL uses one-based positions within each of its CPU and GPU partitions.
        folder = RRL_CPU if i <= 3 else RRL_GPU
        pos = i if i <= 3 else i - 3
        inputs_path = folder / f'cases/{pos:04d}/inputs.json'
        if inputs_path.exists():
            inputs = read(inputs_path)
            if inputs['query'] != blaze['query'] or inputs['candidates'] != blaze['candidates']:
                raise ValueError(f'RRL input mismatch at {i}')
            checked_rrl += 1
        else:
            if rec['status'] == 'ok':
                raise ValueError(f'Successful RRL has no inputs at {i}')
            absent_rrl.append(i)
    new_keys = {key(x) for x in new['samples']}
    if len(new_keys) != 320 or not new_keys.issubset(keys):
        raise ValueError('The new-320 selection does not match the 500')
    cohorts = {'all500': keys, 'previous180': [k for k in keys if k not in new_keys],
               'new320': [k for k in keys if k in new_keys]}
    cohorts['RRL_success455'] = [k for k in keys if maps['RRL 7B adaptation'][k]['status'] == 'ok']
    comparisons = {}
    projects = {}
    for cohort, cohort_keys in cohorts.items():
        comparisons[cohort] = {name: means(mapping, cohort_keys) for name, mapping in maps.items()}
        if cohort == 'all500':
            for repository in sorted({k[1] for k in cohort_keys}):
                group = [k for k in cohort_keys if k[1] == repository]
                projects[repository] = {name: means(mapping, group) for name, mapping in maps.items()}
    paired = {}
    pairs = [('Project semantic S0', 'BLAZE RRF'), ('Project hypothesis H0', 'BLAZE RRF'),
             ('Project original rerank A', 'BLAZE RRF'), ('Project C3', 'BLAZE RRF'),
             ('RRL 7B adaptation', 'BLAZE RRF'), ('Project C3', 'RRL 7B adaptation'),
             ('BLAZE BM25', 'BLAZE RRF'), ('BLAZE dense', 'BLAZE RRF')]
    for left, right in pairs:
        both = [k for k in keys if maps[left][k]['status'] == maps[right][k]['status'] == 'ok']
        paired[left + ' vs ' + right] = {'all500_failed_zero': compare_counts(maps[left], maps[right], keys),
                                       'both_success_count': len(both),
                                       'both_success': compare_counts(maps[left], maps[right], both),
                                       'delta_map_all500': comparisons['all500'][left]['metrics']['MAP@10'] - comparisons['all500'][right]['metrics']['MAP@10']}
    result = {'selection_sha256': selection['selection_sha256'],
              'identity_fields': list(IDENTITY), 'cohorts': comparisons, 'projects': projects, 'paired': paired,
              'audit': {'BLAZE_L0_matching_query_candidates_gold': 500,
                        'RRL_matching_query_candidates': checked_rrl, 'RRL_missing_inputs_positions': absent_rrl,
                        'blaze_recomputed_metrics_per_route': 500,
                        'source_hashes': {str(path.relative_to(PROJECT)): sha256_file(path) for path in
                            [FULL_SELECTION, NEW_SELECTION, BLAZE / 'summary.json', BM25 / 'samples.json',
                             SEMANTIC / 'paired_samples.json', C3 / 'samples.json',
                             RRL_CPU / 'samples.json', RRL_GPU / 'samples.json']},
                        'failure_policy': 'Failed records contribute zero; successful-only sensitivity is separate',
                        'independent_test': False,
                        'caveats': ['BLAZE released encoder adaptation lacks training residual adapters',
                                    'RRL uses Qwen2.5-Coder-7B Q4_K_M; CPU first 3, CUDA remaining 497',
                                    'RRL can inspect README; different token limits, models and computation budget',
                                    '500 reports were used for method development; not an independent test set',
                                    'No statistical significance test performed']}}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'comparison.json', result)
    rows = []
    for cohort, methods_summary in comparisons.items():
        for name, info in methods_summary.items():
            rows.append({'cohort': cohort, 'method': name, 'samples': info['samples'],
                         'successful': info['successful'], 'failed': info['failed'], **info['metrics']})
    with (output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'project_summary.json', projects)
    write_json(output / 'paired_summary.json', paired)
    print(json.dumps({'all500': comparisons['all500'], 'audit': result['audit'], 'paired': paired}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output if args.output.is_absolute() else PROJECT / args.output)
