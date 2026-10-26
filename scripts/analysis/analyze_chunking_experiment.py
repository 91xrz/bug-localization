"""Audit a paired chunking run and export exploratory effect sizes/cost diagnostics."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def paired_effect(a, b, indices):
    delta = np.asarray(a) - np.asarray(b)
    means = delta[indices].mean(axis=1)
    return {'delta': float(delta.mean()), 'ci95': np.quantile(means, [0.025, 0.975]).tolist(),
            'wins': int((delta > 1e-12).sum()), 'ties': int((abs(delta) <= 1e-12).sum()),
            'losses': int((delta < -1e-12).sum())}


def analyze(folder, reference=None):
    summary = read(folder / 'summary.json')
    selection = read(folder / 'selection.json')
    routes = list(summary['routes'])
    records = {r: read(folder / r / 'samples.json') for r in routes}
    count = len(selection['samples'])
    if summary['file_aggregation'] != 'max' or summary.get('fusion'):
        raise ValueError('The frozen experiment requires Max and no RRF')
    keys = ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha')
    expected = [tuple(s[k] for k in keys) for s in selection['samples']]
    output = {'run': str(folder), 'selected_samples': count, 'selection_sha256': summary['selection_sha256'],
              'scope': 'Exploratory, previously observed single-project reports; not a held-out evaluation',
              'bootstrap': {'seed': 20261025, 'replicates': 20000, 'unit': 'report',
                            'caution': 'Reports share a project and snapshots; intervals are descriptive, not cross-project evidence'},
              'routes': {}, 'effects': {}, 'baseline_audit': {}}
    reference_records = {r: read(reference / r / 'samples.json') for r in ('dynamic', 'blagent_ast')} if reference else {}
    if reference and read(reference / 'selection.json')['samples'] != selection['samples']:
        raise ValueError('Historical baseline has a different sample selection')
    for route in routes:
        data = records[route]
        if [tuple(s[k] for k in keys) for s in data] != expected:
            raise ValueError('Sample pairing mismatch: ' + route)
        if any(r['status'] != 'ok' for r in data):
            raise ValueError('Resolve failed samples before claiming an improvement: ' + route)
        independently_aggregated = aggregate_results(data, 10)['metrics']
        if independently_aggregated != summary['routes'][route]['metrics']:
            raise ValueError('Summary differs from per-report metrics')
        tokens, max_error, vector_misses, cache_hits, top10_mismatches, gold_ranks = [], 0, 0, 0, 0, []
        multi, clipped, splits, original_cost, window_limited = 0, 0, 0, 0, 0
        baseline_metric_matches, baseline_rank_matches = 0, 0
        for i, record in enumerate(data):
            result_path = folder / record['result_file']
            result = read(result_path)
            chunks = read(result_path.with_name(route + '_chunks.json'))
            candidate = read(result_path.parent / 'candidates.json')
            if route != routes[0]:
                first = read(folder / records[routes[0]][i]['result_file'])
                if (result['candidate_fingerprint'] != first['candidate_fingerprint']
                        or result['ground_truth'] != first['ground_truth']):
                    raise ValueError('Different candidates or labels')
            assert len(candidate['files']) == result['indexed_files']
            assert len(chunks['chunks']) == result['indexed_chunks']
            check = result['verification']
            assert check['all_chunk_scores_verified'] and check['all_file_scores_verified']
            assert check['same_chunk_scores_for_all_aggregations']
            max_error = max(max_error, check['max_cosine_error'])
            top10_mismatches += sum(not v['top_10_matches_local_order'] for v in check['aggregations'].values())
            vector_misses += result['new_embeddings']
            cache_hits += result['cache_hits']
            for chunk in chunks['chunks']:
                tokens.append(chunk['token_count'])
                assert chunk['token_count'] <= 512 and chunk['prefix_tokens'] <= 64
                multi += len(chunk['structure']['signatures']) > 1
                clipped += chunk['prefix_truncated']
                splits += bool(chunk['split_reason'])
            original_cost += sum(p.get('packing_cost', 0) for p in chunks['original_blocks'])
            window_limited += sum(p.get('atom_count') == 32 for p in chunks['original_blocks'])
            gold = set(result['ground_truth'])
            gold_ranks.append({'issue_id': record['issue_id'], 'AP@10': record['metrics']['average_precision_at_k'],
                               'gold_ranks': {v['file_path']: v['rank'] for v in result['ranking'] if v['file_path'] in gold}})
            if route in reference_records:
                old = read(reference / reference_records[route][i]['result_file'])
                baseline_metric_matches += result['metrics'] == old['metrics']
                baseline_rank_matches += [v['file_path'] for v in result['ranking']] == [v['file_path'] for v in old['ranking']]
        if route in reference_records:
            output['baseline_audit'][route] = {'matching_metrics': baseline_metric_matches,
                                               'matching_full_rankings': baseline_rank_matches, 'total': count}
        output['routes'][route] = {'metrics': independently_aggregated, 'chunk_instances': len(tokens),
                                   'median_tokens': float(np.median(tokens)), 'mean_tokens': float(np.mean(tokens)),
                                   'under128_fraction': sum(t < 128 for t in tokens) / len(tokens),
                                   'multi_signature_chunks': multi, 'clipped_prefixes': clipped,
                                   'fallback_or_secondary_split_chunks': splits, 'packing_cost': original_cost,
                                   'blocks_reaching_32_atom_window': window_limited,
                                   'route_elapsed_seconds': sum(r['elapsed_seconds'] for r in data),
                                   'new_vectors_in_this_run': vector_misses, 'cache_hits': cache_hits,
                                   'max_cosine_error': max_error, 'aggregation_top10_order_mismatches': top10_mismatches,
                                   'gold_ranks': gold_ranks}
    indices = np.random.default_rng(20261025).integers(0, count, size=(20000, count))
    for candidate, baseline in [('ast_token_dp', 'blagent_ast'), ('ast_token_dp', 'dynamic'),
                                ('ast_token_dp', 'ast_token_greedy'), ('ast_token_greedy', 'blagent_ast')]:
        output['effects'][candidate + ' - ' + baseline] = {
            metric: paired_effect([r['metrics'][field] for r in records[candidate]],
                                  [r['metrics'][field] for r in records[baseline]], indices)
            for metric, field in [('MAP@10', 'average_precision_at_k'), ('MRR@10', 'reciprocal_rank_at_k')]}
    output['cost_caution'] = 'Existing baselines are warm; new-vector counts depend on route order. Do not claim speedups from this run.'
    write_json(folder / 'chunking_analysis.json', output)
    rows = [{'route': route, **values['metrics'], **{k: v for k, v in values.items() if k not in ('metrics', 'gold_ranks')}}
            for route, values in output['routes'].items()]
    with (folder / 'chunking_analysis.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in output.items() if k not in ('routes',)}, indent=2))
    print(json.dumps({r: {k: v for k, v in values.items() if k != 'gold_ranks'} for r, values in output['routes'].items()}, indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference', type=Path, help='Historical 44-report baseline for unchanged-route audit')
    args = parser.parse_args()
    analyze(args.run, args.reference)
