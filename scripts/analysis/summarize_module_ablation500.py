"""Remove C from the saved pipeline, preserving its pre-reranker file order."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / 'src'), str(PROJECT / 'scripts/experiments')]
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.mixed_recall import mixed_candidates
from bug_localization.reranking import read_json
from bug_localization.snapshot import sha256_file
from run_mixed_recall500 import Sources, same_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def read(path):
        path = Path(path).resolve()
        hashes[str(path)] = sha256_file(path)
        return read_json(path)

    config = read(PROJECT / 'configs/experiments/mixed_recall_reranker500.json')
    sources = Sources(config)
    c3root = PROJECT / 'results/chunk_ablation_reranking/20261112_C2_C3_500/C3'
    c3rows = read(c3root / 'samples.json')
    baseline = read(PROJECT / 'results/recall_baselines/20261112_L0_S0_H0_500/samples.json')
    for path in sources.metadata:
        hashes[str(path.resolve())] = sha256_file(path)
    if len(c3rows) != 500:
        raise ValueError('Incomplete C3')
    for key in ('S0', 'H0'):
        if len(baseline[key]) != 500:
            raise ValueError('Incomplete baseline')
    write_json(out / 'selection.json', sources.selection)
    write_json(out / 'protocol.json', {
        'modules': {'A': 'Original semantic retrieval (S0)', 'B': 'Hypothesis retrieval (H0)',
                    'A+B': 'C3 candidate pool in original semantic order, without evidence selection/reranking',
                    'A+B+C': 'Existing C3'},
        'pool': 'Hypothesis Top-40, unique original padding to 50',
        'output_order': 'Original semantic order within the pool, matching C3 pre-reranker order',
        'failure_policy': 'Both routes required for A+B; failures zero, denominator 500',
        'scope': 'Offline ablation on existing development data; not a newly tuned fusion or independent test',
        'top10_invariant': '40+10 pool includes original Top-10; insertion-order output instead equals hypothesis Top-10',
        'new_model_inference': False})
    write_json(out / 'status.json', {'status': 'running'})
    records = {m: [] for m in ('A', 'B', 'A+B', 'A+B+C')}
    diagnostics = []
    for sample, c3row in zip(sources.samples, c3rows):
        position = sample['position']
        same_sample(sample, c3row)
        if c3row['position'] != position:
            raise ValueError('C3 position mismatch')
        roots, local, source_rows, _ = sources.locate(sample)
        results = []
        for module, key, root, filename, row in zip(
                ('A', 'B'), ('S0', 'H0'), roots, ('q0_result.json', 'direct_result.json'), source_rows):
            old = baseline[key][position - 1]
            same_sample(sample, old)
            if old['status'] != row['status']:
                raise ValueError('Baseline status mismatch')
            result = None
            if row['status'] == 'ok':
                result = read(root / 'cases' / f'{local:03d}' / filename)
                paths = [r['file_path'] for r in result['ranking']]
                if len(set(paths)) != len(paths):
                    raise ValueError('Duplicate file ranking')
                metrics = ranking_metrics(paths, result['ground_truth'], 10)
                if metrics != row['metrics'] or metrics != result['metrics'] or metrics != old['metrics']:
                    raise ValueError('Baseline metrics mismatch')
            records[module].append({**old, **sample})
            results.append(result)
        a, b = results
        if b is None:
            if position != 308 or c3row['status'] != 'failed' or a is None:
                raise ValueError('Unexpected failure')
            records['A+B'].append({**sample, 'status': 'failed',
                'error_type': 'SourceDependencyFailure', 'error': source_rows[1].get('error', 'Hypothesis failed')})
        else:
            if a is None or a['ground_truth'] != b['ground_truth'] or c3row['status'] != 'ok':
                raise ValueError('Unpaired successful sources')
            candidates = mixed_candidates(a['ranking'], b['ranking'], 'hypothesis', 40, 50)
            # Independently reconstruct the pool and original-order output.
            primary = [r['file_path'] for r in b['ranking'][:40]]
            expected = primary + [r['file_path'] for r in a['ranking'] if r['file_path'] not in set(primary)][:max(0, 50-len(primary))]
            if expected != [r['file_path'] for r in candidates]:
                raise ValueError('Independent pool reconstruction mismatch')
            pool = set(expected)
            ranking = [r['file_path'] for r in a['ranking'] if r['file_path'] in pool]
            if ranking != [r['file_path'] for r in sorted(candidates, key=lambda r: r['original_rank'])]:
                raise ValueError('Pre-reranker order mismatch')
            if ranking[:10] != [r['file_path'] for r in a['ranking'][:10]]:
                raise ValueError('Original Top-10 invariant broken')
            c3path = c3root / c3row['result_file']
            c3 = read(c3path)
            if hashes[str(c3path.resolve())] != c3row['result_sha256']:
                raise ValueError('C3 checksum mismatch')
            same_sample(sample, c3)
            if c3['candidates'] != candidates or c3['ground_truth'] != a['ground_truth']:
                raise ValueError('C3 pool or labels differ')
            for root, filename in zip(roots, ('q0_result.json', 'direct_result.json')):
                path = str((root / 'cases' / f'{local:03d}' / filename).resolve())
                if c3['source_files_sha256'][path] != hashes[path]:
                    raise ValueError('C3 retrieval source checksum mismatch')
            if {r['file_path'] for r in c3['ranking']} != pool or len(c3['ranking']) != len(pool):
                raise ValueError('C3 output inventory mismatch')
            c3metrics = ranking_metrics([r['file_path'] for r in c3['ranking']], c3['ground_truth'], 10)
            if c3metrics != c3['metrics'] or c3metrics != c3row['metrics']:
                raise ValueError('C3 metric mismatch')
            metrics = ranking_metrics(ranking, a['ground_truth'], 10)
            record = {**sample, 'status': 'ok', 'metrics': metrics,
                      'missing_gold': sorted(set(a['ground_truth']) - {r['file_path'] for r in a['ranking']}),
                      'result_file': f'cases/{position:03d}.json'}
            (out / 'cases').mkdir(exist_ok=True)
            write_json(out / record['result_file'], {**sample, 'candidates': candidates,
                'ranking': [{'file_path': p, 'rank': i} for i, p in enumerate(ranking, 1)],
                'ground_truth': a['ground_truth'], 'metrics': metrics})
            records['A+B'].append(record)
            gold = set(a['ground_truth'])
            at = {r['file_path'] for r in a['ranking'][:10]}
            bt = {r['file_path'] for r in b['ranking'][:10]}
            diagnostics.append({**sample, 'a_hit': bool(at & gold), 'b_hit': bool(bt & gold),
                'top10_overlap': len(at & bt), 'a_only_gold': sorted((at-bt)&gold),
                'b_only_gold': sorted((bt-at)&gold), 'pool_recall': len(pool & gold)/len(gold)})
        records['A+B+C'].append(c3row)
        if position % 100 == 0:
            print(f'Verified {position}/500', flush=True)
    cohorts = {'all500': (0, 500), 'previous180': (0, 180), 'new320': (180, 500)}
    summary = {name: {m: aggregate_results(rows[start:end], 10, end-start) for m, rows in records.items()}
               for name, (start, end) in cohorts.items()}
    projects = {repo: {m: aggregate_results([r for r in rows if r['repository'] == repo], 10)
                       for m, rows in records.items()} for repo in sorted({r['repository'] for r in sources.samples})}
    write_json(out / 'samples.json', records)
    write_json(out / 'final_summary.json', summary)
    write_json(out / 'project_summary.json', projects)
    write_json(out / 'diagnostics.json', {'samples': diagnostics,
        'hit_patterns_successful499': dict(Counter(f"A={r['a_hit']},B={r['b_hit']}" for r in diagnostics)),
        'mean_top10_overlap': sum(r['top10_overlap'] for r in diagnostics)/len(diagnostics)})
    fields = ['cohort', 'method', 'successful', 'failed', 'MAP@10', 'MRR@10', 'Top-1', 'Top-5', 'Top-10', 'MeanRecall@10']
    with (out / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cohort, methods in summary.items():
            for method, stats in methods.items():
                writer.writerow({'cohort': cohort, 'method': method, 'successful': stats['successful_samples'],
                                 'failed': stats['failed_samples'], **stats['metrics']})
    for path in (Path(__file__), PROJECT/'src/bug_localization/mixed_recall.py',
                 PROJECT/'src/bug_localization/bm25.py', PROJECT/'src/bug_localization/batch.py',
                 PROJECT/'scripts/experiments/run_mixed_recall500.py'):
        hashes[str(path.resolve())] = sha256_file(path)
    write_json(out / 'source_hashes.json', hashes)
    write_json(out / 'verification.json', {'status': 'passed', 'samples': 500,
        'successful_ab_top10_equals_a': 499, 'expected_failure_positions': [308], 'unexpected_failures': [],
        'checks': ['sample identity', 'source hashes against C3', 'independent candidate reconstruction',
                   'same pool as C3', 'actual pre-reranker order', 'all four methods metrics recomputed']})
    write_json(out / 'status.json', {'status': 'completed_with_failures', 'processed': 500,
        'failed_by_method': {m: summary['all500'][m]['failed_samples'] for m in records}})
    print(out)


if __name__ == '__main__':
    main()
