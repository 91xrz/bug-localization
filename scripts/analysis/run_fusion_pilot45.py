"""CPU-only, predeclared small fusion pilot using saved full rankings."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.snapshot import sha256_file
from compare_direct_fusions import fuse


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def read(path):
        path = Path(path).resolve()
        hashes[str(path)] = sha256_file(path)
        return json.loads(path.read_text())

    config = read(ROOT / 'configs/experiments/fusion_pilot45.json')
    all_samples = read(ROOT / config['selection'])['samples']
    eligible = [s for s in all_samples if config['eligible_positions'][0] <= s['position'] <= config['eligible_positions'][1]]
    selected = []
    for repo in sorted({s['repository'] for s in eligible}):
        rows = [s for s in eligible if s['repository'] == repo]
        selected.extend(random.Random(config['seed'] + ':' + repo).sample(rows, config['per_project']))
    selected.sort(key=lambda s: s['position'])
    assert len(selected) == config['expected_samples']
    # Freeze selection/protocol before reading any per-case outcome.
    write_json(out / 'selection.json', {'samples': selected})
    write_json(out / 'protocol.json', config)
    write_json(out / 'status.json', {'status': 'running'})
    root = ROOT / 'results/hypothesis_reranking/20261111_153819_040009_500/new320'
    orig_rows = read(root / 'semantic/samples.json')['q0']
    hyp_rows = read(root / 'hypothesis/samples.json')['direct_hypothesis']
    names = ['A', 'B'] + [m + '/' + scope for m in config['methods'] for scope in config['candidate_scopes']]
    records = {m: [] for m in names}
    pairs = []
    identity = ('dataset_row', 'repository', 'issue_id', 'before_fix_sha', 'query_sha256', 'labels_sha256')
    for sample in selected:
        pos = sample['position'] - 180
        source = [orig_rows[pos-1], hyp_rows[pos-1]]
        for row in source:
            assert all(sample[k] == row[k] for k in identity)
        cases = [read(root / folder / 'cases' / f'{pos:03d}' / filename) if row['status'] == 'ok' else None
                 for folder, filename, row in zip(('semantic', 'hypothesis'), ('q0_result.json', 'direct_result.json'), source)]
        a, b = cases
        results = {}
        for method, case, row in zip(('A', 'B'), cases, source):
            if case is None:
                records[method].append({**row, **sample})
                results[method] = {'status': 'failed'}
            else:
                paths = [r['file_path'] for r in case['ranking']]
                metrics = ranking_metrics(paths, case['ground_truth'], 10)
                assert metrics == case['metrics'] == row['metrics']
                records[method].append({**row, **sample})
                results[method] = {'status': 'ok', 'metrics': metrics, 'top10': paths[:10]}
        if a is None or b is None:
            for method in names[2:]:
                records[method].append({**sample, 'status': 'failed', 'error_type': 'SourceDependencyFailure', 'error': 'Required retrieval failed'})
                results[method] = {'status': 'failed'}
            pairs.append({**sample, 'results': results})
            continue
        gold = a['ground_truth']
        assert gold == b['ground_truth']
        apaths = [r['file_path'] for r in a['ranking']]
        bpaths = [r['file_path'] for r in b['ranking']]
        assert set(apaths) == set(bpaths)
        primary = bpaths[:40]
        pool_order = primary + [p for p in apaths if p not in set(primary)][:max(0, 50-len(primary))]
        pool = set(pool_order)
        # Scoring uses full-ranking normalization; pool variant only restricts output.
        fused = fuse(a['ranking'], b['ranking'])
        for method in config['methods']:
            for scope in config['candidate_scopes']:
                name = method + '/' + scope
                paths = fused[method] if scope == 'full' else [p for p in fused[method] if p in pool]
                assert len(set(paths)) == len(paths)
                assert set(paths) == (set(apaths) if scope == 'full' else pool)
                metrics = ranking_metrics(paths, gold, 10)
                records[name].append({**sample, 'status': 'ok', 'metrics': metrics, 'missing_gold': sorted(set(gold)-set(apaths))})
                results[name] = {'status': 'ok', 'metrics': metrics, 'top10': paths[:10]}
        pairs.append({**sample, 'ground_truth': gold, 'pool': pool_order,
            'top10_overlap': len(set(apaths[:10]) & set(bpaths[:10])),
            'gold_outside_pool': sorted(set(gold)-pool), 'results': results})
    summary = {m: aggregate_results(rows, 10, len(selected)) for m, rows in records.items()}
    comparisons = []
    ap = lambda r: r.get('metrics', {}).get('average_precision_at_k', 0)
    hit = lambda r: r.get('metrics', {}).get('top_10', 0)
    for method, stats in summary.items():
        changes = [ap(p['results'][method])-ap(p['results']['B']) for p in pairs]
        gained = sum(hit(p['results'][method]) and not hit(p['results']['B']) for p in pairs)
        lost = sum(hit(p['results']['B']) and not hit(p['results'][method]) for p in pairs)
        delta = stats['metrics']['MAP@10'] - summary['B']['metrics']['MAP@10']
        dmrr = stats['metrics']['MRR@10'] - summary['B']['metrics']['MRR@10']
        rule = config['screening_rule']
        passed = (delta >= rule['delta_MAP_vs_B_at_least'] and dmrr >= rule['delta_MRR_vs_B_at_least']
                  and lost-gained <= rule['max_net_Top10_hit_loss_vs_B'])
        comparisons.append({'method': method, **stats['metrics'], 'delta_MAP_vs_B': delta,
            'wins_vs_B': sum(x>1e-12 for x in changes), 'ties_vs_B': sum(abs(x)<=1e-12 for x in changes),
            'losses_vs_B': sum(x < -1e-12 for x in changes), 'Top10_gained_vs_B': gained,
            'Top10_lost_vs_B': lost, 'passes_pilot_screen': passed if method not in ('A','B') else False})
    projects = {repo: {m: aggregate_results([r for r in rows if r['repository']==repo], 10) for m, rows in records.items()}
                for repo in sorted({s['repository'] for s in selected})}
    write_json(out / 'summary.json', summary)
    write_json(out / 'project_summary.json', projects)
    write_json(out / 'paired_samples.json', pairs)
    write_json(out / 'comparison.json', comparisons)
    with (out / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader(); writer.writerows(comparisons)
    for path in (Path(__file__), ROOT/'scripts/analysis/compare_direct_fusions.py', ROOT/'src/bug_localization/bm25.py', ROOT/'src/bug_localization/batch.py'):
        hashes[str(path.resolve())] = sha256_file(path)
    write_json(out / 'source_hashes.json', hashes)
    write_json(out / 'verification.json', {'status': 'passed', 'samples': len(selected),
        'model_calls': 0, 'checks': ['fixed selection before outcomes', 'paired sample identity',
        'same source labels and inventories', 'source ranking metrics recomputed', 'unique full/pool rankings'],
        'failure_counts': {m: s['failed_samples'] for m, s in summary.items()}})
    write_json(out / 'status.json', {'status': 'completed', 'samples': len(selected), 'gpu_used': False, 'auto_expand': False})
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
