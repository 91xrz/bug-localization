"""L0/S0/H0 on the frozen 500 reports; reuse and verify semantic rankings."""
import argparse
import csv
import hashlib
from pathlib import Path
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import run, ranking_metrics, prepare_report
from bug_localization.embeddings import digest
from bug_localization.reranking import read_json
from bug_localization.snapshot import sha256_file
from run_mixed_recall500 import Sources, same_sample


def verify_result(result, reference, inventory):
    if result['candidate_files'] != inventory['files']:
        raise ValueError('BM25/semantic candidate inventories differ')
    if result['ground_truth'] != reference['ground_truth']:
        raise ValueError('BM25/semantic labels differ')
    metrics = ranking_metrics([r['file_path'] for r in result['results']], result['ground_truth'], 10)
    if metrics != result['metrics']:
        raise ValueError('BM25 metrics do not match saved ranks')
    return metrics


def summarize(folder, records):
    cohorts = {'all500': (0, 500), 'new320': (180, 500), 'previous180': (0, 180)}
    summary, projects = {}, {}
    for cohort, (start, end) in cohorts.items():
        groups = {name: rows[start:end] for name, rows in records.items()}
        summary[cohort] = {name: aggregate_results(rows, 10, end-start) for name, rows in groups.items()}
        projects[cohort] = {repo: {name: aggregate_results([r for r in rows if r['repository'] == repo], 10)
                           for name, rows in groups.items()}
                           for repo in sorted({r['repository'] for r in groups['S0']})} if len(groups['L0']) == end-start else {}
    complete = len(records['L0']) == 500
    status = ('completed_with_failures' if any(r['status'] == 'failed' for rows in records.values() for r in rows)
              else 'completed') if complete else 'running'
    write_json(folder / 'samples.json', records)
    write_json(folder / 'final_summary.json', {'status': status, 'cohorts': summary})
    write_json(folder / 'project_summary.json', projects)
    with (folder / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['cohort', 'method', 'successful', 'failed', 'pending',
            'MAP@10', 'MRR@10', 'Top-1', 'Top-5', 'Top-10', 'MeanRecall@10'])
        writer.writeheader()
        for cohort, groups in summary.items():
            for name, stats in groups.items():
                writer.writerow({'cohort': cohort, 'method': name, 'successful': stats['successful_samples'],
                    'failed': stats['failed_samples'], 'pending': stats['pending_samples'], **(stats['metrics'] or {})})
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = read_json(PROJECT / 'configs/experiments/recall_baselines500.json')
    sources = Sources(config)
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    base = read_json(sources.roots[0][0] / 'effective_config.json')['base']
    records = {'L0': [], 'S0': [], 'H0': []}
    metadata = list(sources.metadata)
    references = []
    for sample in sources.samples:
        roots, pos, rows, _ = sources.locate(sample)
        case = roots[0] / 'cases' / f'{pos:03d}'
        reference = read_json(case / 'q0_result.json')
        inventory = read_json(case / 'candidates.json')
        references.append((reference, inventory))
        metadata.append(case / 'candidates.json')
        for name, root, row, filename in zip(('S0', 'H0'), roots, rows, ('q0_result.json', 'direct_result.json')):
            if row['status'] == 'ok':
                path = root / 'cases' / f'{pos:03d}' / filename
                result = read_json(path)
                metadata.append(path)
                metrics = ranking_metrics([r['file_path'] for r in result['ranking']], result['ground_truth'], 10)
                if metrics != row['metrics'] or metrics != result['metrics']:
                    raise ValueError('Semantic ranking/metric disagreement')
                if result['ground_truth'] != reference['ground_truth']:
                    raise ValueError('Semantic labels differ')
                if {r['file_path'] for r in result['ranking']} != {r['file_path'] for r in inventory['files']}:
                    raise ValueError('Semantic candidates differ')
            records[name].append({**row, **sample, 'source_root': str(root)})
    identity = {'config': config, 'base': base, 'sources': {str(p): sha256_file(p) for p in metadata},
                'implementation': {str(p): sha256_file(p) for p in (Path(__file__), PROJECT / 'src/bug_localization/bm25.py')}}
    identity_path = folder / 'effective_config.json'
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError('Inputs or implementation changed; use a new output directory')
    write_json(identity_path, identity)
    write_json(folder / 'selection.json', sources.selection)
    if args.check_only:
        print('Verified all 500 sample identities and S0/H0 rankings; H0 failures retained.', flush=True)
        return
    from datasets import load_from_disk
    from elasticsearch import Elasticsearch
    data = load_from_disk(str(PROJECT / 'datasets' / base['dataset']))[base['split']]
    with Elasticsearch(base['endpoint'], request_timeout=60) as client:
        if client.info()['cluster_name'] != base['cluster_name']:
            raise ValueError('Wrong Elasticsearch cluster')
        for sample, (reference, inventory) in zip(sources.samples, references):
            case = folder / 'L0' / f"{sample['position']:03d}"
            case.mkdir(parents=True, exist_ok=True)
            row = data[sample['dataset_row']]
            current = {**base, 'repository': sample['repository'], 'issue_id': sample['issue_id'], 'top_k': 10}
            import json
            query_hash = hashlib.sha256(json.dumps({f: row.get(f) for f in base['query_fields']}, sort_keys=True).encode()).hexdigest()
            same_sample(sample, {**sample, 'repository': row['repo_name'], 'issue_id': row['issue_id'],
                'before_fix_sha': row['before_fix_sha'], 'query_sha256': query_hash,
                'labels_sha256': hashlib.sha256(str(row['updated_files']).encode()).hexdigest()})
            if (case / 'result.json').exists():
                result = read_json(case / 'result.json')
            else:
                # Preserve every pre-existing index. Only this case's newly created owned index is removed.
                existing = set(client.indices.get(index='bl-bm25-*', allow_no_indices=True))
                result = run(PROJECT, current, report=row, output_directory=case, verbose=False)
                index = result['index']
                if index not in existing:
                    expected = {'project': 'bug-localization', 'repository': sample['repository'],
                                'commit': sample['before_fix_sha'], 'fingerprint': result['index_fingerprint']}
                    if client.indices.get_mapping(index=index)[index]['mappings'].get('_meta') != expected:
                        raise ValueError('Temporary index ownership mismatch')
                    client.indices.delete(index=index)
            if result['query_text'] != prepare_report(row, current)[2] or any(result[k] != sample[k] for k in ('repository', 'issue_id', 'before_fix_sha')):
                raise ValueError('BM25 report identity mismatch')
            metrics = verify_result(result, reference, inventory)
            records['L0'].append({**sample, 'status': 'ok', 'metrics': metrics,
                'missing_gold': result['ground_truth_missing_from_candidates'],
                'result_file': str((case / 'result.json').relative_to(folder))})
            status = summarize(folder, records)
            print(f"[{sample['position']}/500] {sample['repository']} #{sample['issue_id']} L0 AP@10={metrics['average_precision_at_k']:.6f}", flush=True)
    write_json(folder / 'verification.json', {'status': 'passed', 'samples': 500,
        'checks': ['frozen sample/query/label identity', 'same candidate file hashes for L0/S0',
                   'semantic candidate paths and labels', 'all successful route metrics recomputed from rankings'],
        'source_fingerprint': digest(identity)})
    print(status, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        if '--output' in sys.argv:
            folder = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
            if folder.is_dir():
                write_json(folder / 'error.json', {'status': 'failed', 'error_type': type(error).__name__,
                    'error': str(error), 'traceback': traceback.format_exc()})
                summary_path = folder / 'final_summary.json'
                if summary_path.exists():
                    summary = read_json(summary_path)
                    summary['status'] = 'failed'
                    summary['error'] = str(error)
                    write_json(summary_path, summary)
        sys.exit(1)
