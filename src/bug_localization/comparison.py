"""Pinned-sample BM25 versus three semantic routes, with no fusion."""

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .batch import run_batch, write_json
from .bm25 import validate_config
from .vector import RETRIEVAL_ROUTES, run_semantic_batch, validate_vector_config

COMPARISON_ROUTES = ('bm25',) + RETRIEVAL_ROUTES


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def paired_records(lexical_records, semantic_records, routes=None):
    expected = [(r['position'], r['dataset_row'], r['repository'], r['issue_id'], r['before_fix_sha'])
                for r in lexical_records]
    if not expected:
        raise ValueError('Comparison cannot contain zero samples')
    for route in (routes if routes is not None else [r for r in semantic_records if r in RETRIEVAL_ROUTES]):
        actual = [(r['position'], r['dataset_row'], r['repository'], r['issue_id'], r['before_fix_sha'])
                  for r in semantic_records[route]]
        if actual != expected:
            raise ValueError('Comparison samples or order differ: ' + route)


def compare_runs(project, folder, lexical_folder, semantic_folder):
    lexical = read_json(lexical_folder / 'summary.json')
    semantic = read_json(semantic_folder / 'summary.json')
    semantic_routes = tuple(semantic.get('chunking_routes', semantic['routes']))
    comparison_routes = ('bm25',) + semantic_routes
    if (not semantic_routes or not set(semantic_routes).issubset(RETRIEVAL_ROUTES)
            or set(semantic['routes']) != set(semantic_routes) or semantic.get('fusion')):
        raise ValueError('This comparison must not contain RRF results')
    if lexical['selection_sha256'] != semantic['selection_sha256']:
        raise ValueError('BM25/semantic selection fingerprints differ')
    selection = read_json(lexical_folder / 'selection.json')
    if selection['samples'] != read_json(semantic_folder / 'selection.json')['samples']:
        raise ValueError('BM25/semantic selection manifests differ')
    all_records = {'bm25': read_json(lexical_folder / 'samples.json')}
    all_records.update({route: read_json(semantic_folder / route / 'samples.json') for route in semantic_routes})
    paired_records(all_records['bm25'], all_records, semantic_routes)
    summaries = {'bm25': lexical, **semantic['routes']}
    n = len(selection['samples'])
    cutoffs = {values['cutoff'] for values in summaries.values()}
    if len(cutoffs) != 1:
        raise ValueError('Comparison metrics use different evaluation cutoffs')
    verified_cases, unverified_cases, truncated_queries = 0, [], 0
    per_sample = []
    for position in range(n):
        records = {route: values[position] for route, values in all_records.items()}
        successful = {route: record for route, record in records.items() if record['status'] == 'ok'}
        if len(successful) == len(comparison_routes):
            bm = read_json(lexical_folder / records['bm25']['result_file'])
            reference_route = semantic_routes[0]
            semantic_case = (semantic_folder / records[reference_route]['result_file']).parent
            candidate = read_json(semantic_case / 'candidates.json')['files']
            query = read_json(semantic_case / 'query.json')
            if candidate != bm['candidate_files'] or query['body'] != bm['query_text']:
                raise ValueError('Compared routes have different candidates or raw reports')
            for route in semantic_routes:
                result = read_json(semantic_folder / records[route]['result_file'])
                if (result['before_fix_sha'] != bm['before_fix_sha'] or set(result['ground_truth']) != set(bm['ground_truth']) or
                        result['missing_gold'] != bm['ground_truth_missing_from_candidates'] or
                        result['indexed_files'] != len(candidate)):
                    raise ValueError('Compared routes have different snapshots, labels or candidate coverage')
                if result['candidate_fingerprint'] != read_json(semantic_folder / records[reference_route]['result_file'])['candidate_fingerprint']:
                    raise ValueError('Semantic candidate fingerprints differ')
            truncated_queries += int(query['truncated'])
            verified_cases += 1
        else:
            unverified_cases.append({'position': position + 1, 'failed_routes': sorted(set(comparison_routes) - set(successful))})
        for route, record in records.items():
            metrics = record.get('metrics') or {}
            per_sample.append({'position': record['position'], 'route': route, 'repository': record['repository'],
                               'issue_id': record['issue_id'], 'before_fix_sha': record['before_fix_sha'],
                               'status': record['status'], 'AP@10': metrics.get('average_precision_at_k'),
                               'RR@10': metrics.get('reciprocal_rank_at_k'), 'Top-1': metrics.get('top_1'),
                               'Top-5': metrics.get('top_5'), 'Top-10': metrics.get('top_10'),
                               'Recall@10': metrics.get('recall_at_k'), 'gold_files': metrics.get('gold_files'),
                               'missing_gold_files': len(record['missing_gold']) if record['status'] == 'ok' else None,
                               'elapsed_seconds': record.get('elapsed_seconds'), 'error': record.get('error')})
    if next(iter(cutoffs)) != 10:
        raise ValueError('This comparison export currently requires evaluation_k=10')
    report = {'status': 'completed' if all(s['failed_samples'] == 0 for s in summaries.values()) else 'completed_with_failures',
              'selected_samples': n, 'evaluation_k': 10, 'fusion': False,
              'selection_sha256': lexical['selection_sha256'], 'scope': 'Expanded single-project comparison; not full SWE-bench',
              'failure_policy': lexical['failure_policy'],
              'runs': {'bm25': str(lexical_folder.relative_to(project)), 'semantic': str(semantic_folder.relative_to(project))},
              'routes': {route: {key: values[key] for key in ('selected_samples', 'successful_samples', 'failed_samples',
                         'metrics', 'metrics_successful_only', 'coverage', 'failures')} for route, values in summaries.items()}}
    verification = {'same_sample_order_and_snapshots': True, 'candidate_and_query_verified_samples': verified_cases,
                    'not_fully_verified_samples': unverified_cases, 'semantic_query_truncated_samples_in_verified_cases': truncated_queries,
                    'no_fusion': True, 'metric_cutoffs_match': True}
    write_json(folder / 'selection.json', selection)
    write_json(folder / 'summary.json', report)
    write_json(folder / 'verification.json', verification)
    write_json(folder / 'samples.json', per_sample)
    rows = [{'route': route, 'selected_samples': n, 'successful_samples': values['successful_samples'],
             'failed_samples': values['failed_samples'], **values['metrics']} for route, values in summaries.items()]
    for name, data in [('summary.csv', rows), ('samples.csv', per_sample)]:
        with (folder / name).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    return report


def run_comparison(project, config):
    base_path = Path(config['bm25_config'])
    base_path = base_path if base_path.is_absolute() else project / base_path
    semantic_path = Path(config['vector_config'])
    semantic_path = semantic_path if semantic_path.is_absolute() else project / semantic_path
    base = read_json(base_path)
    vector = read_json(semantic_path)
    vector_base = Path(vector['base_config'])
    vector_base = vector_base if vector_base.is_absolute() else project / vector_base
    if read_json(vector_base) != base:
        raise ValueError('Comparison must share exactly the same candidate/base configuration')
    if config['evaluation_k'] != 10 or base['top_k'] != 10:
        raise ValueError('Comparison requires evaluation_k=10 and BM25 top_k=10')
    vector.update({'evaluation_k': 10, 'enable_fusion': False})
    validate_config(base)
    validate_vector_config(vector, base)
    run_id = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    folder = project / 'results/comparison' / run_id
    folder.mkdir(parents=True)
    write_json(folder / 'effective_config.json', config)
    write_json(folder / 'status.json', {'status': 'running', 'stage': 'bm25'})
    print('Comparison output: ' + str(folder), flush=True)
    lexical_folder, _ = run_batch(project, base, config['repositories'], config['limit'])
    write_json(folder / 'status.json', {'status': 'running', 'stage': 'semantic', 'bm25_run': str(lexical_folder)})
    vector['selection_manifest'] = str(lexical_folder / 'selection.json')
    semantic_folder, _ = run_semantic_batch(project, vector)
    report = compare_runs(project, folder, lexical_folder, semantic_folder)
    write_json(folder / 'status.json', {'status': report['status'], 'stage': 'finished', 'runs': report['runs']})
    print(json.dumps({route: values['metrics'] for route, values in report['routes'].items()}, indent=2), flush=True)
    print('Comparison results: ' + str(folder), flush=True)
    return folder, report
