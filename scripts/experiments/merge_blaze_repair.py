"""Validate and merge a BLAZE repair run into its original frozen selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_checkpoints, write_json
from bug_localization.blaze_experiment import ROUTES
from bug_localization.bm25 import ranking_metrics
from bug_localization.snapshot import sha256_file


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def merge(original, retry, output):
    original, retry, output = map(Path, (original, retry, output))
    full, sub = read(original / 'selection.json'), read(retry / 'selection.json')
    if full['dataset_metadata_sha256'] != sub['dataset_metadata_sha256']:
        raise ValueError('Dataset metadata differs')
    orig_config, retry_config = (read(path / 'effective_config.json') for path in (original, retry))
    for field in ('endpoint', 'cluster_name'):
        orig_config['base'].pop(field)
        retry_config['base'].pop(field)
    if orig_config['base'] != retry_config['base']:
        raise ValueError('Candidate/query/evaluation protocol differs')
    original_prov, retry_prov = (read(path / 'provenance.json') for path in (original, retry))
    if original_prov['model'] != retry_prov['model'] or original_prov['server_version'] != retry_prov['server_version']:
        raise ValueError('Model or Elasticsearch version differs')
    if original_prov['implementation_sha256'] != retry_prov['implementation_sha256']:
        raise ValueError('Experiment implementation differs between runs')
    for field in ('source_max_tokens', 'report_max_tokens', 'dense_depth', 'lexical_depth', 'rrf_constant'):
        if orig_config['blaze'][field] != retry_config['blaze'][field]:
            raise ValueError('BLAZE parameter differs: ' + field)
    original_records = {r: read(original / r / 'samples.json') for r in ROUTES}
    retry_records = {r: read(retry / r / 'samples.json') for r in ROUTES}
    if any(len(original_records[r]) != len(full['samples']) or len(retry_records[r]) != len(sub['samples']) for r in ROUTES):
        raise ValueError('Incomplete original or retry run; merge only after all attempts')
    fields = ('dataset_row', 'repository', 'issue_id', 'before_fix_sha')
    lookup = {tuple(s[k] for k in fields): i for i, s in enumerate(full['samples'])}
    if len(lookup) != len(full['samples']):
        raise ValueError('Duplicate original report identities')
    original_indices = []
    for i, sample in enumerate(sub['samples']):
        key = tuple(sample[k] for k in fields)
        if key not in lookup or sample['position'] != i + 1:
            raise ValueError('Retry sample does not match original')
        j = lookup[key]
        for k in ('query_sha256', 'labels_sha256'):
            if sample[k] != full['samples'][j][k]:
                raise ValueError('Retry query/label identity changed')
        original_indices.append(j)
    if len(set(original_indices)) != len(original_indices):
        raise ValueError('Duplicate retry sample')
    expected = {i for i in range(len(full['samples'])) if any(original_records[r][i]['status'] != 'ok' for r in ROUTES)}
    if set(original_indices) != expected:
        raise ValueError('Retry selection must cover exactly original failed reports')
    retry_lookup = {j: i for i, j in enumerate(original_indices)}
    combined = {r: [] for r in ROUTES}
    counts = {r: {'reused_success': 0, 'retry': 0} for r in ROUTES}
    for j, identity in enumerate(full['samples']):
        for route in ROUTES:
            original_record = original_records[route][j]
            for k in fields:
                if original_record[k] != identity[k]:
                    raise ValueError('Original record identity changed')
            use_retry = original_record['status'] != 'ok'
            folder = retry if use_retry else original
            record = dict(retry_records[route][retry_lookup[j]] if use_retry else original_record)
            if any(record[k] != identity[k] for k in fields):
                raise ValueError('Retry record identity changed')
            source_position = record['position']
            record['position'] = identity['position']
            record['source_run'] = str(folder.resolve())
            record['source_position'] = source_position
            for f in ('result_file', 'error_file'):
                if record.get(f):
                    source = (folder / route / record[f]).resolve()
                    if not source.is_file() or not source.is_relative_to(folder.resolve()):
                        raise ValueError('Missing or unsafe source artifact')
                    record[f] = str(source)
            if record['status'] == 'ok':
                result = read(Path(record['result_file']))
                if any(result[k] != identity[k] for k in ('repository', 'issue_id', 'before_fix_sha')):
                    raise ValueError('Case result identity mismatch')
                actual = ranking_metrics([item['file_path'] for item in result['routes'][route]['ranking']],
                                         result['gold_files'], record['metrics']['k'])
                if actual != record['metrics'] or result['missing_gold'] != record['missing_gold']:
                    raise ValueError('Saved metrics or gold coverage do not match ranking')
            counts[route]['retry' if use_retry else 'reused_success'] += 1
            combined[route].append(record)
    output.mkdir(parents=True, exist_ok=False)
    summary = {'selection_sha256': full['selection_sha256'], 'dataset_metadata_sha256': full['dataset_metadata_sha256'],
               'source_runs': {'original': str(original.resolve()), 'retry': str(retry.resolve())},
               'selection_file_sha256': sha256_file(original / 'selection.json'),
               'retry_selection_file_sha256': sha256_file(retry / 'selection.json'),
               'method': 'Reuse successful route results and substitute retry results only for failed routes',
               'service_change': 'Isolated Elasticsearch 9.5.5; identical BM25 settings; full untruncated query',
               'route_source_counts': counts, 'routes': {}}
    write_json(output / 'selection.json', full)
    for route in ROUTES:
        (output / route).mkdir()
        route_summary = aggregate_results(combined[route], orig_config['base']['top_k'], len(full['samples']))
        summary['routes'][route] = route_summary
        write_checkpoints(output / route, combined[route], route_summary)
    summary['status'] = 'completed_with_failures' if any(x['failed_samples'] for x in summary['routes'].values()) else 'completed'
    write_json(output / 'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'routes': {r: {'successful': s['successful_samples'],
          'failed': s['failed_samples'], 'metrics': s['metrics']} for r, s in summary['routes'].items()}}, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--retry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    merge(args.original, args.retry, args.output)
