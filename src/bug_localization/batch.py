"""Deterministic small-batch BM25 evaluation with explicit failure denominators."""

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

from .bm25 import run, validate_config


def select_samples(rows, repositories, limit):
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError('Small-batch limit must be an integer between 1 and 50')
    if not isinstance(repositories, list) or not repositories or any(not isinstance(r, str) or not r for r in repositories):
        raise ValueError('repositories must be a non-empty list of repository names')
    selected, seen = [], set()
    for dataset_row, row in enumerate(rows):
        if row['repo_name'] not in repositories:
            continue
        key = (row['repo_name'], row['issue_id'])
        if key in seen:
            raise ValueError('Duplicate repository/issue in the selected samples: ' + str(key))
        seen.add(key)
        selected.append({'dataset_row': dataset_row, 'report': row})
        if len(selected) == limit:
            break
    if not selected:
        raise ValueError('No samples match the requested repositories')
    return selected


def mean_metrics(records, denominator, cutoff):
    fields = {'MAP@' + str(cutoff): 'average_precision_at_k',
              'MRR@' + str(cutoff): 'reciprocal_rank_at_k',
              'Top-1': 'top_1', 'Top-5': 'top_5', 'Top-10': 'top_10',
              'MeanRecall@' + str(cutoff): 'recall_at_k'}
    if not denominator:
        return None
    return {name: sum(record['metrics'][field] for record in records) / denominator
            for name, field in fields.items()}


def aggregate_results(records, cutoff, selected_count=None):
    if cutoff < 10:
        raise ValueError('Batch Top-10 reporting requires top_k >= 10')
    selected_count = len(records) if selected_count is None else selected_count
    if selected_count < len(records) or selected_count < 1:
        raise ValueError('Selected sample count must be positive and cover all attempted samples')
    if any(record['status'] not in ('ok', 'failed') for record in records):
        raise ValueError('Every attempted sample must be ok or failed')
    successes = [record for record in records if record['status'] == 'ok']
    if any(record['metrics']['k'] != cutoff for record in successes):
        raise ValueError('Cannot aggregate different ranking cutoffs')
    pending = selected_count - len(records)
    total_gold = sum(record['metrics']['gold_files'] for record in successes)
    missing_gold = sum(len(record['missing_gold']) for record in successes)
    return {
        'status': 'running' if pending else 'completed', 'cutoff': cutoff,
        'selected_samples': selected_count, 'attempted_samples': len(records),
        'successful_samples': len(successes), 'failed_samples': len(records) - len(successes),
        'pending_samples': pending,
        'failure_policy': 'Failed attempts contribute zero to primary metrics; successful-only metrics are also reported',
        'metrics': None if pending else mean_metrics(successes, selected_count, cutoff),
        'metrics_successful_only': mean_metrics(successes, len(successes), cutoff),
        'coverage': {'complete_samples': sum(not record['missing_gold'] for record in successes),
                     'incomplete_samples': sum(bool(record['missing_gold']) for record in successes),
                     'not_evaluated_samples': selected_count - len(successes),
                     'gold_files_in_successful_samples': total_gold, 'missing_gold_files': missing_gold,
                     'candidate_coverage': (total_gold - missing_gold) / total_gold if total_gold else None},
        'failures': [{'repository': record['repository'], 'issue_id': record['issue_id'],
                      'error_type': record['error_type'], 'error': record['error']} for record in records if record['status'] == 'failed'],
    }


def write_json(path, value):
    # A terminated run leaves the previous complete checkpoint intact.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_checkpoints(folder, records, summary):
    write_json(folder / 'samples.json', records)
    write_json(folder / 'summary.json', summary)
    fields = ['position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha', 'status',
              'indexed_files', 'gold_files', 'missing_gold_files', 'K', 'AP@K', 'RR@K', 'Top-1', 'Top-5', 'Top-10',
              'Recall@K', 'query_truncated', 'elapsed_seconds', 'result_file', 'error_type', 'error']
    with (folder / 'samples.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = record.get('metrics') or {}
            writer.writerow({**{field: record.get(field, '') for field in fields},
                             'gold_files': metrics.get('gold_files', ''),
                             'K': metrics.get('k', ''),
                             'missing_gold_files': len(record['missing_gold']) if record['status'] == 'ok' else '',
                             'AP@K': metrics.get('average_precision_at_k', ''),
                             'RR@K': metrics.get('reciprocal_rank_at_k', ''),
                             'Top-1': metrics.get('top_1', ''), 'Top-5': metrics.get('top_5', ''),
                             'Top-10': metrics.get('top_10', ''), 'Recall@K': metrics.get('recall_at_k', '')})
    with (folder / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        values = {field: summary[field] for field in ('status', 'selected_samples', 'successful_samples', 'failed_samples', 'pending_samples')}
        values.update(summary['metrics'] or {})
        writer = csv.DictWriter(handle, fieldnames=list(values))
        writer.writeheader()
        writer.writerow(values)


def run_batch(project, base, repositories, limit):
    validate_config(base)
    if base['top_k'] < 10:
        raise ValueError('Batch Top-10 reporting requires top_k >= 10')
    from datasets import load_from_disk
    from elasticsearch import Elasticsearch

    data = load_from_disk(str(project / 'datasets' / base['dataset']))[base['split']]
    selected = select_samples(data, repositories, limit)
    # Reject a global service problem before downloading any new code snapshots.
    with Elasticsearch(base['endpoint'], request_timeout=5) as client:
        info = dict(client.info())
        if info['cluster_name'] != base['cluster_name'] or not info['version']['number'].startswith('9.'):
            raise ValueError('Batch evaluation requires the configured Elasticsearch 9.x cluster')
    now = datetime.now(timezone(timedelta(hours=8)))
    run_id = base['dataset'] + '_' + base['split'] + '_' + now.strftime('%Y%m%d_%H%M%S_%f')
    folder = project / 'results/bm25/batches' / run_id
    folder.mkdir(parents=True)
    metadata_path = project / 'datasets' / base['dataset'] / base['split'] / 'dataset_info.json'
    metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    selection = [{'position': position, 'dataset_row': sample['dataset_row'],
                  'repository': sample['report']['repo_name'], 'issue_id': sample['report']['issue_id'],
                  'before_fix_sha': sample['report'].get('before_fix_sha'),
                  'query_sha256': hashlib.sha256(json.dumps({field: sample['report'].get(field) for field in base['query_fields']}, sort_keys=True).encode()).hexdigest(),
                  'labels_sha256': hashlib.sha256(str(sample['report'].get('updated_files')).encode()).hexdigest()}
                 for position, sample in enumerate(selected, 1)]
    selection_hash = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    write_json(folder / 'selection.json', {'selection_rule': 'First matching records in original dataset order; no outcome filtering',
                                         'dataset_metadata_sha256': metadata_hash, 'selection_sha256': selection_hash,
                                         'samples': selection})
    write_json(folder / 'effective_config.json', {'base': base, 'repositories': repositories, 'requested_limit': limit})
    context = {'run_id': run_id, 'created_at': now.isoformat(), 'dataset': base['dataset'], 'split': base['split'],
               'repositories': repositories, 'requested_limit': limit, 'available_selected_samples': len(selected),
               'selection_sha256': selection_hash, 'dataset_metadata_sha256': metadata_hash,
               'server_version': info['version']['number'], 'scope': 'Small deterministic subset; not full benchmark performance'}
    records = []
    write_checkpoints(folder, records, {**context, **aggregate_results(records, base['top_k'], len(selected))})
    for position, sample in enumerate(selected, 1):
        row = sample['report']
        config = {**base, 'repository': row['repo_name'], 'issue_id': row['issue_id']}
        case = folder / 'cases' / (f'{position:03d}_' + row['repo_name'].replace('/', '__') + '-' + str(row['issue_id']))
        case.mkdir(parents=True)
        record = {'position': position, 'dataset_row': sample['dataset_row'], 'repository': row['repo_name'],
                  'issue_id': row['issue_id'], 'before_fix_sha': row.get('before_fix_sha')}
        started = time.monotonic()
        print(f"[{position}/{len(selected)}] {row['repo_name']} #{row['issue_id']}", flush=True)
        try:
            output = run(project, config, report=row, output_directory=case, verbose=False)
            record.update({'status': 'ok', 'metrics': output['metrics'], 'missing_gold': output['ground_truth_missing_from_candidates'],
                           'indexed_files': output['indexed_files'], 'index': output['index'],
                           'index_fingerprint': output['index_fingerprint'], 'query_truncated': output['query_truncated'],
                           'result_file': (case / 'result.json').relative_to(folder).as_posix()})
            print(f"  AP@{base['top_k']}={output['metrics']['average_precision_at_k']:.4f} "
                  f"RR@{base['top_k']}={output['metrics']['reciprocal_rank_at_k']:.4f} "
                  f"Top-1={output['metrics']['top_1']} missing_gold={len(record['missing_gold'])}", flush=True)
        except Exception as error:
            record.update({'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
            write_json(case / 'error.json', {**record, 'traceback': traceback.format_exc()})
            write_json(case / 'effective_config.json', config)
            print('  Failed:', type(error).__name__, str(error), flush=True)
        record['elapsed_seconds'] = round(time.monotonic() - started, 3)
        records.append(record)
        summary = {**context, **aggregate_results(records, base['top_k'], len(selected))}
        write_checkpoints(folder, records, summary)
    print(json.dumps({field: summary[field] for field in ('selected_samples', 'successful_samples', 'failed_samples', 'metrics', 'coverage')}, ensure_ascii=False, indent=2), flush=True)
    print('Batch results:', folder, flush=True)
    return folder, summary
