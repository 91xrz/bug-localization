"""Pinned, resumable small-batch driver for the complete local SWE corpus."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.embeddings import digest
from bug_localization.snapshot import sha256_file

ROUTES = ('blagent_ast', 'dynamic', 'ast_control')
CONFIG = ROOT / 'configs/experiments/swe_full_semantic.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def build_manifests(rows, metadata_hash, batch_size=10):
    groups, seen = defaultdict(list), set()
    for i, row in enumerate(rows):
        key = (row['repo_name'], row['issue_id'])
        if key in seen:
            raise ValueError('Duplicate sample: ' + repr(key))
        seen.add(key)
        groups[row['repo_name']].append((i, row))
    manifests = []
    names = sorted(groups)
    while any(groups.values()):
        for name in names:
            group = groups[name][:batch_size]
            del groups[name][:batch_size]
            if not group:
                continue
            entries = []
            for position, (i, row) in enumerate(group, 1):
                entries.append({'position': position, 'dataset_row': i, 'repository': row['repo_name'],
                    'issue_id': row['issue_id'], 'before_fix_sha': row.get('before_fix_sha'),
                    'query_sha256': hashlib.sha256(json.dumps({'body': row.get('body')}, sort_keys=True).encode()).hexdigest(),
                    'labels_sha256': hashlib.sha256(str(row.get('updated_files')).encode()).hexdigest()})
            manifests.append({'selection_rule': 'Full dataset, round-robin by repository, original order within repository',
                              'dataset_metadata_sha256': metadata_hash, 'selection_sha256': digest(entries), 'samples': entries})
    return manifests


def initialize(directory, batch_size):
    from datasets import load_from_disk
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / 'plan.json'
    impl_names = ['chunking.py', 'chunk_cache.py', 'parallel_chunking.py', 'ast_chunking.py', 'token_ast.py',
                  'dependency_context.py', 'ast_local_experiment.py', 'languages.py', 'snapshot.py',
                  'embeddings.py', 'aggregation.py', 'vector.py', 'batch.py', 'bm25.py']
    hashes = {name: sha256_file(ROOT / 'src/bug_localization' / name) for name in impl_names}
    hashes['driver'] = sha256_file(Path(__file__))
    hashes['entrypoint'] = sha256_file(ROOT / 'scripts/retrieval/run_vector_batch.py')
    for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'):
        hashes['model/' + name] = sha256_file(ROOT / 'models/Qwen3-Embedding-0.6B' / name)
    hashes['config'] = sha256_file(CONFIG)
    hashes['base_config'] = sha256_file(ROOT / 'configs/retrieval/bm25.json')
    metadata = sha256_file(ROOT / 'datasets/swe_bench_formatted/test/dataset_info.json')
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan['hashes'] != hashes or plan['dataset_metadata_sha256'] != metadata or plan['batch_size'] != batch_size:
            raise ValueError('Frozen code, config, dataset or batch size differs from existing plan')
        return plan
    rows = load_from_disk(str(ROOT / 'datasets/swe_bench_formatted'))['test']
    manifests = build_manifests(rows, metadata, batch_size)
    selection_dir = directory / 'selections'
    selection_dir.mkdir(exist_ok=True)
    batches = []
    for i, manifest in enumerate(manifests, 1):
        path = selection_dir / f'{i:04d}.json'
        write_json(path, manifest)
        batches.append({'id': i, 'selection': str(path), 'count': len(manifest['samples']),
                        'repository': manifest['samples'][0]['repository'], 'hash': manifest['selection_sha256']})
    plan = {'created_at': now(), 'dataset': 'swe_bench_formatted/test', 'total_samples': len(rows),
            'batch_size': batch_size, 'routes': ROUTES, 'hashes': hashes, 'dataset_metadata_sha256': metadata,
            'ordering': 'Round-robin by repository; original sample order within each repository', 'batches': batches}
    write_json(plan_path, plan)
    return plan


def result_for(log):
    if not log.exists():
        return None
    for line in reversed(log.read_text(encoding='utf-8', errors='replace').splitlines()):
        if line.startswith('Semantic results: '):
            return Path(line.split(': ', 1)[1])
    return None


def list_vector_indices():
    from elasticsearch import Elasticsearch
    with Elasticsearch('http://127.0.0.1:9200', request_timeout=120) as client:
        if dict(client.info())['cluster_name'] != 'bug-localization-local':
            raise ValueError('Unexpected Elasticsearch cluster')
        return sorted(client.indices.get_alias(index='bl-vector-*', allow_no_indices=True).keys())


def cleanup_new_indices(folder, preexisting):
    # Delete only verified derived indices created by this batch; never touch pre-existing indices.
    from elasticsearch import Elasticsearch
    targets = {}
    for route in ROUTES:
        for record in json.loads((folder / route / 'samples.json').read_text()):
            if record['status'] != 'ok':
                continue
            result = json.loads((folder / record['result_file']).read_text())
            if not result['index_reused'] and result['index'] not in preexisting:
                targets[result['index']] = result['index_fingerprint']
    with Elasticsearch('http://127.0.0.1:9200', request_timeout=120) as client:
        for index, fingerprint in targets.items():
            if not client.indices.exists(index=index):
                continue
            metadata = client.indices.get_mapping(index=index)[index]['mappings'].get('_meta', {})
            if metadata.get('project') != 'bug-localization' or metadata.get('fingerprint') != fingerprint:
                raise ValueError('Refusing to delete index with unexpected ownership: ' + index)
            client.indices.delete(index=index)
    return sorted(targets)


def summarize(directory, plan, state, status):
    records = {r: [] for r in ROUTES}
    by_repo = defaultdict(lambda: {r: [] for r in ROUTES})
    for batch in plan['batches']:
        info = state['completed'].get(str(batch['id']))
        if not info:
            continue
        folder = Path(info['result'])
        selection = json.loads((folder / 'selection.json').read_text())
        if selection['selection_sha256'] != batch['hash']:
            raise ValueError('Completed batch selection fingerprint mismatch')
        for route in ROUTES:
            entries = json.loads((folder / route / 'samples.json').read_text())
            if len(entries) != batch['count']:
                raise ValueError('Incomplete finished batch')
            records[route].extend(entries)
            for entry in entries:
                by_repo[entry['repository']][route].append(entry)
    attempted = len(records[ROUTES[0]])
    output = {'updated_at': now(), 'status': status, 'total_samples': plan['total_samples'],
              'completed_samples': attempted, 'pending_samples': plan['total_samples'] - attempted,
              'completed_batches': len(state['completed']), 'total_batches': len(plan['batches']),
              'note': 'Partial metrics describe completed samples only; full metrics require all samples. Failures count as zero.',
              'routes': {}, 'projects': {}}
    if attempted:
        for route in ROUTES:
            output['routes'][route] = aggregate_results(records[route], 10)
        for repository, project_routes in by_repo.items():
            output['projects'][repository] = {r: aggregate_results(project_routes[r], 10) for r in ROUTES}
    write_json(directory / 'summary.json', output)
    lines = [f"# SWE 全量语义检索\n\n状态：{status}；完成 {attempted}/{plan['total_samples']} 条，批次 {len(state['completed'])}/{len(plan['batches'])}。\n",
             '未完成时，下表仅针对已完成报告；失败按零计入，不是全量最终指标。\n',
             '| 路线 | 完成 | 失败 | MAP@10 | MRR@10 | Top-1 | Top-5 | Top-10 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r, values in output['routes'].items():
        m = values['metrics']
        lines.append(f"| {r} | {values['attempted_samples']} | {values['failed_samples']} | {m['MAP@10']:.4f} | {m['MRR@10']:.4f} | {m['Top-1']:.4f} | {m['Top-5']:.4f} | {m['Top-10']:.4f} |")
    lines.append('\n按项目指标、缺失标签、失败原因和成功样本指标见 summary.json；各批次的完整排名和块数据见 state.json 所指目录。\n')
    (directory / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=10, choices=range(1, 51))
    parser.add_argument('--min-free-gb', type=float, default=20)
    args = parser.parse_args()
    folder = args.output.resolve()
    plan = initialize(folder, args.batch_size)
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'completed': {}, 'created_at': now()}
    (folder / 'logs').mkdir(exist_ok=True)
    consecutive_errors = 0
    summarize(folder, plan, state, 'running')
    for batch in plan['batches']:
        key = str(batch['id'])
        if key in state['completed']:
            continue
        if shutil.disk_usage(ROOT).free < args.min_free_gb * 1024**3:
            state['stop_reason'] = f'Free disk below {args.min_free_gb} GiB'
            write_json(state_path, state)
            summarize(folder, plan, state, 'stopped_low_disk')
            raise SystemExit(state['stop_reason'])
        logfile = folder / 'logs' / f"{batch['id']:04d}.log"
        # A previously completed run can be adopted after interruption between completion and checkpoint.
        result = result_for(logfile)
        inventory_path = folder / 'logs' / f"{batch['id']:04d}.preexisting.json"
        if result is None:
            write_json(inventory_path, list_vector_indices())
            print(f"[{batch['id']}/{len(plan['batches'])}] {batch['repository']} {batch['count']} samples", flush=True)
            command = [sys.executable, '-B', '-u', str(ROOT / 'scripts/retrieval/run_vector_batch.py'),
                       '--config', str(CONFIG), '--selection', batch['selection']]
            with logfile.open('a', encoding='utf-8') as log:
                log.write(f'\nSTART {now()}\n')
                log.flush()
                result_code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            result = result_for(logfile)
            if result is None:
                consecutive_errors += 1
                print(f'Batch {key} failed before completion (exit {result_code}); will be retried on resume.', flush=True)
                state['stop_reason'] = f'Batch {key} did not complete, exit {result_code}; see {logfile}'
                write_json(state_path, state)
                summarize(folder, plan, state, 'stopped_batch_error')
                raise SystemExit(1)
        summary = json.loads((result / 'summary.json').read_text())
        if summary['selection_sha256'] != batch['hash'] or set(summary['routes']) != set(ROUTES):
            raise ValueError('Result selection/routes do not match the plan')
        if any(x['pending_samples'] or x['attempted_samples'] != batch['count'] for x in summary['routes'].values()):
            raise ValueError('Cannot mark incomplete batch as completed')
        if not inventory_path.exists():
            raise ValueError('Cannot identify pre-existing indexes safely')
        cleaned = cleanup_new_indices(result, set(json.loads(inventory_path.read_text())))
        state['completed'][key] = {'result': str(result), 'completed_at': now(), 'removed_temporary_indexes': cleaned,
                                    'failures': {r: summary['routes'][r]['failed_samples'] for r in ROUTES}}
        state.pop('stop_reason', None)
        write_json(state_path, state)
        progress = summarize(folder, plan, state, 'running')
        print(f"Completed {progress['completed_samples']}/{plan['total_samples']}; failures: {state['completed'][key]['failures']}", flush=True)
        if all(summary['routes'][r]['successful_samples'] == 0 for r in ROUTES):
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        if consecutive_errors >= 3:
            state['stop_reason'] = 'Three consecutive batches failed on every route'
            write_json(state_path, state)
            summarize(folder, plan, state, 'stopped_repeated_failures')
            raise SystemExit(state['stop_reason'])
    summarize(folder, plan, state, 'completed')
    print('Complete:', folder / 'SUMMARY.md', flush=True)


if __name__ == '__main__':
    main()
