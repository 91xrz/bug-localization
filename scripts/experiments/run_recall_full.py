"""Resumable full-corpus L0/S0/H0 with frozen inputs and GPU-aware dispatch."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import collect_files, prepare_report, ranking_metrics, run as bm25_run
from bug_localization.embeddings import LocalEmbedder, digest
from bug_localization.snapshot import get_snapshot, sha256_file
from bug_localization.vector import retrieve, select_reports
from run_direct_hypothesis import Generator, PROMPT_DIRECT

METHODS = ('L0', 'S0', 'H0')
DEPENDENCIES = (
    '__init__', 'aggregation', 'ast_chunking', 'batch', 'bench4bl', 'bm25',
    'chunk_cache', 'chunking', 'embeddings', 'languages', 'lca', 'model_files',
    'parallel_chunking', 'report_hypothesis', 'snapshot', 'vector', 'hybrid',
)


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def pin(path, value):
    if path.exists() and read(path) != value:
        raise ValueError('Frozen input changed: ' + str(path))
    save(path, value)


def state(folder, stage, status, **extra):
    save(folder / 'status' / (stage + '.json'), {
        'stage': stage, 'status': status, 'pid': os.getpid(), 'updated_at': now(), **extra})


def prepare(folder, config_path):
    from datasets import load_from_disk
    config = read(config_path)
    semantic = read(ROOT / config['semantic_config'])
    base = read(ROOT / semantic['base_config'])
    if (base['dataset'], base['split'], base['query_fields']) != (config['dataset'], config['split'], ['body']):
        raise ValueError('Require frozen SWE body-only inputs')
    cache_run = ROOT / config['cache_run']
    cached_inputs = read(cache_run / 'inputs.json')
    if cached_inputs['semantic'] != semantic or cached_inputs['candidates'] != base:
        raise ValueError('Warm cache configuration differs')
    # The original service has reached its shard limit. Use the already running
    # ES 9.5.5 service, preserving every scoring and candidate parameter.
    base = {**base, 'endpoint': config['evaluation_endpoint'], 'cluster_name': config['evaluation_cluster']}
    from elasticsearch import Elasticsearch
    with Elasticsearch(base['endpoint'], request_timeout=10) as client:
        info = client.info()
        if info['cluster_name'] != base['cluster_name'] or info['version']['number'] != '9.5.5':
            raise ValueError('Require the configured Elasticsearch 9.5.5 service')
    cache_status = read(cache_run / 'status.json')
    if cache_status['status'] != 'completed' or cache_status['query_complete'] != config['expected_samples']:
        raise ValueError('Require completed full-corpus vector preparation')
    data_root = ROOT / 'datasets' / base['dataset']
    data = load_from_disk(str(data_root))[base['split']]
    if len(data) != config['expected_samples']:
        raise ValueError('Unexpected dataset size')
    samples = []
    for index, row in enumerate(data):
        samples.append({'position': index + 1, 'dataset_row': index, 'repository': row['repo_name'],
            'issue_id': row['issue_id'], 'before_fix_sha': row['before_fix_sha'],
            'query_sha256': hashlib.sha256(json.dumps({'body': row['body']}, sort_keys=True).encode()).hexdigest(),
            'labels_sha256': hashlib.sha256(str(row['updated_files']).encode()).hexdigest()})
    old = read(ROOT / config['development_selection'])['samples']
    for item in old:
        if any(item[k] != samples[item['dataset_row']][k] for k in item if k != 'position'):
            raise ValueError('Development cohort identity changed')
    selection = {'dataset_metadata_sha256': sha256_file(data_root / base['split'] / 'dataset_info.json'),
        'selection_sha256': digest(samples), 'selection_rule': 'Every dataset row, in original order; no filtering',
        'samples': samples, 'project_counts': dict(Counter(r['repository'] for r in samples)),
        'development_rows': [r['dataset_row'] for r in old],
        'cohort_note': 'Complement of development500 is not claimed to be an independent blind test.'}
    folder.mkdir(parents=True, exist_ok=True)
    pin(folder / 'selection.json', selection)
    files = [Path(__file__), ROOT / 'scripts/experiments/run_direct_hypothesis.py', config_path,
             ROOT / config['semantic_config'], ROOT / semantic['base_config'],
             ROOT / config['generation_config'], ROOT / config['development_selection'],
             cache_run / 'model.json', cache_run / 'inputs.json', folder / 'selection.json',
             *(ROOT / 'src/bug_localization' / (name + '.py') for name in DEPENDENCIES),
             data_root / 'dataset_dict.json', *sorted((data_root / base['split']).glob('*'))]
    protocol = {'config': config, 'semantic': semantic, 'base': base,
        'generation': read(ROOT / config['generation_config']),
        'embedding': read(cache_run / 'model.json'),
        'files_sha256': {str(p.resolve()): sha256_file(p) for p in files if p.is_file()},
        'embedding_policy': 'All S0/H0 ranks recomputed with warm two-GPU identity; do not mix old one-GPU ranks.',
        'query_policy': 'Original body only. Direct hypothesis prompt and generation unchanged; no enrichment or reranking.'}
    pin(folder / 'protocol.json', protocol)
    context(folder)
    state(folder, 'prepare', 'completed', samples=len(samples), projects=len(selection['project_counts']))
    print(json.dumps({'output': str(folder), 'samples': len(samples), 'projects': selection['project_counts']}, indent=2), flush=True)


def context(folder):
    protocol = read(folder / 'protocol.json')
    for path, expected in protocol['files_sha256'].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError('Frozen implementation/input changed: ' + path)
    semantic = {**protocol['semantic'], 'selection_manifest': str(folder / 'selection.json')}
    selected, selection = select_reports(ROOT, semantic, protocol['base'], max_samples=5000)
    return protocol, semantic, selected, selection['samples']


def gpu_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    return {int(parts[0]): int(parts[1]) for line in output.splitlines() if (parts := line.split(','))}


def wait_gpu(folder, stage, devices, minimum, config, **extra):
    while True:
        free = gpu_memory()
        if all(free.get(device, 0) >= minimum for device in devices):
            state(folder, stage, 'running', gpu_ids=devices, free_mib=free, **extra)
            return
        state(folder, stage, 'waiting_for_gpu', gpu_ids=devices, minimum_free_mib=minimum, free_mib=free, **extra)
        print(f'{now()} {stage}: waiting for {minimum} MiB on GPU {devices}; free={free}', flush=True)
        time.sleep(config['poll_seconds'])


def shard(body):
    return int(digest(body), 16) % 2


def generate(folder, worker):
    from transformers import AutoTokenizer
    protocol, _, selected, metas = context(folder)
    config = protocol['config']
    stage = 'generate' + str(worker)
    generator = Generator(protocol['generation'])
    pin(folder / 'generation_identity' / (str(worker) + '.json'), generator.identity)
    tokenizer = AutoTokenizer.from_pretrained(generator.path, local_files_only=True, trust_remote_code=False)
    assigned = [(m, s['report']) for m, s in zip(metas, selected) if shard(s['report']['body']) == worker]
    pending = []
    done = failures = reused = 0

    def persist(meta, value):
        save(folder / 'generation' / f"{meta['position']:04d}.json", {'sample': meta, **value})

    # First copy all compatible cached generations and detect oversized inputs,
    # even when another experiment currently occupies the assigned GPU.
    for meta, row in assigned:
        path = folder / 'generation' / f"{meta['position']:04d}.json"
        if path.exists():
            value = read(path)
            if (value['sample'] != meta or value['payload'] != {'original_report': row['body']}
                    or value['identity'] != generator.identity):
                raise ValueError('Generation checkpoint identity changed')
        else:
            payload = {'original_report': row['body']}
            key = digest({'identity': generator.identity, 'payload': payload})
            if (generator.folder / (key + '.json')).exists():
                value = {**generator.generate(row['body'], cache_only=True), 'cache_reused': True}
                reused += 1
            else:
                text = tokenizer.apply_chat_template([
                    {'role': 'system', 'content': PROMPT_DIRECT},
                    {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                count = len(tokenizer(text)['input_ids'])
                if count <= protocol['generation']['max_input_tokens']:
                    pending.append((meta, row))
                    continue
                value = {'identity': generator.identity, 'payload': payload, 'key': key,
                    'query': '', 'generation_error': 'Input exceeds limit', 'error_type': 'InputTooLong',
                    'input_tokens': count, 'cache_reused': False}
            persist(meta, value)
        done += 1
        failures += bool(value['generation_error'])
    state(folder, stage, 'running', selected=len(assigned), processed=done, failures=failures, reused=reused)
    if pending:
        wait_gpu(folder, stage, [config['gpu_ids'][worker]], config['generation_min_free_mib'], config,
                 selected=len(assigned), processed=done, failures=failures, reused=reused)
    for meta, row in pending:
        start = time.monotonic()
        value = generator.generate(row['body'])
        persist(meta, {**value, 'cache_reused': False, 'elapsed_seconds': time.monotonic() - start})
        done += 1
        failures += bool(value['generation_error'])
        state(folder, stage, 'running', selected=len(assigned), processed=done, failures=failures, reused=reused)
        print(f'{stage}: {done}/{len(assigned)}; row={meta["dataset_row"]}; failures={failures}', flush=True)
    state(folder, stage, 'completed', selected=len(assigned), processed=done, failures=failures, reused=reused)


def inventory_reference(protocol, meta):
    key = digest([meta['repository'], meta['before_fix_sha']])
    return read(ROOT / protocol['config']['cache_run'] / 'snapshots' / key / 'candidates.json')['files']


def record(folder, method, meta, result, candidates, gold, missing, **extra):
    paths = [r['file_path'] for r in result['ranking']]
    metrics = ranking_metrics(paths, gold, 10)
    output = {**meta, 'status': 'ok', 'metrics': metrics, 'missing_gold': missing,
        'candidate_fingerprint': digest(candidates), 'ground_truth': gold, **extra}
    path = folder / method / 'cases' / f"{meta['position']:04d}" / 'result.json'
    save(path, {**result, **output})
    return {**output, 'result_file': str(path.relative_to(folder)), 'result_sha256': sha256_file(path)}


def resumed(folder, method, meta):
    path = folder / method / 'cases' / f"{meta['position']:04d}" / 'record.json'
    if not path.exists():
        return None
    row = read(path)
    if any(row[k] != v for k, v in meta.items()):
        raise ValueError('Result checkpoint identity changed')
    if row['status'] == 'ok':
        result_path = folder / row['result_file']
        if sha256_file(result_path) != row['result_sha256']:
            raise ValueError('Result checksum changed')
        result = read(result_path)
        if ranking_metrics([r['file_path'] for r in result['ranking']], result['ground_truth'], 10) != row['metrics']:
            raise ValueError('Result metrics differ from rankings')
    return row


def checkpoint(folder, method, meta, row, records, total):
    save(folder / method / 'cases' / f"{meta['position']:04d}" / 'record.json', row)
    records.append(row)
    save(folder / method / 'samples.json', records)
    state(folder, method, 'running', selected=total, processed=len(records),
          failures=sum(r['status'] == 'failed' for r in records))
    print(f'{now()} {method}: {len(records)}/{total}; {meta["repository"]} #{meta["issue_id"]}; {row["status"]}', flush=True)


def run_bm25(folder):
    from elasticsearch import Elasticsearch
    protocol, _, selected, metas = context(folder)
    base = protocol['base']
    records = []
    with Elasticsearch(base['endpoint'], request_timeout=120) as client:
        if client.info()['cluster_name'] != base['cluster_name']:
            raise ValueError('Wrong Elasticsearch cluster')
        for meta, sample in zip(metas, selected):
            saved = resumed(folder, 'L0', meta)
            if saved is None:
                start = time.monotonic()
                case = folder / 'L0/cases' / f"{meta['position']:04d}"
                case.mkdir(parents=True, exist_ok=True)
                cfg = {**base, 'repository': meta['repository'], 'issue_id': meta['issue_id']}
                existing = set(client.indices.get(index='bl-bm25-*', allow_no_indices=True))
                try:
                    value = bm25_run(ROOT, cfg, report=sample['report'], output_directory=case, verbose=False)
                except ValueError as error:
                    if str(error) != 'The analyzer produced no query terms':
                        raise
                    saved = {**meta, 'status': 'failed', 'error_type': 'EmptyLexicalQuery', 'error': str(error)}
                else:
                    if value['candidate_files'] != inventory_reference(protocol, meta):
                        raise ValueError('BM25 and frozen semantic candidates differ')
                    _, gold, query = prepare_report(sample['report'], cfg)
                    if value['query_text'] != query or value['ground_truth'] != gold:
                        raise ValueError('BM25 query/labels differ')
                    saved = record(folder, 'L0', meta, {**value, 'ranking': value['results']},
                        value['candidate_files'], gold, value['ground_truth_missing_from_candidates'],
                        elapsed_seconds=time.monotonic() - start)
                    index = value['index']
                    if index not in existing:
                        expected = {'project': 'bug-localization', 'repository': meta['repository'],
                                    'commit': meta['before_fix_sha'], 'fingerprint': value['index_fingerprint']}
                        if client.indices.get_mapping(index=index)[index]['mappings'].get('_meta') != expected:
                            raise ValueError('BM25 temporary index ownership differs')
                        client.indices.delete(index=index)
            checkpoint(folder, 'L0', meta, saved, records, len(metas))
    state(folder, 'L0', 'completed', selected=len(metas), processed=len(records))


def run_semantic(folder, method):
    from elasticsearch import Elasticsearch
    from bug_localization.ast_chunking import RoutingChunkCache
    from bug_localization.parallel_chunking import create_pool, chunk_documents
    protocol, semantic, selected, metas = context(folder)
    base, config = protocol['base'], protocol['config']
    encoder = LocalEmbedder(ROOT, semantic)
    if encoder.identity != protocol['embedding']['identity']:
        raise ValueError('Embedding identity changed')
    pin(folder / method / 'embedding_identity.json', encoder.identity)
    cache = RoutingChunkCache(ROOT, encoder.tokenizer, semantic, encoder.fingerprint)
    pool = create_pool(ROOT, semantic, encoder.fingerprint)
    original_load = encoder.load_model

    def guarded_load():
        if method == 'S0' or not allow_query_miss:
            raise ValueError('Expected complete code/original query cache; refusing to change embedding inputs')
        if encoder.model is None:
            wait_gpu(folder, method, config['gpu_ids'], config['embedding_min_free_mib'], config,
                     selected=len(metas), processed=len(records))
        original_load()

    encoder.load_model = guarded_load
    records = []
    allow_query_miss = False
    try:
        with Elasticsearch(base['endpoint'], request_timeout=120) as client:
            if client.info()['cluster_name'] != base['cluster_name']:
                raise ValueError('Wrong Elasticsearch cluster')
            for meta, sample in zip(metas, selected):
                saved = resumed(folder, method, meta)
                if saved is not None:
                    checkpoint(folder, method, meta, saved, records, len(metas))
                    continue
                start = time.monotonic()
                row = sample['report']
                _, gold, query = prepare_report(row, {**base, 'repository': meta['repository'], 'issue_id': meta['issue_id']})
                generated = None
                if method == 'H0':
                    generated = read(folder / 'generation' / f"{meta['position']:04d}.json")
                    if generated['sample'] != meta or generated['payload'] != {'original_report': row['body']}:
                        raise ValueError('Hypothesis input changed')
                    query = generated['query']
                    if generated['generation_error'] or not query.strip():
                        saved = {**meta, 'status': 'failed', 'error_type': 'GenerationError',
                                 'error': generated['generation_error'] or 'Empty hypothesis'}
                        checkpoint(folder, method, meta, saved, records, len(metas))
                        continue
                source, manifest = get_snapshot(ROOT, meta['repository'], meta['before_fix_sha'])
                documents, skipped = collect_files(source, manifest, base)
                candidates = [{'file_path': d['file_path'], 'sha256': d['sha256']} for d in documents]
                if candidates != inventory_reference(protocol, meta):
                    raise ValueError('Frozen candidate hashes differ')
                chunks = [c for item in chunk_documents(pool, documents, 'blagent_ast') for c in item[0]]
                assets = folder / 'assets' / digest([meta['repository'], meta['before_fix_sha']])
                if method == 'S0':
                    pin(assets / 'candidates.json', {'files': candidates, 'skipped': skipped})
                    pin(assets / 'chunks.json', chunks)
                elif read(assets / 'candidates.json')['files'] != candidates or read(assets / 'chunks.json') != chunks:
                    raise ValueError('S0/H0 candidate or chunk inputs differ')
                allow_query_miss = False
                vectors = encoder.encode([c['embedding_text'] for c in chunks])
                allow_query_miss = True
                query_vector, query_info = encoder.query(query)
                allow_query_miss = False
                identity = {'repository': meta['repository'], 'commit': meta['before_fix_sha'], 'route': 'blagent_ast',
                    'model_fingerprint': encoder.fingerprint, **cache.index_identity(documents, 'blagent_ast'),
                    'full_experiment': digest(str(folder))}
                fingerprint = digest({'identity': identity, 'chunks': [c['chunk_id'] for c in chunks],
                    'vectors_sha256': hashlib.sha256(vectors.astype('<f4').tobytes()).hexdigest()})
                index = 'bl-vector-blagent_ast-' + meta['before_fix_sha'][:12] + '-' + fingerprint[:16]
                created = not client.indices.exists(index=index)
                try:
                    output = retrieve(client, documents, chunks, vectors, query_vector, identity, 'max', semantic['aggregation_temperature'])
                    if output['index'] != index:
                        raise ValueError('Unexpected semantic index')
                    missing = sorted(set(gold) - {d['file_path'] for d in documents})
                    saved = record(folder, method, meta, {**output, 'query': query, 'query_info': query_info},
                        candidates, gold, missing, elapsed_seconds=time.monotonic() - start,
                        assets=str(assets.relative_to(folder)),
                        generation_file=str((folder / 'generation' / f"{meta['position']:04d}.json").relative_to(folder)) if generated else None)
                finally:
                    if created and client.indices.exists(index=index):
                        ownership = client.indices.get_mapping(index=index)[index]['mappings'].get('_meta', {})
                        if ownership.get('fingerprint') != fingerprint or ownership.get('full_experiment') != digest(str(folder)):
                            raise ValueError('Temporary semantic index ownership differs')
                        client.indices.delete(index=index)
                checkpoint(folder, method, meta, saved, records, len(metas))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        encoder.close()
    state(folder, method, 'completed', selected=len(metas), processed=len(records),
          embedding_hits=encoder.hits, embedding_misses=encoder.misses)


def summarize(folder):
    selection = read(folder / 'selection.json')
    metas = selection['samples']
    old = set(selection['development_rows'])
    groups = {'all2294': {m['dataset_row'] for m in metas}, 'development500': old,
              'remaining1794': {m['dataset_row'] for m in metas} - old}
    records = {method: read(folder / method / 'samples.json') if (folder / method / 'samples.json').exists() else []
               for method in METHODS}
    cohorts, projects, macro = {}, {}, {}
    for cohort, ids in groups.items():
        cohorts[cohort] = {method: aggregate_results([r for r in rows if r['dataset_row'] in ids], 10, len(ids))
                           for method, rows in records.items()}
    for repo in sorted(selection['project_counts']):
        projects[repo] = {method: aggregate_results([r for r in rows if r['repository'] == repo], 10,
                            selection['project_counts'][repo]) for method, rows in records.items()}
    complete = all(len(rows) == len(metas) for rows in records.values())
    if complete:
        for method in METHODS:
            macro[method] = {metric: sum(p[method]['metrics'][metric] for p in projects.values()) / len(projects)
                             for metric in cohorts['all2294'][method]['metrics']}
    failed = any(r['status'] == 'failed' for rows in records.values() for r in rows)
    result = {'status': ('completed_with_failures' if failed else 'completed') if complete else 'running',
              'updated_at': now(), 'cohorts': cohorts, 'project_macro': macro}
    save(folder / 'summary.json', result)
    save(folder / 'project_summary.json', projects)
    with (folder / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['cohort', 'method', 'successful', 'failed', 'pending',
            'MAP@10', 'MRR@10', 'Top-1', 'Top-5', 'Top-10', 'MeanRecall@10'])
        writer.writeheader()
        for cohort, methods in cohorts.items():
            for method, stats in methods.items():
                writer.writerow({'cohort': cohort, 'method': method, 'successful': stats['successful_samples'],
                    'failed': stats['failed_samples'], 'pending': stats['pending_samples'], **(stats['metrics'] or {})})
    return result


def verify(folder):
    protocol, _, _, metas = context(folder)
    rows = {method: read(folder / method / 'samples.json') for method in METHODS}
    for method, records in rows.items():
        if len(records) != len(metas):
            raise ValueError('Incomplete method: ' + method)
        for meta, row in zip(metas, records):
            if resumed(folder, method, meta) != row:
                raise ValueError('Summary/checkpoint mismatch')
            if row['status'] == 'ok' and row['candidate_fingerprint'] != digest(inventory_reference(protocol, meta)):
                raise ValueError('Candidate fingerprint differs')
    for paired in zip(*(rows[m] for m in METHODS)):
        successful = [r for r in paired if r['status'] == 'ok']
        if len({digest(r['ground_truth']) for r in successful}) > 1:
            raise ValueError('Method labels differ')
    save(folder / 'verification.json', {'status': 'passed', 'samples': len(metas), 'methods': list(METHODS),
        'checks': ['frozen inputs and source implementation', 'all selected rows retained',
                   'result checksums and metrics independently recomputed', 'same candidate hashes and labels',
                   'S0/H0 share complete chunk inventories', 'every ES chunk score checked against NumPy']})


def launch(folder):
    protocol, _, _, _ = context(folder)
    config = protocol['config']
    processes, logs, exits = {}, {}, {}

    def start(stage):
        args = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--output', str(folder), '--stage', stage]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, config['gpu_ids'])),
               'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
        if stage.startswith('generate'):
            env['CUDA_VISIBLE_DEVICES'] = str(config['gpu_ids'][int(stage[-1])])
        log = (folder / 'logs' / (stage + '.log')).open('a')
        logs[stage] = log
        processes[stage] = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        state(folder, stage, 'starting', worker_pid=processes[stage].pid)

    (folder / 'logs').mkdir(exist_ok=True)
    for stage in ('L0', 'S0', 'generate0', 'generate1'):
        start(stage)
    state(folder, 'H0', 'queued', depends_on=['S0', 'generate0', 'generate1'])
    while True:
        for stage, process in processes.items():
            code = process.poll()
            if code is not None:
                exits[stage] = code
        if 'H0' not in processes:
            dependencies = ['S0', 'generate0', 'generate1']
            if all(exits.get(s) == 0 for s in dependencies):
                start('H0')
            elif any(exits.get(s, 0) != 0 for s in dependencies):
                state(folder, 'H0', 'blocked_by_worker_failure', dependency_exit_codes=exits)
        summary = summarize(folder)
        any_error = any(code != 0 for code in exits.values())
        state(folder, 'launcher', 'running_with_worker_error' if any_error else 'running',
              workers={s: p.pid for s, p in processes.items()}, exit_codes=exits)
        if len(exits) == len(processes):
            if any_error:
                raise RuntimeError('Worker failed; completed checkpoints retained: ' + str(exits))
            if summary['status'].startswith('completed'):
                verify(folder)
                state(folder, 'launcher', summary['status'], exit_codes=exits)
                break
        time.sleep(config['poll_seconds'])
    for log in logs.values():
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/recall_full_swe.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['prepare', 'launch', 'L0', 'S0', 'H0', 'generate0', 'generate1', 'summarize', 'verify'], default='launch')
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / (args.stage + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.stage == 'prepare':
                prepare(folder, args.config.resolve())
            elif args.stage == 'launch':
                launch(folder)
            elif args.stage == 'L0':
                run_bm25(folder)
            elif args.stage in ('S0', 'H0'):
                run_semantic(folder, args.stage)
            elif args.stage.startswith('generate'):
                generate(folder, int(args.stage[-1]))
            elif args.stage == 'verify':
                verify(folder)
            else:
                summarize(folder)
        except Exception as error:
            state(folder, args.stage, 'failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
