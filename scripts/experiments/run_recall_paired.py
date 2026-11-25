"""Snapshot-shared S0/H0 continuation; CPU retrieval, GPU-1-only query encoding."""
import argparse
from collections import OrderedDict
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import run_recall_full as old
from bug_localization.embeddings import LocalEmbedder, digest
from bug_localization.model_files import encoder_implementation_sha256

ROOT = old.ROOT


class CacheMiss(Exception):
    pass


class CachedVectors:
    """Validate/read existing vectors without initializing CUDA or a model."""
    def __init__(self, config, identity):
        from transformers import AutoTokenizer
        self.config = config
        self.identity = identity
        self.fingerprint = digest(identity)
        self.dimension = identity['dimension']
        model = ROOT / config['model_path']
        for name, checksum in identity['files_sha256'].items():
            if old.sha256_file(model / name) != checksum:
                raise ValueError('Model bytes changed: ' + name)
        if (identity['torch'] != version('torch') or identity['transformers'] != version('transformers')
                or identity['encoder_sha256'] != encoder_implementation_sha256(ROOT / 'src/bug_localization/embeddings.py')):
            raise ValueError('Embedding environment changed')
        folder = ROOT / 'cache/embeddings' / self.fingerprint
        if old.read(folder / 'model.json') != identity:
            raise ValueError('Cache identity differs')
        self.database = sqlite3.connect(f'file:{folder / "vectors.sqlite3"}?mode=ro', uri=True, timeout=60)
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False, padding_side='left')
        self.tokenizer.model_max_length = 10**9
        self.hits = 0

    def encode(self, texts):
        values = {}
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        for key in dict.fromkeys(keys):
            row = self.database.execute('SELECT data, checksum FROM vectors WHERE key=?', (key,)).fetchone()
            if row is None:
                raise CacheMiss(key)
            raw, checksum = row
            vector = np.frombuffer(raw, dtype='<f4')
            if (hashlib.sha256(raw).hexdigest() != checksum or vector.shape != (self.dimension,)
                    or not np.isfinite(vector).all() or abs(np.linalg.norm(vector) - 1) > 1e-3):
                raise ValueError('Invalid cached embedding')
            values[key] = vector
            self.hits += 1
        return np.stack([values[key] for key in keys])

    query = LocalEmbedder.query

    def close(self):
        self.database.close()


def single_identity(protocol):
    dual = protocol['embedding']['identity']
    return {**dual, 'gpu_count': 1, 'gpu': dual['gpu'][:1]}


def grouped(metas):
    groups = OrderedDict()
    for meta in metas:
        groups.setdefault((meta['repository'], meta['before_fix_sha']), []).append(meta)
    return list(groups.values())


def freeze(folder):
    protocol, semantic, selected, metas = old.context(folder)
    dual = CachedVectors(semantic, protocol['embedding']['identity'])
    single = CachedVectors({**semantic, 'gpu_count': 1}, single_identity(protocol))
    comparisons = []
    previous = old.read(ROOT / protocol['config']['development_selection'])['samples']
    # Round-robin projects; compare actual body-only, normalized original queries.
    projects = {}
    for m in previous:
        projects.setdefault(m['repository'], []).append(m)
    ordered = [items[i] for i in range(max(map(len, projects.values()))) for items in projects.values() if i < len(items)]
    for meta in ordered:
        row = selected[meta['dataset_row']]['report']
        _, _, query = old.prepare_report(row, {**protocol['base'], 'repository': meta['repository'], 'issue_id': meta['issue_id']})
        try:
            a, ai = dual.query(query)
            b, bi = single.query(query)
        except CacheMiss:
            continue
        if ai != bi or not np.array_equal(a, b):
            raise ValueError('Single-query compatibility check is not bitwise identical')
        comparisons.append({'dataset_row': meta['dataset_row'], 'repository': meta['repository'], 'max_absolute_error': 0.0})
        if len(comparisons) == 32:
            break
    dual.close()
    single.close()
    if len(comparisons) < 8:
        raise ValueError('Not enough single-query cache comparisons')
    value = {'original_protocol_sha256': old.sha256_file(folder / 'protocol.json'),
        'runner_sha256': old.sha256_file(Path(__file__)), 'physical_gpu_ids_allowed': [1],
        'code_embedding_identity': protocol['embedding']['identity'],
        'hypothesis_query_embedding_identity': single_identity(protocol),
        'compatibility': comparisons,
        'changes': ['Group all reports by repository and exact source commit',
                    'Load and validate source, chunks and cached code vectors once per snapshot',
                    'Use the same ES index for every pending S0/H0 query before cleanup',
                    'Keep completed records; write new records in original selection order',
                    'Prepare singleton H0 query vectors on physical GPU1 only; retain actual single-GPU identity'],
        'query_equivalence': 'Original query() encodes a batch of one; dual DataParallel scatters it to only its first GPU. Same model, BF16, kernel, tokens, singleton batch; 32 cached controls checked bitwise.',
        'failure_policy': protocol['config']['failure_policy']}
    old.pin(folder / 'paired_protocol.json', value)
    old.state(folder, 'paired_prepare', 'completed', compared_queries=len(comparisons), allowed_gpus=[1])
    print(json.dumps({'compatibility_checks': len(comparisons), 'projects': len({r['repository'] for r in comparisons}), 'max_error': 0.0}), flush=True)


def context(folder):
    amendment = old.read(folder / 'paired_protocol.json')
    if (amendment['runner_sha256'] != old.sha256_file(Path(__file__))
            or amendment['original_protocol_sha256'] != old.sha256_file(folder / 'protocol.json')):
        raise ValueError('Paired execution provenance changed')
    return old.context(folder)


def query_path(folder, meta):
    return folder / 'hypothesis_query_vectors' / f"{meta['position']:04d}.json"


def load_prepared(folder, meta):
    path = query_path(folder, meta)
    if not path.exists():
        return None
    result = old.read(path)
    if result['sample'] != meta:
        raise ValueError('Prepared query identity differs')
    generation = folder / 'generation' / f"{meta['position']:04d}.json"
    if old.sha256_file(generation) != result['generation_sha256']:
        raise ValueError('Generation changed after query encoding')
    if result['status'] == 'ok':
        expected_identity = old.read(folder / 'paired_protocol.json')['hypothesis_query_embedding_identity']
        if result['embedding_identity'] != expected_identity:
            raise ValueError('Prepared query encoder identity differs')
        vector = np.asarray(result['vector'], dtype='<f4')
        if (vector.shape != (1024,) or not np.isfinite(vector).all()
                or abs(np.linalg.norm(vector) - 1) > 1e-3
                or hashlib.sha256(vector.tobytes()).hexdigest() != result['vector_sha256']):
            raise ValueError('Prepared query vector is corrupt')
    return result


def prepare_queries(folder):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Only physical GPU1 may be visible for H0 query preparation')
    protocol, semantic, selected, metas = context(folder)
    cfg = {**semantic, 'gpu_count': 1}
    cache = CachedVectors(cfg, single_identity(protocol))
    encoder = None
    ready = {m['position'] for m in metas if load_prepared(folder, m) is not None}
    pending = {}
    while len(ready) < len(metas):
        for meta, sample in zip(metas, selected):
            pos = meta['position']
            if pos in ready or pos in pending:
                continue
            path = folder / 'generation' / f'{pos:04d}.json'
            if not path.exists():
                continue
            gen = old.read(path)
            if gen['sample'] != meta or gen['payload'] != {'original_report': sample['report']['body']}:
                raise ValueError('Generation input differs')
            entry = {'sample': meta, 'generation_sha256': old.sha256_file(path), 'query': gen['query']}
            if gen['generation_error'] or not gen['query'].strip():
                old.save(query_path(folder, meta), {**entry, 'status': 'failed', 'error': gen['generation_error'] or 'Empty hypothesis'})
                ready.add(pos)
                continue
            try:
                vector, info = cache.query(gen['query'])
            except CacheMiss:
                pending[pos] = entry
                continue
            save_query(folder, entry, vector, info, cache.identity, True)
            ready.add(pos)
        free = old.gpu_memory().get(1, 0)
        generation_done = old.read(folder / 'status/generate1.json')['status'] == 'completed'
        if pending and generation_done and (encoder is not None or free >= 2500):
            if encoder is None:
                encoder = LocalEmbedder(ROOT, cfg)
                if encoder.identity != cache.identity:
                    raise ValueError('GPU1 query encoder differs from the validated identity')
                # Freshly encode an original query and verify the physical GPU1
                # execution against the existing dual-GPU singleton cache.
                reference = old.read(folder / 'paired_protocol.json')['compatibility'][0]
                meta = metas[reference['dataset_row']]
                _, _, body = old.prepare_report(selected[meta['dataset_row']]['report'],
                    {**protocol['base'], 'repository': meta['repository'], 'issue_id': meta['issue_id']})
                expected, info = cache.query(body)
                original_db = encoder.database
                encoder.database = sqlite3.connect(':memory:')
                encoder.database.execute('CREATE TABLE vectors (key TEXT PRIMARY KEY, data BLOB NOT NULL, checksum TEXT NOT NULL)')
                fresh, actual_info = encoder.query(body)
                encoder.database.close()
                encoder.database = original_db
                if actual_info != info or not np.array_equal(expected, fresh):
                    raise ValueError('Fresh singleton GPU1 query differs from frozen cached control')
                old.save(folder / 'gpu1_query_verification.json', {'status': 'passed', 'dataset_row': meta['dataset_row'], 'max_absolute_error': 0.0})
            for pos in list(pending):
                entry = pending.pop(pos)
                vector, info = encoder.query(entry['query'])
                save_query(folder, entry, vector, info, encoder.identity, False)
                ready.add(pos)
                old.state(folder, 'queries', 'running', selected=len(metas), processed=len(ready), pending_generation=len(metas)-len(ready)-len(pending))
        old.state(folder, 'queries', 'completed' if len(ready) == len(metas) else 'waiting_for_generation_or_gpu',
                  selected=len(metas), processed=len(ready), uncached_queries=len(pending), gpu1_free_mib=free)
        if len(ready) < len(metas):
            time.sleep(15)
    cache.close()
    if encoder:
        encoder.close()


def save_query(folder, entry, vector, info, identity, reused):
    old.save(query_path(folder, entry['sample']), {**entry, 'status': 'ok', 'query_info': info,
        'vector': vector.tolist(), 'vector_sha256': hashlib.sha256(vector.astype('<f4').tobytes()).hexdigest(),
        'embedding_identity': identity, 'cache_reused': reused})


def append_record(folder, method, meta, record, records, total):
    old.save(folder / method / 'cases' / f"{meta['position']:04d}" / 'record.json', record)
    records[method][meta['position']] = record
    rows = sorted(records[method].values(), key=lambda r: r['position'])
    old.save(folder / method / 'samples.json', rows)
    old.state(folder, method, 'completed' if len(rows) == total else 'running', selected=total,
              processed=len(rows), failures=sum(r['status'] == 'failed' for r in rows))


def execute_snapshot(folder, protocol, semantic, selected, group, prepared, records, cache, pool, client, total):
    from bug_localization.ast_chunking import RoutingChunkCache
    from bug_localization.parallel_chunking import chunk_documents
    start = time.monotonic()
    meta = group[0]
    source, manifest = old.get_snapshot(ROOT, meta['repository'], meta['before_fix_sha'])
    documents, skipped = old.collect_files(source, manifest, protocol['base'])
    candidates = [{'file_path': d['file_path'], 'sha256': d['sha256']} for d in documents]
    for item in group:
        if candidates != old.inventory_reference(protocol, item):
            raise ValueError('Snapshot candidate hashes differ')
    source_seconds = time.monotonic() - start
    chunks = [chunk for item in chunk_documents(pool, documents, 'blagent_ast') for chunk in item[0]]
    assets = folder / 'assets' / digest([meta['repository'], meta['before_fix_sha']])
    old.pin(assets / 'candidates.json', {'files': candidates, 'skipped': skipped})
    old.pin(assets / 'chunks.json', chunks)
    chunk_seconds = time.monotonic() - start - source_seconds
    vectors = cache.encode([c['embedding_text'] for c in chunks])
    vector_seconds = time.monotonic() - start - source_seconds - chunk_seconds
    chunk_cache = RoutingChunkCache(ROOT, cache.tokenizer, semantic, cache.fingerprint)
    identity = {'repository': meta['repository'], 'commit': meta['before_fix_sha'], 'route': 'blagent_ast',
        'model_fingerprint': cache.fingerprint, **chunk_cache.index_identity(documents, 'blagent_ast'),
        'full_experiment': digest(str(folder))}
    fingerprint = digest({'identity': identity, 'chunks': [c['chunk_id'] for c in chunks],
                          'vectors_sha256': hashlib.sha256(vectors.astype('<f4').tobytes()).hexdigest()})
    index = 'bl-vector-blagent_ast-' + meta['before_fix_sha'][:12] + '-' + fingerprint[:16]
    calls = []
    try:
        for item in group:
            pos = item['position']
            row = selected[item['dataset_row']]['report']
            _, gold, original = old.prepare_report(row, {**protocol['base'], 'repository': item['repository'], 'issue_id': item['issue_id']})
            missing = sorted(set(gold) - {d['file_path'] for d in documents})
            for method in ('S0', 'H0'):
                if pos in records[method]:
                    continue
                query_start = time.monotonic()
                if method == 'H0' and prepared[pos]['status'] == 'failed':
                    record = {**item, 'status': 'failed', 'error_type': 'GenerationError', 'error': prepared[pos]['error']}
                else:
                    if method == 'S0':
                        query = original
                        vector, info = cache.query(query)
                        query_identity = cache.identity
                    else:
                        query = prepared[pos]['query']
                        vector = np.asarray(prepared[pos]['vector'], dtype='<f4')
                        info = prepared[pos]['query_info']
                        query_identity = prepared[pos]['embedding_identity']
                    result = old.retrieve(client, documents, chunks, vectors, vector, identity, 'max', semantic['aggregation_temperature'])
                    if result['index'] != index or (calls and not result['index_reused']):
                        raise ValueError('Queries did not share the same snapshot index')
                    calls.append({'method': method, 'position': pos, 'index_reused': result['index_reused'],
                                  'seconds': time.monotonic() - query_start})
                    record = old.record(folder, method, item, {**result, 'query': query, 'query_info': info}, candidates,
                        gold, missing, assets=str(assets.relative_to(folder)), elapsed_seconds=time.monotonic()-query_start,
                        query_embedding_identity=query_identity, shared_snapshot_execution=True,
                        generation_file=f'generation/{pos:04d}.json' if method == 'H0' else None)
                append_record(folder, method, item, record, records, total)
                print(f'{old.now()} paired {method} row={item["dataset_row"]} S0={len(records["S0"])}/{total} H0={len(records["H0"])}/{total}', flush=True)
    finally:
        if client.indices.exists(index=index):
            ownership = client.indices.get_mapping(index=index)[index]['mappings'].get('_meta', {})
            if ownership.get('fingerprint') != fingerprint or ownership.get('full_experiment') != digest(str(folder)):
                raise ValueError('Refusing cleanup of a foreign index')
            client.indices.delete(index=index)
    timing = {'repository': meta['repository'], 'commit': meta['before_fix_sha'], 'reports': len(group),
        'source_seconds': source_seconds, 'chunk_seconds': chunk_seconds, 'vector_read_seconds': vector_seconds,
        'total_seconds': time.monotonic()-start, 'queries': calls, 'updated_at': old.now()}
    old.save(folder / 'snapshot_timings' / (digest([meta['repository'], meta['before_fix_sha']]) + '.json'), timing)
    return timing


def paired(folder):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Retrieval must run with CUDA hidden')
    from elasticsearch import Elasticsearch
    from bug_localization.parallel_chunking import create_pool
    protocol, semantic, selected, metas = context(folder)
    cache = CachedVectors(semantic, protocol['embedding']['identity'])
    pool = create_pool(ROOT, semantic, cache.fingerprint)
    records = {m: {} for m in ('S0', 'H0')}
    for method in records:
        for meta in metas:
            saved = old.resumed(folder, method, meta)
            if saved is not None:
                records[method][meta['position']] = saved
        old.save(folder / method / 'samples.json', sorted(records[method].values(), key=lambda r:r['position']))
    groups = grouped(metas)
    # Prefer wholly unprocessed snapshots to amortize both queries immediately.
    groups.sort(key=lambda group: sum(m['position'] in records['S0'] for m in group))
    try:
        with Elasticsearch(protocol['base']['endpoint'], request_timeout=120) as client:
            if client.info()['cluster_name'] != protocol['base']['cluster_name']:
                raise ValueError('Wrong ES cluster')
            while groups:
                progress = False
                for group in list(groups):
                    if all(m['position'] in records['S0'] and m['position'] in records['H0'] for m in group):
                        groups.remove(group)
                        continue
                    prepared = {m['position']: load_prepared(folder, m) for m in group}
                    if any(value is None for value in prepared.values()):
                        continue
                    timing = execute_snapshot(folder, protocol, semantic, selected, group, prepared, records, cache, pool, client, len(metas))
                    groups.remove(group)
                    progress = True
                    old.state(folder, 'paired', 'running', snapshots_remaining=len(groups),
                              S0=len(records['S0']), H0=len(records['H0']), last_snapshot=timing)
                if groups and not progress:
                    old.state(folder, 'paired', 'waiting_for_query_vectors', snapshots_remaining=len(groups),
                              S0=len(records['S0']), H0=len(records['H0']))
                    time.sleep(15)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        cache.close()
    old.state(folder, 'paired', 'completed', S0=len(records['S0']), H0=len(records['H0']))


def launch(folder):
    protocol, _, _, metas = context(folder)
    processes = {}
    logs = []
    def start(stage, command, devices):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': devices, 'TOKENIZERS_PARALLELISM': 'false',
               'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
        log = (folder / 'logs' / (stage + '.log')).open('a')
        logs.append(log)
        processes[stage] = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
        old.state(folder, stage, 'starting', worker_pid=processes[stage].pid)
    original = str(ROOT / 'scripts/experiments/run_recall_full.py')
    wrapper = str(ROOT / 'scripts/experiments/resume_recall_full_resources.py')
    gen0 = old.read(folder / 'status/generate0.json')
    if gen0['status'] != 'completed':
        raise ValueError('GPU0 generation must already be completed; GPU0 is reserved for other work')
    if old.read(folder / 'status/L0.json')['status'] != 'completed':
        start('L0', [sys.executable, '-B', '-u', original, '--output', str(folder), '--stage', 'L0'], '')
    if old.read(folder / 'status/generate1.json')['status'] != 'completed':
        start('generate1', [sys.executable, '-B', '-u', wrapper, '--output', str(folder), '--worker', '--minimum-free-mib', '18500'], '1')
    for stage, devices in (('queries','1'), ('paired','')):
        start(stage, [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--output', str(folder), '--stage', stage], devices)
    while True:
        exits = {s:p.poll() for s,p in processes.items() if p.poll() is not None}
        errors = {s:code for s,code in exits.items() if code != 0}
        summary = old.summarize(folder)
        old.state(folder, 'launcher', 'running_with_worker_error' if errors else 'running',
                  workers={s:p.pid for s,p in processes.items()}, exit_codes=exits,
                  execution='snapshot-paired', allowed_gpu_ids=[1])
        if errors:
            # Do not let an operational fault silently become sample failures.
            raise RuntimeError('Worker failed; checkpoints retained: ' + str(errors))
        if len(exits) == len(processes):
            if not summary['status'].startswith('completed'):
                raise ValueError('Workers exited before all reports completed')
            old.verify(folder)
            old.state(folder, 'launcher', summary['status'], allowed_gpu_ids=[1])
            break
        time.sleep(15)
    for log in logs:
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare','queries','paired','launch'), default='launch')
    args = parser.parse_args()
    folder = args.output.resolve()
    lock_name = {'launch':'launch', 'paired':'S0'}.get(args.stage, args.stage)
    with (folder / (lock_name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            {'prepare':freeze, 'queries':prepare_queries, 'paired':paired, 'launch':launch}[args.stage](folder)
        except Exception as error:
            old.state(folder, args.stage, 'failed', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    main()
