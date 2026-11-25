"""Resumable, cache-only corpus encoding. No Elasticsearch or evaluation required."""

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from urllib.error import HTTPError, URLError

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))

from bug_localization.bm25 import collect_files, prepare_report
from bug_localization.embeddings import LocalEmbedder, digest
from bug_localization.languages import grammar_versions, language_for_path, parser_for_language
from bug_localization.parallel_chunking import create_pool, _chunk
from bug_localization.snapshot import get_snapshot, sha256_file
from bug_localization.vector import validate_vector_config


# The cache pipeline's direct and transitive local dependencies, including the
# lazy snapshot providers and spawned chunk workers. Unrelated experiment modules
# must not interrupt a multi-day cache run merely by being added to this package.
IMPLEMENTATION_FILES = (
    'scripts/retrieval/warm_embedding_cache.py',
    *('src/bug_localization/' + name + '.py' for name in (
        '__init__', 'aggregation', 'ast_chunking', 'batch', 'bench4bl', 'bm25',
        'chunk_cache', 'chunking', 'embeddings', 'languages', 'lca', 'model_files',
        'parallel_chunking', 'snapshot', 'vector')),
)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


@contextmanager
def lock(path, blocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def pin(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Frozen inputs changed; use a new run directory: ' + str(path))
    else:
        save(path, value)


def load_job(config_path):
    config = json.loads(config_path.read_text())
    base = json.loads((PROJECT / config['base_config']).read_text())
    validate_vector_config(config, base)
    for ext in base['extensions']:
        parser_for_language(language_for_path('source' + ext, config.get('language_overrides', {})))
    return config, base


def freeze(folder, config, base, limit=None):
    from datasets import load_from_disk
    root = PROJECT / 'datasets' / base['dataset']
    files = [root / 'dataset_dict.json', *sorted((root / base['split']).glob('*'))]
    for name in ('bench4bl_sources.json',):
        if (root / name).exists():
            files.append(root / name)
    identity = {'semantic': config, 'candidates': base, 'limit': limit,
                'dataset_files': {str(f.relative_to(root)): sha256_file(f) for f in files if f.is_file()},
                'grammars': grammar_versions(),
                'implementation': implementation_identity()}
    pin(folder / 'inputs.json', identity)
    rows = load_from_disk(str(root))[base['split']]
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    selection = [{'dataset_row': i, 'repository': r['repo_name'], 'issue_id': r['issue_id'],
                  'before_fix_sha': r['before_fix_sha'], 'report_sha256': digest(report_identity(r))}
                 for i, r in enumerate(rows)]
    pin(folder / 'selection.json', {'dataset': base['dataset'], 'split': base['split'],
                                   'selection_sha256': digest(selection), 'samples': selection})
    return rows


def report_identity(row):
    # Raw Arrow files are hashed above, including timestamp and other metadata.
    # The selection fingerprint uses only the actual retrieval inputs.
    return {field: row.get(field) for field in
            ('repo_name', 'issue_id', 'before_fix_sha', 'title', 'body', 'updated_files')}


def grouped_rows(rows):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((row['repo_name'], row['before_fix_sha']), []).append(index)
    return groups


def bounded_chunks(pool, documents, route, pending_limit):
    """Limit queued CPU work while GPU encoding consumes chunks."""
    source = iter(documents)
    pending = deque()
    for _ in range(pending_limit):
        document = next(source, None)
        if document is None:
            break
        pending.append(pool.submit(_chunk, (document, route)))
    while pending:
        yield pending.popleft().result()
        document = next(source, None)
        if document is not None:
            pending.append(pool.submit(_chunk, (document, route)))


class Progress:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS items (kind TEXT, key TEXT, status TEXT, detail TEXT, PRIMARY KEY(kind,key))')

    def should_run(self, kind, key, retry):
        row = self.db.execute('SELECT status FROM items WHERE kind=? AND key=?', (kind, str(key))).fetchone()
        return row is None or (retry and row[0] == 'failed')

    def put(self, kind, key, status, detail):
        self.db.execute('INSERT OR REPLACE INTO items VALUES (?,?,?,?)',
                        (kind, str(key), status, json.dumps({'updated_at': now(), **detail}, ensure_ascii=False)))
        self.db.commit()

    def counts(self):
        return {kind + '_' + status: count for kind, status, count in
                self.db.execute('SELECT kind,status,COUNT(*) FROM items GROUP BY kind,status')}

    def failures(self):
        return [{'kind': k, 'key': key, **json.loads(detail)} for k, key, detail in
                self.db.execute("SELECT kind,key,detail FROM items WHERE status='failed'")]


def source_for(base, repository, commit):
    # The downloader and GPU worker may reach the same revision concurrently.
    with lock(PROJECT / 'repos/.warmup-locks' / (digest([repository, commit]) + '.lock'), blocking=True):
        return fetch_source(base, repository, commit)


def fetch_source(base, repository, commit):
    options = {'source_provider': base['source_provider'], 'source_dataset': base['dataset']} if base.get('source_provider') else {}
    for attempt in range(3):
        try:
            return get_snapshot(PROJECT, repository, commit, **options)
        except (URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))


def fatal(error):
    # Configuration/resource failures must stop the worker, not mark the corpus bad.
    return isinstance(error, (MemoryError, sqlite3.Error)) or (
        isinstance(error, OSError) and error.errno in (12, 28)) or any(term in str(error).lower() for term in (
            'out of memory', 'embedding cache checksum', 'invalid cached embedding', 'model produced an invalid'))


def implementation_identity(project=None):
    project = PROJECT if project is None else Path(project)
    return {name: sha256_file(project / name) for name in IMPLEMENTATION_FILES}


def validate_implementation(frozen):
    current = implementation_identity()
    changes = sorted(name for name in frozen.keys() | current.keys()
                     if frozen.get(name) != current.get(name))
    if changes:
        raise ValueError('Cache implementation changed since queue start: ' + ', '.join(changes))


def prefetch(plan_path, folder):
    """Prepare both BeetleBox splits while earlier datasets are being encoded."""
    from datasets import load_from_disk
    plan = json.loads(plan_path.read_text())
    selected = {}
    dataset_files = {}
    for job in plan['jobs']:
        _, base = load_job(PROJECT / job['config'])
        if base['dataset'] != 'BeetleBox' or base['split'] in selected:
            continue
        selected[base['split']] = base
        root = PROJECT / 'datasets/BeetleBox'
        for f in sorted((root / base['split']).glob('*')):
            if f.is_file():
                dataset_files[str(f.relative_to(root))] = sha256_file(f)
    with lock(folder / 'prefetch.lock'):
        pin(folder / 'inputs.json', {'datasets': selected, 'files': dataset_files, 'implementation': implementation_identity()})
        groups = {}
        for split, base in selected.items():
            for key in grouped_rows(load_from_disk(str(PROJECT / 'datasets/BeetleBox'))[split]):
                groups.setdefault(key, base)
        progress = Progress(folder / 'progress.sqlite3')
        status = 'running'

        def checkpoint(active=None):
            save(folder / 'status.json', {'status': status, 'updated_at': now(), 'snapshots_total': len(groups),
                 **progress.counts(), 'active': active})

        def acquire(item):
            (repo, commit), base = item
            key = repo + '@' + str(commit)
            try:
                if shutil.disk_usage(PROJECT / 'repos').free < plan['min_free_gb'] * 1024 ** 3:
                    raise OSError(28, 'Source volume free-space reserve reached')
                source, manifest = source_for(base, repo, commit)
                return key, 'complete', {'source': str(source), 'files': len(manifest['files']), 'manifest_sha256': digest(manifest)}
            except Exception as error:
                if fatal(error):
                    raise
                return key, 'failed', {'error': str(error), 'traceback': traceback.format_exc()}

        try:
            checkpoint()
            pending = deque()
            items = iter((k, b) for k, b in groups.items() if progress.should_run('snapshot', k[0] + '@' + str(k[1]), False))
            with ThreadPoolExecutor(max_workers=2) as pool:
                for _ in range(2):
                    item = next(items, None)
                    if item is not None:
                        pending.append(pool.submit(acquire, item))
                while pending:
                    key, state, detail = pending.popleft().result()
                    progress.put('snapshot', key, state, detail)
                    print(state + ' ' + key, flush=True)
                    checkpoint(key)
                    item = next(items, None)
                    if item is not None:
                        pending.append(pool.submit(acquire, item))
            failures = progress.failures()
            save(folder / 'failures.json', failures)
            status = 'completed_with_failures' if failures else 'completed'
            checkpoint()
        except BaseException as error:
            status = 'failed'
            save(folder / 'failures.json', progress.failures())
            checkpoint(str(error))
            raise
        finally:
            progress.db.close()


def run_job(config_path, folder, limit=None, retry=False, min_free_gb=100):
    config, base = load_job(config_path)
    folder.mkdir(parents=True, exist_ok=True)
    with lock(folder / 'worker.lock'):
        rows = freeze(folder, config, base, limit)
        groups = grouped_rows(rows)
        progress = Progress(folder / 'progress.sqlite3')
        embedder = LocalEmbedder(PROJECT, config)
        cache_path = PROJECT / 'cache/embeddings' / embedder.fingerprint / 'vectors.sqlite3'
        pin(folder / 'model.json', {'fingerprint': embedder.fingerprint, 'identity': embedder.identity,
                                   'cache_device': cache_path.stat().st_dev, 'cache_inode': cache_path.stat().st_ino})
        embedder.database.execute('PRAGMA mmap_size=1073741824')
        embedder.database.execute('PRAGMA cache_size=-65536')
        pool = create_pool(PROJECT, config, embedder.fingerprint)
        started = time.monotonic()
        chunk_hits = chunk_misses = 0
        status = 'running'
        active = {}

        def checkpoint():
            save(folder / 'status.json', {'status': status, 'updated_at': now(), 'dataset': base['dataset'],
                 'split': base['split'], 'model_path': config['model_path'], 'model_fingerprint': embedder.fingerprint,
                 'reports_total': len(rows), 'snapshots_total': len(groups), **progress.counts(),
                 'process_cache_hits': embedder.hits, 'process_new_embeddings': embedder.misses,
                 'process_chunk_hits': chunk_hits, 'process_chunk_misses': chunk_misses,
                 'elapsed_process_seconds': round(time.monotonic() - started, 1), 'active': active})

        try:
            with lock(cache_path.parent / 'warmup.lock'):
                checkpoint()
                for position, ((repository, commit), indices) in enumerate(groups.items(), 1):
                    if shutil.disk_usage(PROJECT / 'cache').free < min_free_gb * 1024 ** 3:
                        raise OSError(28, 'Cache volume has less than the configured free-space reserve')
                    key = repository + '@' + str(commit)
                    active = {'snapshot_position': position, 'repository': repository, 'commit': commit, 'stage': 'queries'}
                    for index in indices:
                        if not progress.should_run('query', index, retry):
                            continue
                        row = rows[index]
                        try:
                            _, _, body = prepare_report(row, {**base, 'repository': repository, 'issue_id': row['issue_id']})
                            _, info = embedder.query(body)
                            progress.put('query', index, 'complete', {'repository': repository, 'issue_id': row['issue_id'], **info})
                        except Exception as error:
                            if fatal(error):
                                raise
                            progress.put('query', index, 'failed', {'repository': repository, 'error': str(error), 'traceback': traceback.format_exc()})
                    if not progress.should_run('snapshot', key, retry):
                        if position % 25 == 0:
                            checkpoint()
                        continue
                    print(f'[{position}/{len(groups)}] {repository}@{commit} ({len(indices)} reports)', flush=True)
                    try:
                        active['stage'] = 'source_verification'
                        checkpoint()
                        source, manifest = source_for(base, repository, commit)
                        documents, skipped = collect_files(source, manifest, base)
                        candidate_paths = {d['file_path'] for d in documents}
                        case_id = digest([repository, commit])
                        case = folder / 'snapshots' / case_id
                        save(case / 'candidates.json', {'repository': repository, 'commit': commit,
                             'files': [{'file_path': d['file_path'], 'sha256': d['sha256']} for d in documents],
                             'skipped': skipped, 'source_provenance': manifest.get('bench4bl_source', manifest.get('source_provenance'))})
                        missing = {}
                        import ast
                        for index in indices:
                            missing[str(index)] = sorted(set(ast.literal_eval(rows[index]['updated_files'])) - candidate_paths)
                        save(case / 'label_coverage.json', {'missing_gold_by_dataset_row': missing,
                             'policy': 'Preserve every label and report; never filter sources using labels'})
                        before_hits, before_misses = embedder.hits, embedder.misses
                        counts = Counter()
                        for route in config['chunking_routes']:
                            active.update(stage='chunking_and_encoding', route=route, files_total=len(documents), files_done=0)
                            pending_texts = []
                            for file_number, (chunks, _, hits, misses) in enumerate(bounded_chunks(
                                    pool, documents, route, 2 * config['chunk_workers']), 1):
                                chunk_hits += hits
                                chunk_misses += misses
                                # Batch across files: a changed file may have only one uncached chunk.
                                pending_texts.extend(c['embedding_text'] for c in chunks)
                                while len(pending_texts) >= 2048:
                                    embedder.encode(pending_texts[:2048])
                                    del pending_texts[:2048]
                                counts[route] += len(chunks)
                                active['files_done'] = file_number
                                if file_number % 50 == 0:
                                    checkpoint()
                            if pending_texts:
                                embedder.encode(pending_texts)
                        progress.put('snapshot', key, 'complete', {'reports': len(indices), 'candidate_files': len(documents),
                             'chunks': dict(counts), 'cache_hits': embedder.hits - before_hits,
                             'new_embeddings': embedder.misses - before_misses,
                             'source_manifest_sha256': digest(manifest), 'artifact': str(case.relative_to(folder))})
                    except Exception as error:
                        if fatal(error):
                            raise
                        progress.put('snapshot', key, 'failed', {'reports': len(indices), 'repository': repository,
                                     'commit': commit, 'error': str(error), 'traceback': traceback.format_exc()})
                        print('FAILED ' + key + ': ' + str(error), flush=True)
                    checkpoint()
                failures = progress.failures()
                save(folder / 'failures.json', failures)
                status = 'completed_with_failures' if failures else 'completed'
                active = {}
                checkpoint()
                return 2 if failures else 0
        except BaseException as error:
            status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            active['error'] = str(error)
            save(folder / 'failures.json', progress.failures())
            checkpoint()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            embedder.close()
            progress.db.close()


def smoke(config_path, folder):
    import numpy as np
    config, _ = load_job(config_path)
    model = LocalEmbedder(PROJECT, config)
    try:
        # Distinct inputs fill both devices at the frozen maximum batch/length.
        texts = []
        for index in range(config['embedding_batch_size']):
            seed = f'cache smoke unique {index}: ' + 'source code function return value ' * 200
            ids = model.tokenizer.encode(seed, add_special_tokens=False)[:config['max_tokens'] - 1]
            text = model.tokenizer.decode(ids)
            while len(model.tokenizer.encode(text, add_special_tokens=True)) > config['max_tokens']:
                text = text[:-1]
            texts.append(text)
        first = model.encode(texts)
        misses = model.misses
        second = model.encode(texts)
        if model.misses != misses or not np.array_equal(first, second):
            raise AssertionError('Cache replay failed')
        save(folder / 'smoke.json', {'status': 'completed', 'updated_at': now(), 'model_fingerprint': model.fingerprint,
             'identity': model.identity, 'shape': list(first.shape), 'cache_replay_exact': True,
             'max_input_tokens': max(len(model.tokenizer.encode(t)) for t in texts),
             'hits': model.hits, 'misses': model.misses})
    finally:
        model.close()


def queue(plan_path, folder, retry=False):
    plan = json.loads(plan_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    with lock(folder / 'queue.lock'):
        # Freeze every job before dispatch so late stages cannot silently use edited settings.
        frozen = {'plan': plan, 'configs': {}, 'implementation': implementation_identity(), 'dataset_files': {}}
        for job in plan['jobs']:
            config, base = load_job(PROJECT / job['config'])
            frozen['configs'][job['id']] = {'semantic': config, 'candidates': base}
            root = PROJECT / 'datasets' / base['dataset']
            for f in [root / 'dataset_dict.json', *sorted((root / base['split']).glob('*')),
                      *root.glob('*_sources.json')]:
                if f.is_file() and str(f.relative_to(PROJECT)) not in frozen['dataset_files']:
                    frozen['dataset_files'][str(f.relative_to(PROJECT))] = sha256_file(f)
        pin(folder / 'plan.json', frozen)
        outcomes = []
        try:
            for job in plan['jobs']:
                current = json.loads((PROJECT / job['config']).read_text())
                if frozen['configs'][job['id']] != {'semantic': current, 'candidates': json.loads((PROJECT / current['base_config']).read_text())}:
                    raise ValueError('Job configuration changed since queue start')
                validate_implementation(frozen['implementation'])
                if any(sha256_file(PROJECT / f) != h for f, h in frozen['dataset_files'].items()):
                    raise ValueError('Dataset changed since queue start')
                save(folder / 'status.json', {'status': 'running', 'updated_at': now(), 'active_job': job['id'], 'outcomes': outcomes})
                command = [sys.executable, '-u', str(Path(__file__)), 'worker', '--config', str(PROJECT / job['config']),
                           '--output', str(folder / job['id']), '--min-free-gb', str(plan['min_free_gb'])]
                if retry:
                    command.append('--retry-failures')
                print('START ' + job['id'], flush=True)
                with (folder / (job['id'] + '.log')).open('a', buffering=1) as log:
                    code = subprocess.call(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
                outcomes.append({'job': job['id'], 'exit_code': code, 'finished_at': now()})
                print('FINISH ' + job['id'] + ' exit=' + str(code), flush=True)
                if code not in (0, 2):
                    raise RuntimeError('Worker failed; queue stopped: ' + job['id'])
            status = 'completed_with_failures' if any(x['exit_code'] for x in outcomes) else 'completed'
            save(folder / 'status.json', {'status': status, 'updated_at': now(), 'outcomes': outcomes})
        except BaseException as error:
            save(folder / 'status.json', {'status': 'failed', 'updated_at': now(), 'outcomes': outcomes, 'error': str(error)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['worker', 'queue', 'smoke', 'prefetch'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retry-failures', action='store_true')
    parser.add_argument('--min-free-gb', type=int, default=100)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.mode == 'queue':
        queue(args.config, args.output, args.retry_failures)
    elif args.mode == 'prefetch':
        prefetch(args.config, args.output)
    elif args.mode == 'smoke':
        smoke(args.config, args.output)
    else:
        return run_job(args.config, args.output, args.limit, args.retry_failures, args.min_free_gb)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
