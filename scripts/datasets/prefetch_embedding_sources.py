"""Prepare pinned BeetleBox sources via incremental Git or validated curl archives."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cache_warmup', PROJECT / 'scripts/retrieval/warm_embedding_cache.py')
warmup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warmup)
MAX_ARCHIVE_BYTES = 100_000_000
MAX_SOURCE_BYTES = 500_000_000


def validate_archive(path, repository, commit):
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('Snapshot exceeds the 100 MB download limit')
    expected = repository.split('/')[1] + '-' + commit
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not members or sum(item.file_size for item in members) > MAX_SOURCE_BYTES:
            raise ValueError('Empty snapshot or uncompressed snapshot exceeds 500 MB')
        for item in members:
            parts = PurePosixPath(item.filename).parts
            if (not parts or parts[0] != expected or '\\' in item.filename
                    or any(p in ('.', '..') or ':' in p for p in parts)):
                raise ValueError('Archive path/root does not match the requested revision')
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError('Archive CRC mismatch: ' + corrupt)


def download_archive(repository, commit, destination):
    temporary = destination.with_name('snapshot.curl.part')
    url = f'https://codeload.github.com/{repository}/zip/{commit}'
    attempts = []
    for attempt in range(3):
        command = ['curl', '--http2', '--fail', '--location', '--silent', '--show-error',
                   '--connect-timeout', '20', '--max-time', '1800', '--speed-limit', '1024',
                   '--speed-time', '45', '--max-filesize', str(MAX_ARCHIVE_BYTES),
                   '--output', str(temporary), '--write-out', '%{http_code}', url]
        result = subprocess.run(command, capture_output=True, text=True, timeout=1830)
        http_status = result.stdout.strip()
        detail = {'attempt': attempt + 1, 'curl_exit': result.returncode, 'http_status': http_status,
                  'error': result.stderr.strip()[-2000:]}
        attempts.append(detail)
        if result.returncode == 0:
            try:
                validate_archive(temporary, repository, commit)
            except (zipfile.BadZipFile, EOFError) as error:
                detail['error'] = str(error)
            else:
                temporary.replace(destination)
                return attempts
        elif http_status in ('400', '401', '403', '404', '410', '422') or result.returncode == 63:
            break
        if attempt < 2:
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError('Archive download failed: ' + json.dumps(attempts))


def git_archive(repository, commit, destination):
    """Reuse Git objects between revisions; archive only the requested commit."""
    mirror = PROJECT / 'repos/.embedding-source-git' / (repository + '.git')
    remote = f'https://github.com/{repository}.git'
    options = ['git', '--no-replace-objects', '-c', 'core.hooksPath=/dev/null',
               '-c', 'core.fsmonitor=false', '-c', 'core.attributesFile=/dev/null',
               '-c', 'credential.helper=', '-c', 'http.lowSpeedLimit=1024',
               '-c', 'http.lowSpeedTime=45', '-C', str(mirror)]
    environment = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_NO_REPLACE_OBJECTS': '1'}

    def git(*args, check=True):
        return subprocess.run([*options, *args], env=environment, capture_output=True,
                              text=True, check=check, timeout=1800)

    with warmup.lock(mirror.with_suffix('.lock'), blocking=True):
        if not mirror.exists():
            subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', 'init', '--bare', str(mirror)],
                           env=environment, capture_output=True, check=True, timeout=30)
        local = git('rev-parse', '--verify', commit + '^{commit}', check=False)
        attempts = 0
        if local.returncode != 0:
            for attempt in range(3):
                attempts = attempt + 1
                try:
                    git('fetch', '--depth=1', '--no-tags', '--no-auto-gc', remote, commit + ':refs/cache/' + commit)
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    if attempt == 2:
                        raise RuntimeError('Exact-commit Git fetch failed: ' + str(error.stderr)[-2000:]) from error
                    time.sleep(2 ** (attempt + 1))
        if git('rev-parse', '--verify', commit + '^{commit}').stdout.strip() != commit:
            raise ValueError('Fetched Git commit does not match requested revision')
        git('update-ref', 'refs/cache/' + commit, commit)
        temporary = destination.with_name('snapshot.git.part')
        git('archive', '--format=zip', '--prefix=' + repository.split('/')[1] + '-' + commit + '/',
            '--output=' + str(temporary), commit)
        validate_archive(temporary, repository, commit)
        temporary.replace(destination)
        provenance = {'transport': 'git_archive', 'remote': remote, 'commit': commit,
                      'mirror': str(mirror), 'fetch_attempts': attempts}
        warmup.save(destination.with_name('archive_transport.json'), provenance)
        return [provenance]


def acquire(repository, commit, min_free_gb, transport='curl'):
    if (not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository or '')
            or any(part in ('.', '..') for part in repository.split('/'))):
        raise ValueError('Invalid repository')
    if not re.fullmatch('[0-9a-f]{40}', commit or ''):
        raise ValueError('A full pinned revision is required')
    base = PROJECT / 'repos' / repository / commit
    with warmup.lock(PROJECT / 'repos/.warmup-locks' / (warmup.digest([repository, commit]) + '.lock'), blocking=True):
        if shutil.disk_usage(PROJECT / 'repos').free < min_free_gb * 1024 ** 3:
            raise OSError(28, 'Source volume free-space reserve reached')
        attempts = []
        if not (base / 'snapshot.json').exists() and not (base / 'source').exists():
            base.mkdir(parents=True, exist_ok=True)
            archive = base / 'snapshot.zip'
            if not archive.exists():
                attempts = (git_archive if transport == 'git' else download_archive)(repository, commit, archive)
        # Reuse the original source validation/materialization and manifest format.
        source, manifest = warmup.get_snapshot(PROJECT, repository, commit)
        provenance = base / 'archive_transport.json'
        if provenance.exists() and 'source_transport' not in manifest:
            manifest['source_transport'] = json.loads(provenance.read_text())
            warmup.save(base / 'snapshot.json', manifest)
        return {'source': str(source), 'files': len(manifest['files']),
                'manifest_sha256': warmup.digest(manifest), 'download_attempts': attempts}


def run(plan_path, folder, previous=None, limit=None, transport='curl'):
    from datasets import load_from_disk
    plan = json.loads(plan_path.read_text())
    groups, splits, files = {}, {}, {}
    for job in plan['jobs']:
        _, base = warmup.load_job(PROJECT / job['config'])
        if base['dataset'] != 'BeetleBox' or base['split'] in splits:
            continue
        splits[base['split']] = base
        root = PROJECT / 'datasets/BeetleBox'
        for f in sorted((root / base['split']).glob('*')):
            if f.is_file():
                files[str(f.relative_to(root))] = warmup.sha256_file(f)
        for key in warmup.grouped_rows(load_from_disk(str(root))[base['split']]):
            groups.setdefault(key, None)
    identity = {'datasets': splits, 'dataset_files': files, 'limit': limit,
                'implementation': {**warmup.implementation_identity(),
                                   str(Path(__file__).relative_to(PROJECT)): warmup.sha256_file(Path(__file__))},
                'download': {'transport': transport, 'curl_http2': transport == 'curl', 'attempts': 3, 'max_seconds_per_attempt': 1800,
                             'max_archive_bytes': MAX_ARCHIVE_BYTES, 'max_source_bytes': MAX_SOURCE_BYTES}}
    with warmup.lock(folder / 'prefetch.lock'):
        warmup.pin(folder / 'inputs.json', identity)
        database = folder / 'progress.sqlite3'
        if previous and not database.exists():
            old_inputs = json.loads((previous / 'inputs.json').read_text())
            if old_inputs['datasets'] != splits or old_inputs.get('dataset_files', old_inputs.get('files')) != files:
                raise ValueError('Previous prefetch used different dataset inputs')
            with sqlite3.connect(f'file:{previous / "progress.sqlite3"}?mode=ro', uri=True) as old:
                with sqlite3.connect(database) as new:
                    old.backup(new)
            warmup.save(folder / 'migration.json', {'previous': str(previous), 'at': warmup.now(),
                         'policy': 'Retain completed checkpoints; retry every failed or pending revision'})
        progress = warmup.Progress(database)
        work = [(r, c) for r, c in groups if progress.should_run('snapshot', r + '@' + c, True)]
        if limit is not None:
            work = work[:limit]
        state = 'running'

        def checkpoint(active=None):
            warmup.save(folder / 'status.json', {'status': state, 'updated_at': warmup.now(),
                        'snapshots_total': len(groups), 'attempts_this_run': len(work),
                        **progress.counts(), 'active': active, 'downloader': 'git_archive' if transport == 'git' else 'curl_http2'})

        def process(item):
            repo, commit = item
            try:
                return repo + '@' + commit, 'complete', acquire(repo, commit, plan['min_free_gb'], transport)
            except Exception as error:
                if warmup.fatal(error):
                    raise
                return repo + '@' + commit, 'failed', {'error': str(error), 'traceback': traceback.format_exc()}

        try:
            checkpoint()
            pending = deque()
            items = iter(work)
            with ThreadPoolExecutor(max_workers=2) as pool:
                for _ in range(2):
                    item = next(items, None)
                    if item is not None:
                        pending.append(pool.submit(process, item))
                while pending:
                    key, status, detail = pending.popleft().result()
                    progress.put('snapshot', key, status, detail)
                    print(status + ' ' + key, flush=True)
                    checkpoint(key)
                    item = next(items, None)
                    if item is not None:
                        pending.append(pool.submit(process, item))
            failures = progress.failures()
            warmup.save(folder / 'failures.json', failures)
            counts = progress.counts()
            state = ('partial' if counts.get('snapshot_complete', 0) + counts.get('snapshot_failed', 0) < len(groups)
                     else 'completed_with_failures' if failures else 'completed')
            checkpoint()
        except BaseException as error:
            state = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            warmup.save(folder / 'failures.json', progress.failures())
            checkpoint(str(error))
            raise
        finally:
            progress.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--transport', choices=['curl', 'git'], default='curl')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    run(args.config, args.output, args.resume_from, args.limit, args.transport)
