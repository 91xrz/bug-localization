"""Materialize verified local Git archives rejected by conservative preparation limits."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import tempfile
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('warmup', PROJECT / 'scripts/retrieval/warm_embedding_cache.py')
warmup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warmup)


def materialize(project, repository, commit, archive, max_archive, max_source, reserve):
    """Caller owns the same snapshot lock used by retrieval and source prefetch."""
    if (not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository)
            or any(p in ('.', '..') for p in repository.split('/'))
            or not re.fullmatch('[0-9a-f]{40}', commit)):
        raise ValueError('Invalid repository or pinned commit')
    base = project / 'repos' / repository / commit
    source = base / 'source'
    if (base / 'snapshot.json').exists():
        _, manifest = warmup.get_snapshot(project, repository, commit)
        return manifest
    if source.exists():
        raise ValueError('Existing incomplete source requires separate review')
    if archive.stat().st_size > max_archive:
        raise ValueError('Archive exceeds the recovery size limit')
    expected_root = repository.split('/')[1] + '-' + commit
    with zipfile.ZipFile(archive) as zipped:
        if zipped.comment.decode('ascii', errors='replace') != commit:
            raise ValueError('Git archive comment does not identify the exact requested commit')
        files = []
        seen = set()
        for item in zipped.infolist():
            path = PurePosixPath(item.filename)
            parts = path.parts
            if (path.is_absolute() or not parts or parts[0] != expected_root
                    or '\\' in item.filename or any(part in ('.', '..') for part in parts)):
                raise ValueError('Unsafe archive path: ' + item.filename)
            if item.is_dir():
                continue
            if len(parts) < 2:
                raise ValueError('Archive root cannot be a regular file')
            relative = PurePosixPath(*parts[1:]).as_posix()
            if relative in seen:
                raise ValueError('Duplicate archive member: ' + relative)
            seen.add(relative)
            files.append((item, relative))
        size = sum(item.file_size for item, _ in files)
        if not files or size > max_source:
            raise ValueError('Expanded archive exceeds the recovery size limit or has no files')
        if shutil.disk_usage(base).free < size + reserve:
            raise OSError(28, 'Insufficient free space for this snapshot plus reserve')
        archive_hash = warmup.sha256_file(archive)
        stage = Path(tempfile.mkdtemp(prefix='source.repair-', dir=base))
        inventory = {}
        written = 0
        try:
            for number, (item, relative) in enumerate(files):
                if number % 256 == 0 and shutil.disk_usage(base).free < reserve:
                    raise OSError(28, 'Free-space reserve reached during extraction')
                destination = stage / relative
                if not destination.resolve().is_relative_to(stage.resolve()):
                    raise ValueError('Archive path escapes staging directory')
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                count = 0
                # Zip symlinks are ordinary files containing link text, matching
                # the original ZIP provider. No checkout or repository code runs.
                with zipped.open(item) as src, destination.open('xb') as dst:
                    while block := src.read(1024 * 1024):
                        count += len(block)
                        written += len(block)
                        if written > max_source or count > item.file_size:
                            raise ValueError('Expanded bytes exceed declared size or recovery limit')
                        digest.update(block)
                        dst.write(block)
                if count != item.file_size:
                    raise ValueError('Truncated archive member')
                inventory[relative] = digest.hexdigest()
            # Reading every member above checks its ZIP CRC before publication.
            transport = {'transport': 'git_archive', 'commit': commit,
                         'archive_file': archive.name, 'crc_verified': True,
                         'recovery': 'larger validated archive; POSIX colon filenames retained',
                         'max_archive_bytes': max_archive, 'max_source_bytes': max_source}
            manifest = {'repository': repository, 'commit': commit,
                        'source_url': f'https://codeload.github.com/{repository}/zip/{commit}',
                        'archive_bytes': archive.stat().st_size, 'archive_sha256': archive_hash,
                        'files': inventory, 'source_transport': transport}
            stage.rename(source)
            warmup.save(base / 'snapshot.json', manifest)
            warmup.save(base / 'archive_transport.json', transport)
            return manifest
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def run(audit_path, previous, output, max_archive, max_source, reserve):
    audit = json.loads(audit_path.read_text())
    with warmup.lock(output / 'repair.lock'):
        warmup.pin(output / 'inputs.json', {'audit': str(audit_path), 'audit_sha256': warmup.sha256_file(audit_path),
                    'script_sha256': warmup.sha256_file(Path(__file__)), 'max_archive_bytes': max_archive,
                    'max_source_bytes': max_source, 'reserve_bytes': reserve})
        progress = warmup.Progress(output / 'progress.sqlite3')
        state = 'running'

        def checkpoint(active=None):
            warmup.save(output / 'status.json', {'status': state, 'updated_at': warmup.now(),
                        'total': len(audit), **progress.counts(), 'active': active})

        try:
            checkpoint()
            for row in audit:
                repo, commit = row['repository'], row['commit']
                key = repo + '@' + commit
                if not progress.should_run('snapshot', key, False):
                    continue
                if not row.get('archive_exists') and not (PROJECT / 'repos' / repo / commit / 'snapshot.json').exists():
                    progress.put('snapshot', key, 'deferred', {'reason': 'No complete local Git archive'})
                    continue
                checkpoint(key)
                try:
                    archive = PROJECT / 'repos' / repo / commit / 'snapshot.git.part'
                    with warmup.lock(PROJECT / 'repos/.warmup-locks' / (warmup.digest([repo, commit]) + '.lock'), blocking=True):
                        manifest = materialize(PROJECT, repo, commit, archive, max_archive, max_source, reserve)
                    detail = {'updated_at': warmup.now(), 'files': len(manifest['files']),
                              'manifest_sha256': warmup.digest(manifest), 'recovered_by': str(output)}
                    progress.put('snapshot', key, 'complete', detail)
                    # Update only still-failed source-preparation rows. SQLite
                    # serializes these short transactions with the active downloader.
                    with sqlite3.connect(previous / 'progress.sqlite3', timeout=30) as db:
                        db.execute("UPDATE items SET status='complete',detail=? WHERE kind='snapshot' AND key=? AND status='failed'",
                                   (json.dumps(detail), key))
                    print('complete ' + key, flush=True)
                except Exception as error:
                    if warmup.fatal(error):
                        raise
                    progress.put('snapshot', key, 'failed', {'error': str(error)})
                    print('failed ' + key + ': ' + str(error), flush=True)
                checkpoint(key)
            state = 'completed_with_exceptions' if any(k.endswith(('_failed', '_deferred')) for k in progress.counts()) else 'completed'
            warmup.save(output / 'failures.json', progress.failures())
            checkpoint()
        except BaseException as error:
            state = 'failed'
            checkpoint(str(error))
            raise
        finally:
            progress.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--source-progress', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-archive-gib', type=float, required=True)
    parser.add_argument('--max-source-gib', type=float, required=True)
    parser.add_argument('--reserve-gib', type=int, default=100)
    args = parser.parse_args()
    if min(args.max_archive_gib, args.max_source_gib, args.reserve_gib) <= 0:
        parser.error('All size limits must be positive')
    run(args.audit, args.source_progress, args.output, int(args.max_archive_gib * 1024**3),
        int(args.max_source_gib * 1024**3), args.reserve_gib * 1024**3)
