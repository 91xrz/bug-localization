"""Offline Long Code Arena adapter: author Git archives, pinned base commits."""
import ast
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import uuid
import zipfile

from .snapshot import sha256_file, snapshot_inventory

PROTOCOL = 'lca_base_sha_v1'
SOURCE_EXTENSIONS = {'.java', '.py', '.pyi', '.kt', '.kts'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def safe_path(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive/source path: ' + name)
    return p


def git(repo, *args, input=None):
    env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_TERMINAL_PROMPT': '0'}
    result = subprocess.run(['git', '--no-replace-objects', '-c', 'core.fsmonitor=false',
                             '-C', str(repo), *args], input=input, capture_output=True,
                            timeout=600, env=env)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors='replace')[:1000])
    return result.stdout


def extract_archive(archive, destination, expected_sha256):
    """Extract regular files atomically; ZIP symlinks remain inert text, not links.

    Git blobs, not the archived checkout, supply all experimental source code.
    """
    marker = destination / '.lca-extraction.json'
    if marker.exists():
        if json.loads(marker.read_text())['archive_sha256'] != expected_sha256:
            raise ValueError('Extracted archive identity mismatch')
        return
    if destination.exists():
        raise ValueError('Incomplete extraction needs review: ' + str(destination))
    if sha256_file(archive) != expected_sha256:
        raise ValueError('Downloaded archive checksum mismatch: ' + str(archive))
    staging = destination.with_name(destination.name + '.preparing-' + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            for info in infos:
                safe_path(info.filename)
            for info in infos:
                target = staging / info.filename
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
        if not (staging / '.git/objects').is_dir():
            raise ValueError('Archive has no Git object database')
        if (staging / '.git/objects/info/alternates').exists():
            raise ValueError('External Git object alternates are not supported')
        # Do not trust downloaded Git config, hooks, or an archived checkout.
        # Read-only Git commands use local objects; no remote, filter or include config.
        config = staging / '.git/config'
        if config.exists():
            config.rename(staging / '.git/config.upstream')
        config.write_text('[core]\nrepositoryformatversion = 0\nbare = false\n', encoding='utf-8')
        save(staging / '.lca-extraction.json', {'archive_sha256': expected_sha256,
                                               'entries': len(infos), 'worktree_is_not_used': True})
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise


def tree_entries(repo, commit):
    if not re.fullmatch('[0-9a-f]{40}', commit or ''):
        raise ValueError('LCA requires a full base_sha')
    entries = []
    for record in git(repo, 'ls-tree', '-r', '-z', '--full-tree', commit).split(b'\0'):
        if not record:
            continue
        header, raw_path = record.split(b'\t', 1)
        mode, kind, blob = header.decode('ascii').split()
        name = raw_path.decode('utf-8')
        safe_path(name)
        if kind == 'blob' and mode in ('100644', '100755') and PurePosixPath(name).suffix.lower() in SOURCE_EXTENSIONS:
            entries.append({'path': name, 'blob': blob, 'mode': mode})
    return entries


def normalize_row(row, language):
    labels = ast.literal_eval(row['changed_files'])
    if not isinstance(labels, list) or not labels or not all(isinstance(p, str) and p for p in labels):
        raise ValueError('Invalid LCA changed_files')
    for name in labels:
        safe_path(name)
    commit = row['base_sha']
    if not re.fullmatch('[0-9a-f]{40}', commit or ''):
        raise ValueError('Missing/invalid base_sha')
    repository = row['repo_owner'] + '/' + row['repo_name']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
        raise ValueError('Invalid repository name')
    return {'repo_name': repository, 'issue_id': int(row['id']),
            'title': row['issue_title'] or '', 'body': row['issue_body'] or '',
            'before_fix_sha': commit, 'updated_files': repr(list(dict.fromkeys(labels))),
            'issue_url': row['issue_url'], 'pull_url': row['pull_url'],
            'lca_text_id': row['text_id'], 'lca_language': language,
            'source_revision_role': 'pull_request_base_sha', 'snapshot_protocol': PROTOCOL}


def prepare(project, raw_root, languages=('java', 'py', 'kt'), materialize_test=True):
    import pyarrow.parquet as pq
    from datasets import Dataset, DatasetDict

    status = json.loads((raw_root / 'metadata/download-status.json').read_text())
    if status['status'] != 'complete':
        raise ValueError('LCA downloads must be complete before preparation')
    summary = {'protocol': PROTOCOL, 'raw_root': str(raw_root.resolve()), 'languages': {}, 'status': 'running'}
    progress = raw_root / 'metadata/preparation-status.json'
    save(progress, summary)
    for lang in languages:
        output = project / 'datasets' / ('lca_' + lang + '_formatted')
        splits = {split: [normalize_row(r, lang) for r in pq.read_table(
                    next((raw_root / lang).glob(split + '-*.parquet'))).to_pylist()]
                  for split in ('test',)}
        grouped = defaultdict(set)
        for rows in splits.values():
            identities = [(r['repo_name'], r['issue_id']) for r in rows]
            if len(identities) != len(set(identities)):
                raise ValueError('Duplicate LCA repository/id in split ' + lang)
            for row in rows:
                grouped[row['repo_name']].add(row['before_fix_sha'])
        registry = {'protocol': PROTOCOL, 'raw_root': str(raw_root.resolve()), 'sources': {}}
        info = {'splits': {s: len(rows) for s, rows in splits.items()}, 'repositories': len(grouped),
                'verified_repositories': 0, 'verified_commits': 0, 'test_snapshots': 0}
        summary['languages'][lang] = info
        for repository, commits in sorted(grouped.items()):
            slug = repository.replace('/', '__')
            archive_name = 'repos/' + lang + '/' + slug + '.zip'
            archive_info = status['ready'][archive_name]
            repo = raw_root / 'repositories' / lang / slug
            print('Extract and verify', lang, repository, len(commits), flush=True)
            extract_archive(raw_root / archive_name, repo, archive_info['sha256'])
            commits = sorted(commits)
            result = git(repo, 'cat-file', '--batch-check=%(objectname) %(objecttype)',
                         input=('\n'.join(commits) + '\n').encode()).decode().splitlines()
            if result != [c + ' commit' for c in commits]:
                raise ValueError('Missing/non-commit base_sha in ' + repository + ': ' + repr(result[:5]))
            # Validate all referenced source objects without fetching from any remote.
            git(repo, 'fsck', '--connectivity-only', '--no-dangling')
            for commit in commits:
                tree = git(repo, 'rev-parse', commit + '^{tree}').decode().strip()
                registry['sources'][repository + '@' + commit] = {
                    'repository': repository, 'commit': commit, 'tree': tree,
                    'repo_path': repo.relative_to(raw_root).as_posix(),
                    'archive': archive_name, 'archive_sha256': archive_info['sha256']}
            info['verified_repositories'] += 1
            info['verified_commits'] += len(commits)
            save(progress, summary)
        if output.exists():
            existing = json.loads((output / 'lca_sources.json').read_text())
            if existing != registry:
                raise ValueError('Existing converted dataset differs; refusing overwrite')
        else:
            staging = output.with_name(output.name + '.preparing-' + uuid.uuid4().hex)
            DatasetDict({s: Dataset.from_list(rows) for s, rows in splits.items()}).save_to_disk(str(staging))
            save(staging / 'lca_sources.json', registry)
            save(staging / 'conversion_report.json', {'protocol': PROTOCOL, **info,
                 'issue_id_semantics': 'LCA row id, not GitHub issue number; original issue_url retained',
                 'split_warning': 'dev includes train and test; do not combine them as independent splits'})
            staging.rename(output)
        if materialize_test:
            for row in splits['test']:
                local_snapshot(project, output.name, row['repo_name'], row['before_fix_sha'])
                info['test_snapshots'] += 1
                save(progress, summary)
        save(output / 'preparation_report.json', info)
    summary['status'] = 'complete'
    save(progress, summary)
    return summary


def local_snapshot(project, dataset, repository, commit):
    if not re.fullmatch(r'lca_(java|py|kt)_formatted', dataset or ''):
        raise ValueError('Invalid LCA source dataset')
    if not re.fullmatch('[0-9a-f]{40}', commit or ''):
        raise ValueError('Invalid LCA base_sha')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository or ''):
        raise ValueError('Invalid LCA repository')
    registry = json.loads((project / 'datasets' / dataset / 'lca_sources.json').read_text())
    if registry['protocol'] != PROTOCOL:
        raise ValueError('Unknown LCA protocol')
    entry = registry['sources'].get(repository + '@' + commit)
    if not entry or entry['repository'] != repository or entry['commit'] != commit:
        raise ValueError('Unregistered LCA repository/base_sha')
    raw_root = Path(registry['raw_root'])
    repo = raw_root / safe_path(entry['repo_path'])
    if git(repo, 'rev-parse', commit + '^{tree}').decode().strip() != entry['tree']:
        raise ValueError('LCA tree identity mismatch')
    base = project / 'repos/lca' / dataset / repository / commit
    manifest_path = base / 'snapshot.json'
    source = base / 'source'
    provenance = {'protocol': PROTOCOL, **entry}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('repository') != repository or manifest.get('commit') != commit or
                manifest.get('source_provenance') != provenance or snapshot_inventory(source) != manifest['files']):
            raise ValueError('LCA snapshot identity or source contents changed')
        if 'source_url' not in manifest:
            manifest['source_url'] = 'https://huggingface.co/datasets/JetBrains-Research/lca-bug-localization/blob/f2bb26a1c3320ccc1d7b02d2630077ff327a80f6/' + entry['archive']
            save(manifest_path, manifest)
        return source, manifest
    if source.exists():
        raise ValueError('Incomplete LCA snapshot requires review')
    base.mkdir(parents=True, exist_ok=True)
    staging = base / ('source.preparing-' + uuid.uuid4().hex)
    staging.mkdir()
    entries = tree_entries(repo, commit)
    process = subprocess.Popen(['git', '--no-replace-objects', '-c', 'core.fsmonitor=false', '-C', str(repo),
                                'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env={**os.environ, 'GIT_NO_LAZY_FETCH': '1'})
    try:
        for item in entries:
            process.stdin.write((item['blob'] + '\n').encode()); process.stdin.flush()
            header = process.stdout.readline().decode().split()
            if len(header) != 3 or header[:2] != [item['blob'], 'blob']:
                raise ValueError('Missing source blob: ' + item['path'])
            size = int(header[2]); raw = process.stdout.read(size)
            if len(raw) != size or process.stdout.read(1) != b'\n':
                raise ValueError('Incomplete Git blob')
            if hashlib.sha1(b'blob ' + str(size).encode() + b'\0' + raw).hexdigest() != item['blob']:
                raise ValueError('Git blob hash mismatch')
            target = staging / item['path']; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        process.stdin.close()
        if process.wait(timeout=30):
            raise ValueError('Git object reader failed')
        staging.rename(source)
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
        if not process.stdin.closed: process.stdin.close()
        process.stdout.close(); process.stderr.close()
        if staging.exists(): shutil.rmtree(staging)
    manifest = {'repository': repository, 'commit': commit, 'source_provider': 'lca_git',
                'source_provenance': provenance, 'files': snapshot_inventory(source),
                'source_url': 'https://huggingface.co/datasets/JetBrains-Research/lca-bug-localization/blob/f2bb26a1c3320ccc1d7b02d2630077ff327a80f6/' + entry['archive'],
                'archive_sha256': entry['archive_sha256']}
    save(manifest_path, manifest)
    return source, manifest
