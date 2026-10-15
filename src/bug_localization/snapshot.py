"""Fetch one pinned GitHub ZIP snapshot without cloning repository history."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.request import Request, urlopen
import zipfile


def sha256_file(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def snapshot_inventory(root):
    return {path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(root.rglob('*')) if path.is_file()}


def get_snapshot(project, repository, commit):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
        raise ValueError('Expected a GitHub owner/repository name')
    if not commit or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('A verified, full before_fix_sha is required; the Ye dataset needs separate commit resolution')
    owner, name = repository.split('/')
    base = project / 'repos' / owner / name / commit
    source = base / 'source'
    manifest_path = base / 'snapshot.json'
    url = f'https://codeload.github.com/{repository}/zip/{commit}'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['repository'] != repository or manifest['commit'] != commit:
            raise ValueError('Snapshot identity mismatch')
        if snapshot_inventory(source) != manifest['files']:
            raise ValueError('Cached source files changed; refusing to index an unverified snapshot')
        return source, manifest
    if source.exists():
        raise ValueError('Incomplete snapshot exists; review the directory before retrying: ' + str(base))
    base.mkdir(parents=True, exist_ok=True)
    archive = base / 'snapshot.zip'
    temporary = base / 'snapshot.zip.part'
    if not archive.exists():
        request = Request(url, headers={'User-Agent': 'bug-localization-bm25'})
        downloaded = 0
        with urlopen(request, timeout=45) as response, temporary.open('wb') as handle:
            while chunk := response.read(256 * 1024):
                downloaded += len(chunk)
                if downloaded > 100_000_000:
                    raise ValueError('Snapshot exceeds the 100 MB download limit')
                handle.write(chunk)
        temporary.rename(archive)
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        if sum(info.file_size for info in members) > 500_000_000:
            raise ValueError('Uncompressed snapshot exceeds 500 MB')
        top_levels = {PurePosixPath(info.filename).parts[0] for info in members}
        if len(top_levels) != 1 or next(iter(top_levels)) != name + '-' + commit:
            raise ValueError('Archive root does not match the requested repository revision')
        # Validate all paths before writing any member. Only read source; never run it.
        destinations = []
        for info in members:
            parts = PurePosixPath(info.filename).parts[1:]
            if '\\' in info.filename or any(p in ('.', '..') or ':' in p for p in parts):
                raise ValueError('Unsafe archive path')
            destination = source.joinpath(*parts).resolve()
            if not destination.is_relative_to(source.resolve()):
                raise ValueError('Archive path escapes the snapshot directory')
            destinations.append((info, destination))
        source.mkdir()
        for info, destination in destinations:
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(handle.read(info))
    manifest = {'repository': repository, 'commit': commit, 'source_url': url,
                'archive_bytes': archive.stat().st_size, 'archive_sha256': sha256_file(archive),
                'files': snapshot_inventory(source)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return source, manifest
