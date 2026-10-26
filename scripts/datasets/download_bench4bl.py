"""Download original Bench4BL subject archives; no repository code is executed."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import posixpath
import shutil
import tarfile
import threading
import time
from urllib.request import Request, urlopen

PROJECT = Path(__file__).resolve().parents[2]


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def fetch(root, entry, progress, lock, retries=5, timeout=120):
    group, name = entry['group'], entry['file']
    if group not in ('Apache', 'Commons', 'JBoss', 'Wildfly', 'Spring', 'Previous') or Path(name).name != name or not name.endswith('.tar'):
        raise ValueError('Invalid archive identity')
    url = entry['mirror_url']
    if url != 'https://phoenixnap.dl.sourceforge.net/project/irblsensitivity/' + group + '/' + name:
        raise ValueError('Unexpected mirror URL')
    target = root / '_archives' / group / name
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.tar.part')
    key = group + '/' + name
    expected = entry.get('bytes')
    if not target.exists():
        last_error = None
        for attempt in range(retries):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {'User-Agent': 'Bench4BL-research-download'}
                if offset:
                    headers['Range'] = f'bytes={offset}-'
                with urlopen(Request(url, headers=headers), timeout=timeout) as response:
                    if 'text/html' in response.headers.get('Content-Type', ''):
                        raise ValueError('Server returned HTML rather than an archive')
                    if response.status == 206:
                        if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                            raise ValueError('Resume range differs from local offset')
                        mode = 'ab'
                    else:
                        offset, mode = 0, 'wb'
                    length = response.headers.get('Content-Length')
                    response_total = offset + int(length) if length else None
                    if expected and response_total and response_total != expected:
                        raise ValueError('Remote archive size changed since manifest preparation')
                    with partial.open(mode) as handle:
                        while chunk := response.read(64 * 1024):
                            handle.write(chunk)
                            offset += len(chunk)
                            with lock:
                                progress[key] = offset
                    if (expected and offset != expected) or (response_total and offset != response_total):
                        raise ValueError('Incomplete archive transfer')
                # Validate TAR structure before publishing a completed download.
                with tarfile.open(partial) as archive:
                    archive.getmembers()
                partial.replace(target)
                last_error = None
                break
            except Exception as error:
                last_error = error
                time.sleep(1)
        if last_error:
            raise last_error
    if expected and target.stat().st_size != expected:
        raise ValueError('Existing archive size mismatch; preserved for review')
    with target.open('rb') as handle:
        checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
    destination = root / 'data' / group / Path(name).stem
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target) as archive:
        members = archive.getmembers()
        lookup = {posixpath.normpath(m.name): m for m in members}
        links = []
        # Validate every path before extraction. Never run scripts from the dataset.
        for member in members:
            resolved = (destination / member.name).resolve()
            if not resolved.is_relative_to(destination.resolve()) or '\\' in member.name or ':' in member.name:
                raise ValueError('Unsafe archive member path')
            if member.issym() or member.islnk():
                linkname = posixpath.normpath(posixpath.join(posixpath.dirname(member.name), member.linkname)
                                             if member.issym() else member.linkname)
                link_target = (destination / linkname).resolve()
                if (not link_target.is_relative_to(destination.resolve()) or '\\' in member.linkname
                        or ':' in member.linkname or linkname not in lookup or not lookup[linkname].isfile()):
                    raise ValueError('Link must resolve directly to a regular file inside the archive')
                links.append({'path': member.name, 'target': linkname, 'representation': 'regular_file_copy'})
            elif not (member.isfile() or member.isdir()):
                raise ValueError('Archive contains a special file; retain original TAR for review')
        unpacked = sum(m.size for m in members if m.isfile())
        if shutil.disk_usage(root).free < unpacked + 1024**3:
            raise ValueError('Insufficient space for safe extraction')
        archive.extractall(destination, members=[m for m in members if m.isfile() or m.isdir()], filter='data')
        # Materialize safe document links for unprivileged Windows compatibility.
        # Original symbolic-link metadata remains untouched in the original TAR.
        for link in links:
            shutil.copyfile(destination / link['target'], destination / link['path'])
        files = [m for m in members if m.isfile()]
        for member in files:
            if (destination / member.name).stat().st_size != member.size:
                raise ValueError('Extracted file size mismatch')
    with lock:
        progress[key] = target.stat().st_size
    return {**entry, 'status': 'complete', 'bytes': target.stat().st_size, 'sha256': checksum,
            'archive': target.relative_to(root).as_posix(), 'extracted_to': destination.relative_to(root).as_posix(),
            'extracted_files': len(files), 'unpacked_bytes': unpacked,
            'materialized_links': links,
            'verified_at': datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT / 'datasets/Bench4BL')
    parser.add_argument('--workers', type=int, choices=range(1, 7), default=4)
    parser.add_argument('--retries', type=int, choices=range(1, 11), default=5)
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'metadata/download_manifest.json').read_text())
    if len(manifest) != 51 or len({(e['group'], e['file']) for e in manifest}) != 51:
        raise ValueError('Expected the 51 original Bench4BL subject archives')
    results, progress, lock = {}, {}, threading.Lock()
    previous_path = root / 'download_status.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding='utf-8'))['subjects']
        for entry in manifest:
            key = entry['group'] + '/' + entry['file']
            old = previous.get(key, {})
            path = root / '_archives' / entry['group'] / entry['file']
            if (old.get('status') == 'complete' and path.is_file() and path.stat().st_size == old['bytes']
                    and (root / 'data' / entry['group'] / Path(entry['file']).stem).is_dir()):
                results[key] = old
                progress[key] = old['bytes']
    stop = threading.Event()
    def monitor():
        while not stop.wait(20):
            with lock:
                transferred = sum(progress.values())
            print(f'Transferred/reused: {transferred / 1e9:.3f} GB; completed subjects: {len(results)}/51', flush=True)
    reporter = threading.Thread(target=monitor, daemon=True)
    reporter.start()
    save(root / 'download_status.json', {'status': 'running', 'subjects': results,
         'started_at': datetime.now(timezone.utc).isoformat()})
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch, root, entry, progress, lock, args.retries, args.timeout): entry
                       for entry in manifest if entry['group'] + '/' + entry['file'] not in results}
            for future in as_completed(futures):
                entry = futures[future]
                key = entry['group'] + '/' + entry['file']
                try:
                    results[key] = future.result()
                except Exception as error:
                    results[key] = {**entry, 'status': 'failed', 'error': str(error)}
                save(root / 'download_status.json', {'status': 'running', 'subjects': results})
                print(key, results[key]['status'], results[key].get('error', ''), flush=True)
    finally:
        stop.set()
    status = 'complete' if all(r['status'] == 'complete' for r in results.values()) and len(results) == 51 else 'incomplete'
    save(root / 'download_status.json', {'status': status, 'source': 'https://github.com/exatoa/Bench4BL',
        'subjects': results, 'archive_bytes': sum(r.get('bytes', 0) for r in results.values() if r['status'] == 'complete')})
    print('Dataset status:', status, root, flush=True)
    if status != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
