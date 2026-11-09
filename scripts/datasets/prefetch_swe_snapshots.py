"""Prepare frozen SWE snapshots on disk before running retrieval experiments."""
import concurrent.futures
import datetime
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bug_localization.snapshot import get_snapshot, sha256_file

OUT = ROOT / 'results/full_semantic/swe_20261029/snapshot_download'
STAGE = ROOT / 'cache/snapshot-prefetch'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(pair):
    repository, commit = pair
    target = ROOT / 'repos' / repository / commit
    try:
        if (target / 'snapshot.json').exists():
            _, manifest = get_snapshot(ROOT, repository, commit)
            archive = target / 'snapshot.zip'
            if archive.exists() and sha256_file(archive) != manifest['archive_sha256']:
                raise ValueError('Existing archive checksum differs from manifest')
            return pair, 'verified_existing', None
        base = STAGE / 'repos' / repository / commit
        # Isolated, incomplete downloads never become experiment snapshots.
        if base.exists():
            shutil.rmtree(base)  # Only this downloader's staging copy; final snapshots are untouched.
        base.mkdir(parents=True, exist_ok=True)
        archive = base / 'snapshot.zip'
        existing = target / 'snapshot.zip'
        if existing.exists():
            shutil.copy2(existing, archive)
        else:
            part = base / 'download.part'
            url = f'https://codeload.github.com/{repository}/zip/{commit}'
            logfile = OUT / 'requests' / f"{repository.replace('/', '--')}-{commit}.log"
            logfile.parent.mkdir(parents=True, exist_ok=True)
            with logfile.open('a') as log:
                log.write(f'\nSTART {now()}\n'); log.flush()
                subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                    '--connect-timeout', '30', '--max-time', '900', '--speed-limit', '1024',
                    '--speed-time', '90', '--retry', '4', '--retry-all-errors', '--retry-delay', '10',
                    '--max-filesize', '100000000', '--output', str(part), url],
                    stdout=log, stderr=log, check=True, timeout=4800)
            part.rename(archive)
        if archive.stat().st_size > 100_000_000:
            raise ValueError('Archive exceeds project size limit')
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist()) > 500_000_000:
                raise ValueError('Uncompressed archive exceeds project size limit')
            if z.testzip() is not None:
                raise ValueError('ZIP CRC check failed')
        get_snapshot(STAGE, repository, commit)
        get_snapshot(STAGE, repository, commit)  # Verify extracted inventory against manifest.
        if target.exists():
            backup = OUT / 'previous_incomplete' / repository / f'{commit}-{time.time_ns()}'
            backup.parent.mkdir(parents=True, exist_ok=True)
            target.rename(backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        base.rename(target)
        return pair, 'downloaded_verified', None
    except Exception as exc:
        return pair, 'failed', f'{type(exc).__name__}: {exc}'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'download.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((OUT.parent / 'plan.json').read_text())
    pairs = set()
    for batch in plan['batches']:
        for row in json.loads(Path(batch['selection']).read_text())['samples']:
            pairs.add((row['repository'], row['before_fix_sha']))
    pending = sorted(pairs, key=lambda p: ((ROOT / 'repos' / p[0] / p[1] / 'snapshot.json').exists(), p))
    verified = set()
    failures = {}
    round_number = 0
    print(f'{now()} START {len(pairs)} unique snapshots, workers=2', flush=True)
    while pending:
        round_number += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(prepare, pair) for pair in pending]
            for future in concurrent.futures.as_completed(futures):
                pair, status, error = future.result()
                key = '/'.join(pair)
                if error:
                    failures[key] = error
                else:
                    verified.add(pair)
                    failures.pop(key, None)
                record = {'time': now(), 'snapshot': key, 'status': status, 'error': error}
                with (OUT / 'events.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                state = {'updated_at': now(), 'total': len(pairs), 'verified': len(verified),
                    'remaining': len(pairs) - len(verified), 'round': round_number,
                    'status': 'completed' if len(verified) == len(pairs) else 'running', 'failures': failures}
                tmp = OUT / 'status.json.tmp'
                tmp.write_text(json.dumps(state, indent=2) + '\n')
                tmp.replace(OUT / 'status.json')
                print(f"{now()} {len(verified)}/{len(pairs)} {status} {key}" + (f' {error}' if error else ''), flush=True)
        pending = [p for p in pending if p not in verified]
        if pending:
            print(f'{now()} Retrying {len(pending)} failed snapshots in 60s', flush=True)
            time.sleep(60)
    print(f'{now()} All snapshots verified; retrieval remains stopped.', flush=True)


if __name__ == '__main__':
    main()
