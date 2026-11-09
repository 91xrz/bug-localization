"""Resume the frozen run, then retry failed reports without overwriting original results."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.run_swe_full import (initialize, now, ROUTES, CONFIG, result_for,
    list_vector_indices, cleanup_new_indices, aggregate_results, write_json, digest)

FOLDER = ROOT / 'results/full_semantic/swe_20261029'
RETRY = FOLDER / 'retry_after_snapshot_download'


def key(row):
    return row['repository'], row['issue_id'], row['before_fix_sha']


def records(folder):
    output = {}
    for route in ROUTES:
        entries = json.loads((folder / route / 'samples.json').read_text())
        output[route] = {}
        for row in entries:
            item = dict(row)
            item['source_result_directory'] = str(folder)
            if item.get('result_file'):
                item['result_file'] = str(folder / item['result_file'])
            if key(row) in output[route]:
                raise ValueError('Duplicate result')
            output[route][key(row)] = item
    return output


def validate_result(folder, selection):
    summary = json.loads((folder / 'summary.json').read_text())
    if summary['selection_sha256'] != selection['selection_sha256'] or set(summary['routes']) != set(ROUTES):
        raise ValueError('Retry result differs from frozen selection')
    expected = {key(row) for row in selection['samples']}
    for entries in records(folder).values():
        if set(entries) != expected:
            raise ValueError('Retry result has incomplete or different reports')


def merge_successes(original, retry):
    # Only fill original failures; never select the better of two successful scores.
    for route in ROUTES:
        for identity, replacement in retry[route].items():
            if identity not in original[route]:
                raise ValueError('Unknown retry report')
            if original[route][identity]['status'] != 'ok' and replacement['status'] == 'ok':
                original[route][identity] = replacement


def main():
    RETRY.mkdir(parents=True, exist_ok=True)
    lock = (RETRY / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = initialize(FOLDER.resolve(), 10)
    ready = json.loads((FOLDER / 'snapshot_download/status.json').read_text())
    if ready['status'] != 'completed' or ready['remaining']:
        raise ValueError('Snapshot prefetch is not complete')
    print(now(), 'Resuming unfinished frozen batches', flush=True)
    with (FOLDER / 'driver.log').open('a') as log:
        subprocess.run([sys.executable, '-B', '-u', str(ROOT / 'scripts/experiments/run_swe_full.py'),
            '--output', str(FOLDER)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    state = json.loads((FOLDER / 'state.json').read_text())
    if len(state['completed']) != len(plan['batches']):
        raise ValueError('Original run still has unfinished batches')
    baseline = RETRY / 'original_state.json'
    if baseline.exists():
        if json.loads(baseline.read_text())['completed'] != state['completed']:
            raise ValueError('Original batch results changed since retry began')
    else:
        write_json(baseline, state)
    merged = {r: {} for r in ROUTES}
    retry_state_path = RETRY / 'state.json'
    retry_state = json.loads(retry_state_path.read_text()) if retry_state_path.exists() else {'completed': {}}
    for batch in plan['batches']:
        folder = Path(state['completed'][str(batch['id'])]['result'])
        original = records(folder)
        for route in ROUTES:
            if set(merged[route]) & set(original[route]):
                raise ValueError('Duplicate original report')
            merged[route].update(original[route])
        failed = {k for entries in original.values() for k, row in entries.items() if row['status'] != 'ok'}
        if not failed:
            continue
        if min(shutil.disk_usage(ROOT).free, shutil.disk_usage(RETRY).free) < 20 * 1024**3:
            raise RuntimeError('Less than 20 GiB free on SSD or result disk')
        manifest = json.loads(Path(batch['selection']).read_text())
        selected = [dict(r, position=i) for i, r in enumerate(
            (r for r in manifest['samples'] if key(r) in failed), 1)]
        if len(selected) != len(failed):
            raise ValueError('Failed reports missing from selection')
        manifest.update(samples=selected, selection_sha256=digest(selected),
                        selection_rule='Retry reports with any failed route; original frozen inputs')
        task = RETRY / f"batch_{batch['id']:04d}"
        task.mkdir(exist_ok=True)
        selection_path = task / 'selection.json'
        write_json(selection_path, manifest)
        logfile = task / 'run.log'
        result = result_for(logfile)
        inventory = task / 'preexisting.json'
        if result is None:
            write_json(inventory, list_vector_indices())
            print(now(), 'Retry batch', batch['id'], len(selected), 'reports', flush=True)
            with logfile.open('a') as log:
                subprocess.run([sys.executable, '-B', '-u', str(ROOT / 'scripts/retrieval/run_vector_batch.py'),
                    '--config', str(CONFIG), '--selection', str(selection_path)], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, check=False)
            result = result_for(logfile)
        if result is None:
            raise RuntimeError('Retry produced no completed result: ' + str(logfile))
        validate_result(result, manifest)
        cleanup_new_indices(result, set(json.loads(inventory.read_text())))
        retried = records(result)
        merge_successes(merged, retried)
        retry_state['completed'][str(batch['id'])] = {'result': str(result), 'updated_at': now(),
            'failures': {r: sum(x['status'] != 'ok' for x in retried[r].values()) for r in ROUTES}}
        write_json(retry_state_path, retry_state)
        print(now(), 'Retried', batch['id'], retry_state['completed'][str(batch['id'])]['failures'], flush=True)
    summaries = {}
    unresolved = []
    for route in ROUTES:
        rows = list(merged[route].values())
        if len(rows) != plan['total_samples']:
            raise ValueError('Merged report coverage differs from full dataset')
        write_json(RETRY / f'{route}_merged_samples.json', rows)
        summaries[route] = aggregate_results(rows, 10)
        unresolved.extend({'route': route, **r} for r in rows if r['status'] != 'ok')
    write_json(RETRY / 'unresolved.json', unresolved)
    write_json(RETRY / 'summary.json', {'updated_at': now(), 'status': 'completed_with_failures' if unresolved else 'completed',
        'note': 'Original successful results retained; only original failures replaced by successful retries.',
        'total_samples': plan['total_samples'], 'unresolved_route_reports': len(unresolved), 'routes': summaries})
    print(now(), 'Retries complete; unresolved route/report pairs:', len(unresolved), flush=True)


if __name__ == '__main__':
    main()
