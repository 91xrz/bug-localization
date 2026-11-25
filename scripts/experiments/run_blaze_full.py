"""Resumable BLAZE full baseline with an owned ES node and bounded index lifetime."""
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.blaze import digest
from bug_localization.blaze_experiment import ROUTES, evaluate_case, validate_config
from bug_localization.blaze_model import BlazeEncoder
from bug_localization.bm25 import ranking_metrics
from bug_localization.snapshot import sha256_file
from bug_localization.vector import select_reports


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def state(folder, stage, **extra):
    save(folder / 'status.json', {'updated_at': now(), 'status': stage, 'pid': os.getpid(), **extra})


def cleanup_index(client, index, fingerprint):
    """Only remove the exact owned temporary index after saving its results."""
    if not index.startswith('bl-blaze-') or index != 'bl-blaze-' + fingerprint[:32]:
        raise ValueError('Unexpected temporary index identity')
    owner = {'project': 'bug-localization', 'method': 'BLAZE-release', 'fingerprint': fingerprint}
    if client.indices.get_mapping(index=index)[index]['mappings'].get('_meta') != owner:
        raise ValueError('Refusing to remove an unowned index')
    client.indices.delete(index=index)


def validate_bundle(folder, meta, bundle):
    if bundle['sample'] != meta:
        raise ValueError('Checkpoint sample identity mismatch')
    path = folder / bundle['result_file']
    if sha256_file(path) != bundle['result_sha256']:
        raise ValueError('Checkpoint result checksum mismatch')
    result = read(path)
    if set(bundle['records']) != set(ROUTES) or set(result['routes']) != set(ROUTES):
        raise ValueError('Incomplete three-route checkpoint')
    for route in ROUTES:
        row = bundle['records'][route]
        metrics = ranking_metrics([r['file_path'] for r in result['routes'][route]['ranking']], result['gold_files'], 10)
        if (any(row.get(k) != v for k, v in meta.items()) or row['status'] != 'ok'
                or row['metrics'] != metrics or result['routes'][route]['metrics'] != metrics
                or row['missing_gold'] != result['missing_gold']):
            raise ValueError('Checkpoint route identity/metrics mismatch')
    return result


def run(folder):
    from elasticsearch import Elasticsearch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Require CUDA_VISIBLE_DEVICES=0')
    config_path = ROOT / 'configs/experiments/blaze_full_swe.json'
    config = read(config_path); base = read(ROOT / config['base_config'])
    reference_root = ROOT / 'results/blaze-repair-20261116/retry'
    reference = read(reference_root / 'summary.json')
    old = read(reference_root / 'effective_config.json')
    if {k: v for k, v in config.items() if k not in ('base_config', 'selection_manifest', 'repositories', 'limit')} != {
            k: v for k, v in old['blaze'].items() if k not in ('base_config', 'selection_manifest', 'repositories', 'limit')}:
        raise ValueError('BLAZE method settings changed')
    if {k: v for k, v in base.items() if k not in ('endpoint', 'cluster_name')} != {
            k: v for k, v in old['base'].items() if k not in ('endpoint', 'cluster_name')}:
        raise ValueError('BLAZE candidate settings changed')
    validate_config(config, base)
    dependencies = {name: version(name) for name in reference['dependencies']}
    if dependencies != reference['dependencies']:
        raise ValueError('BLAZE dependencies differ from 500-sample reference')
    for name, checksum in reference['implementation_sha256'].items():
        if sha256_file(ROOT / 'src/bug_localization' / name) != checksum:
            raise ValueError('BLAZE implementation changed: ' + name)
    selected, selection = select_reports(ROOT, config, base, max_samples=5000)
    metas = selection['samples']
    if len(metas) != 2294 or {s['dataset_row'] for s in metas} != set(range(2294)):
        raise ValueError('Require all 2294 original rows')
    source_root = ROOT / 'results/recall_full/20261118_swe2294_v2'
    recall_protocol = read(source_root / 'protocol.json')
    cache_root = ROOT / recall_protocol['config']['cache_run']
    files = [Path(__file__), config_path, ROOT / config['base_config'],
             ROOT / config['selection_manifest'], source_root / 'protocol.json',
             reference_root / 'summary.json', reference_root / 'effective_config.json',
             *(ROOT / 'src/bug_localization' / name for name in reference['implementation_sha256'])]
    protocol = {'config': config, 'base': base, 'dependencies': dependencies,
                'reference_model': reference['model'], 'physical_gpu': 0,
                'files_sha256': {str(p.resolve()): sha256_file(p) for p in files},
                'index_policy': 'Private ES 29200; exact unchanged lexical scorer; delete only verified owned index after checkpoint',
                'failure_policy': 'Stop on integrity/infrastructure or route error; retain pending samples for recovery'}
    protocol_path = folder / 'protocol.json'
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError('Frozen full-run protocol changed')
    save(protocol_path, protocol)
    save(folder / 'selection.json', selection)
    records = {route: [] for route in ROUTES}
    completed = set()
    for meta in metas:
        checkpoint = folder / 'cases' / f"{meta['position']:04d}" / 'record.json'
        if checkpoint.exists():
            bundle = read(checkpoint)
            validate_bundle(folder, meta, bundle)
            completed.add(meta['position'])
            for route in ROUTES:
                records[route].append(bundle['records'][route])
    def publish():
        summaries = {}
        for route in ROUTES:
            rows = sorted(records[route], key=lambda r: r['position'])
            save(folder / route / 'samples.json', rows)
            summaries[route] = aggregate_results(rows, 10, len(metas))
            save(folder / route / 'summary.json', summaries[route])
        summary = {'status': 'completed' if len(completed) == len(metas) else 'running',
                   'variant': reference['variant'], 'selected_samples': len(metas),
                   'attempted_samples': len(completed), 'routes': summaries}
        save(folder / 'summary.json', summary)
        return summary
    publish()
    if len(completed) == len(metas):
        state(folder, 'completed', processed=len(completed), selected=len(metas))
        return
    for port in (29200, 29300):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
    runtime = ROOT / '.runtime/blaze-full-20261119-es'
    runtime.mkdir(parents=True, exist_ok=True)
    args = ['.runtime/elasticsearch/elasticsearch-9.5.5/bin/elasticsearch',
            '-Ecluster.name=blaze-full-20261119', '-Enode.name=blaze-full-node',
            '-Enetwork.host=127.0.0.1', '-Ehttp.port=29200', '-Etransport.port=29300',
            '-Ediscovery.type=single-node', '-Expack.security.enabled=false',
            '-Expack.security.enrollment.enabled=false', '-Expack.ml.enabled=false',
            '-Ethread_pool.search.size=1', '-Epath.data='+str(runtime/'data'), '-Epath.logs='+str(runtime/'logs')]
    server = None
    try:
        with (folder / 'es-console.log').open('a') as log:
            server = subprocess.Popen(args, cwd=ROOT, start_new_session=True,
                env={**os.environ, 'ES_JAVA_OPTS': '-Xms2g -Xmx2g'}, stdout=log, stderr=subprocess.STDOUT)
            state(folder, 'starting_private_es', server_pid=server.pid, processed=len(completed))
            with Elasticsearch(base['endpoint'], request_timeout=120) as client:
                for _ in range(90):
                    if server.poll() is not None:
                        raise RuntimeError('Private ES exited')
                    try:
                        info = dict(client.options(request_timeout=2, max_retries=0).info())
                        break
                    except Exception:
                        time.sleep(1)
                else:
                    raise TimeoutError('Private ES startup timed out')
                if info['cluster_name'] != base['cluster_name'] or info['version']['number'] != '9.5.5':
                    raise ValueError('Unexpected private ES identity')
                save(folder / 'server.json', {'argv': args, 'info': info})
                # This data directory is exclusive to this run; remove leftovers from an interrupted case.
                for index, mapping in client.indices.get_mapping(index='bl-blaze-*', allow_no_indices=True).items():
                    cleanup_index(client, index, mapping['mappings'].get('_meta', {}).get('fingerprint', ''))
                state(folder, 'loading_encoder', processed=len(completed), physical_gpu=0)
                encoder = BlazeEncoder(ROOT / config['model_path'], device=config['device'],
                                       batch_size=config['embedding_batch_size'], cpu_threads=config['cpu_threads'])
                if encoder.identity != reference['model']:
                    raise ValueError('Encoder differs from the 500-sample reference')
                save(folder / 'model.json', {'identity': encoder.identity, 'physical_gpu': 0,
                                            'device': config['device'], 'batch_size': encoder.batch_size})
                for meta, sample in zip(metas, selected):
                    if meta['position'] in completed:
                        continue
                    state(folder, 'running', processed=len(completed), selected=len(metas),
                          position=meta['position'], repository=meta['repository'], server_pid=server.pid)
                    started = time.monotonic()
                    result = evaluate_case(ROOT, base, config, sample['report'], encoder, client)
                    if result['route_errors']:
                        save(folder / 'last_route_error.json', {'sample': meta, 'errors': result['route_errors']})
                        raise RuntimeError('BLAZE route error; stopping for diagnosis')
                    expected = read(cache_root / 'snapshots' / digest([meta['repository'], meta['before_fix_sha']]) / 'candidates.json')['files']
                    if result['candidates'] != expected:
                        raise ValueError('Candidate inventory differs from full-corpus control')
                    case = folder / 'cases' / f"{meta['position']:04d}"
                    save(case / 'result.json', result)
                    bundle = {'sample': meta, 'result_file': str((case / 'result.json').relative_to(folder)),
                              'result_sha256': sha256_file(case / 'result.json'), 'records': {}}
                    for route in ROUTES:
                        bundle['records'][route] = {**meta, 'status': 'ok',
                            'metrics': result['routes'][route]['metrics'], 'missing_gold': result['missing_gold'],
                            'indexed_files': result['indexed_files'], 'elapsed_seconds': time.monotonic()-started,
                            'result_file': '../'+bundle['result_file'], 'result_sha256': bundle['result_sha256']}
                    validate_bundle(folder, meta, bundle)
                    save(case / 'record.json', bundle)  # all three routes commit together
                    for route in ROUTES:
                        records[route].append(bundle['records'][route])
                    completed.add(meta['position'])
                    publish()
                    cleanup_index(client, result['lexical']['index'], result['lexical']['index_fingerprint'])
                    print(f"{now()} BLAZE {len(completed)}/{len(metas)} {meta['repository']} #{meta['issue_id']} "
                          f"seconds={time.monotonic()-started:.1f} cache_hit={result['embedding_cache']['cache_hit']}", flush=True)
                state(folder, 'completed', processed=len(completed), selected=len(metas))
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=10)
        save(folder / 'server_shutdown.json', {'updated_at': now(), 'server_pid': server.pid if server else None,
            'server_stopped': server is None or server.poll() is not None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise InterruptedError('Interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (folder / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(folder)
        except BaseException as error:
            state(folder, 'interrupted' if isinstance(error, InterruptedError) else 'failed',
                  error_type=type(error).__name__, error=str(error))
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
