"""Launch the frozen RRL full evaluation and release its own GPU server on exit."""
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bug_localization.batch import write_json
from bug_localization.snapshot import sha256_file
from bug_localization.vector import select_reports


def read(path):
    return json.loads(path.read_text())


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def launch(output, folder):
    config_path = ROOT / 'configs/experiments/rrl_full_swe.json'
    config = read(config_path)
    previous = ROOT / 'results/rrl/crossproject500-gpu1-remaining-20261116'
    reference_config = read(previous / 'effective_config.json')
    reference = read(previous / 'summary.json')
    allowed_changes = {'selection_manifest', 'repositories', 'limit'}
    if {k: v for k, v in config.items() if k not in allowed_changes} != {
            k: v for k, v in reference_config['rrl'].items() if k not in allowed_changes}:
        raise ValueError('RRL settings differ from the successful 500-sample configuration')
    base = read(ROOT / config['base_config'])
    if base != reference_config['base']:
        raise ValueError('Shared candidate/query settings changed')
    for name, checksum in reference['implementation_sha256'].items():
        if sha256_file(ROOT / 'src/bug_localization' / name) != checksum:
            raise ValueError('RRL implementation changed: ' + name)
    selected, selection = select_reports(ROOT, config, base, max_samples=5000)
    if len(selected) != 2294 or len({s['dataset_row'] for s in selected}) != 2294:
        raise ValueError('Require all 2294 distinct dataset rows')
    if sha256_file(ROOT / config['model_file']) != reference['model']['sha256']:
        raise ValueError('GGUF differs from the successful 500-sample run')
    for name, checksum in reference['model']['tokenizer'].items():
        if sha256_file(ROOT / config['tokenizer_path'] / name) != checksum:
            raise ValueError('Tokenizer differs from reference')
    if output.exists():
        raise ValueError('Output already exists; preserve it and use a verified remaining selection for recovery')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 8097))
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    free = {int(a): int(b) for a, b in (line.split(',') for line in memory.splitlines())}
    if free.get(1, 0) < 10000:
        raise ValueError('GPU1 has less than 10000 MiB free')
    server_args = ['.runtime/llama/rrl-cuda-build/bin/llama-server', '-m', config['model_file'],
                   '--host', '127.0.0.1', '--port', '8097', '--alias', config['model'],
                   '-c', '16384', '-np', '1', '-t', '8', '-ngl', '99']
    client_args = [str(ROOT / '.venv/bin/python'), '-u', 'scripts/retrieval/run_rrl.py',
                   '--config', str(config_path), '--output', str(output)]
    files = [config_path, ROOT / config['base_config'], ROOT / config['selection_manifest'],
             Path(__file__), ROOT / 'scripts/retrieval/run_rrl.py', ROOT / server_args[0],
             *(ROOT / 'src/bug_localization' / (name + '.py') for name in
               ('rrl', 'rrl_index', 'bm25', 'snapshot', 'vector', 'batch', 'blaze'))]
    protocol = {'created_at': now(), 'physical_gpu': 1, 'selection_count': 2294,
                'result_directory': str(output), 'reference_run': str(previous),
                'server_argv': server_args, 'client_argv': client_args,
                'files_sha256': {str(p.resolve()): sha256_file(p) for p in files},
                'automatic_server_shutdown': True}
    server = client = None
    def status(stage, **extra):
        write_json(folder / 'status.json', {'updated_at': now(), 'status': stage,
            'pid': os.getpid(), 'server_pid': server.pid if server else None,
            'client_pid': client.pid if client else None, **extra})
    try:
        with (folder / 'server.log').open('w') as server_log, (folder / 'run.log').open('w') as run_log:
            server = subprocess.Popen(server_args, cwd=ROOT, start_new_session=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1'}, stdout=server_log, stderr=subprocess.STDOUT)
            status('loading_model')
            for _ in range(120):
                if server.poll() is not None:
                    raise RuntimeError('RRL server exited during loading')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8097/props', timeout=2) as response:
                        properties = json.load(response)
                    break
                except (OSError, ValueError):
                    time.sleep(1)
            else:
                raise TimeoutError('RRL server did not become ready')
            for key in ('default_generation_settings', 'total_slots', 'model_alias', 'model_ftype', 'build_info'):
                if properties.get(key) != reference['server_properties'].get(key):
                    raise ValueError('RRL server differs from reference: ' + key)
            if Path(properties['model_path']).resolve() != (ROOT / config['model_file']).resolve():
                raise ValueError('Wrong served model path')
            protocol.update(server_properties=properties, server_properties_match_previous500=True)
            write_json(folder / 'launch_protocol.json', protocol)
            client = subprocess.Popen(client_args, cwd=ROOT, start_new_session=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '8',
                     'TOKENIZERS_PARALLELISM': 'false'}, stdout=run_log, stderr=subprocess.STDOUT)
            while client.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError('RRL server exited; stopping client to avoid false sample failures')
                summary_path = output / 'summary.json'
                summary = read(summary_path) if summary_path.exists() else {}
                status('running', **{k: summary[k] for k in
                    ('attempted_samples', 'successful_samples', 'failed_samples', 'pending_samples') if k in summary})
                time.sleep(5)
            summary = read(output / 'summary.json')
            rows = read(output / 'samples.json')
            if (summary['status'] != 'completed' or summary['attempted_samples'] != 2294
                    or len(rows) != 2294 or {r['dataset_row'] for r in rows} != set(range(2294))
                    or client.returncode != int(summary['failed_samples'] > 0)):
                raise ValueError('RRL exited without a complete full-corpus result')
            status('completed_with_failures' if summary['failed_samples'] else 'completed',
                   attempted_samples=2294, failed_samples=summary['failed_samples'], metrics=summary['metrics'])
    except BaseException as error:
        status('failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stop(client)
        stop(server)
        write_json(folder / 'shutdown.json', {'updated_at': now(),
            'server_pid': server.pid if server else None, 'server_stopped': server is None or server.poll() is not None,
            'client_pid': client.pid if client else None, 'client_exit_code': client.poll() if client else None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    folder = output.with_name(output.name + '-launch')
    folder.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise InterruptedError('Launcher interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (folder / 'launcher.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            launch(output, folder)
        except BaseException as error:
            write_json(folder / 'launcher_error.json', {'updated_at': now(),
                'error_type': type(error).__name__, 'error': str(error)})
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
