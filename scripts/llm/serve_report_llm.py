"""Start the user-installed llama.cpp server; never downloads software or weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=os.environ.get('IRBL_LLM_FILE'), help='Path to user-provided GGUF')
    parser.add_argument('--server', default=os.environ.get('IRBL_LLAMA_SERVER'), help='Path to llama-server executable')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--gpu-layers', default='auto')
    parser.add_argument('--background', action='store_true', help='Detach and save stdout/stderr and PID')
    args = parser.parse_args()
    local = PROJECT / '.runtime/llama/local-runtime.json'
    saved = json.loads(local.read_text(encoding='utf-8-sig')) if local.exists() else {}
    model = Path(args.model or saved.get('model_path', ''))
    server = Path(args.server) if args.server else PROJECT / '.runtime/llama' / ('llama-server.exe' if os.name == 'nt' else 'llama-server')
    if not model.is_file() or not server.is_file():
        parser.error('Provide existing --model and --server paths (or IRBL_LLM_FILE / IRBL_LLAMA_SERVER)')
    if not 1 <= args.port <= 65535 or args.context < 1024:
        parser.error('Invalid port or context size')
    with model.open('rb') as handle:
        model_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    version = subprocess.run([str(server.resolve()), '--version'], capture_output=True, text=True, check=True)
    config = json.loads((PROJECT / 'configs/experiments/report_multiview.json').read_text(encoding='utf-8'))
    if not config['llm']['model_revision'].startswith('sha256:' + model_hash + ';'):
        parser.error('Model SHA256 differs from experiment model_revision; update the experiment config first: ' + model_hash)
    command = [str(server.resolve()), '-m', str(model.resolve()), '--host', '127.0.0.1', '--port', str(args.port),
               '--alias', config['llm']['model'], '--ctx-size', str(args.context), '--parallel', '1',
               '--gpu-layers', args.gpu_layers, '--reasoning', 'off', '--no-webui']
    logs = PROJECT / 'results/llm'
    logs.mkdir(parents=True, exist_ok=True)
    if args.background:
        options = {'creationflags': subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
        with (logs / 'server.stdout.log').open('ab') as stdout, (logs / 'server.stderr.log').open('ab') as stderr:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, **options)
        runtime = {'pid': process.pid, 'model_path': str(model.resolve()), 'model_sha256': model_hash,
                   'server_path': str(server.resolve()), 'version': version.stdout + version.stderr, 'command': command}
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(runtime, indent=2) + '\n', encoding='utf-8')
        print('Started PID', process.pid, '; check results/llm/server.stderr.log for readiness')
    else:
        print('Model SHA256:', model_hash, flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
