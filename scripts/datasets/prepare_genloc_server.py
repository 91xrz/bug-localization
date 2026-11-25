"""Launch an isolated, local BF16 Qwen3 server for GenLoc tool calling."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import write_json
from bug_localization.snapshot import sha256_file


def prepare(start=False, visible_gpus='0', context=32768, port=8101,
            output='results/genloc/setup-20261120'):
    folder = PROJECT / output
    folder.mkdir(parents=True, exist_ok=True)
    source_manifest = PROJECT / 'models/BLAgent-Qwen3-8B-BF16/manifest.json'
    source = json.loads(source_manifest.read_text())
    model = Path(source['gguf_path'])
    server = PROJECT / '.runtime/llama/rrl-cuda-build/bin/llama-server'
    if sha256_file(model) != source['gguf_sha256']:
        raise ValueError('Existing BF16 model checksum differs from its manifest')
    if sha256_file(server) != source['server_sha256']:
        raise ValueError('Existing llama.cpp binary checksum differs from its manifest')
    cmd = [str(server), '-m', str(model), '--alias', 'genloc-qwen3-8b-bf16',
           '--host', '127.0.0.1', '--port', str(port), '-c', str(context),
           '-np', '1', '-ngl', '99', '--split-mode', 'none', '-b', '1024',
           '-ub', '128', '-fa', 'on', '--cache-type-k', 'f16',
           '--cache-type-v', 'f16', '--no-context-shift', '--cache-ram', '1024',
           '-t', '8', '--jinja', '--reasoning', 'off',
           '--chat-template-kwargs', '{"enable_thinking":false}']
    record = {**source, 'slots': 1, 'per_slot_context': context,
              'source_manifest_sha256': sha256_file(source_manifest),
              'server_binary': str(server.resolve()), 'command': cmd,
              'cuda_visible_devices': visible_gpus,
              'inference_endpoint': f'http://127.0.0.1:{port}',
              'model_alias': 'genloc-qwen3-8b-bf16', 'thinking': False,
              'no_context_shift': True, 'flash_attention': True}
    target = folder / 'server_manifest.json'
    if target.exists() and json.loads(target.read_text()) != record:
        raise ValueError('Existing GenLoc server manifest differs; use a new output directory')
    write_json(target, record)
    if not start:
        print(target)
        return record
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            raise ValueError(f'Port {port} already occupied; no existing process was changed')
    with (folder / 'server.log').open('a') as log:
        process = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=visible_gpus),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(folder / 'server_process.json', {'pid': process.pid, 'command': cmd,
               'cuda_visible_devices': visible_gpus, 'manifest': str(target)})
    print(json.dumps({'pid': process.pid, 'manifest': str(target), 'endpoint': record['inference_endpoint']}), flush=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'GenLoc server exited with code {process.returncode}; inspect server.log')
        try:
            with urllib.request.urlopen(record['inference_endpoint'] + '/props', timeout=2) as response:
                props = json.load(response)
            if props.get('total_slots') != 1 or props.get('default_generation_settings', {}).get('n_ctx') != context:
                raise ValueError('Server context/slot settings differ from requested configuration')
            write_json(folder / 'server_properties.json', props)
            print('GenLoc server ready', flush=True)
            return record
        except (OSError, TimeoutError):
            time.sleep(1)
    raise TimeoutError('Server still loading after 60 seconds; process remains available in server_process.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--visible-gpus', default='0')
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--port', type=int, default=8101)
    parser.add_argument('--output', default='results/genloc/setup-20261120')
    args = parser.parse_args()
    prepare(args.start, args.visible_gpus, args.context, args.port, args.output)
