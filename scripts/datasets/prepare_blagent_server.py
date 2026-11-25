"""Convert and launch the existing Qwen3-8B checkpoint as a two-slot BF16 server."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src'))
from bug_localization.snapshot import sha256_file
from bug_localization.model_files import required_model_files
from bug_localization.blagent_runtime import request_json

COMMIT='b2e5e9b28b2484fbf94b543432ece638996a8b97'


def prepare(start=False, cpu=False, require_new=False, visible_gpus='0,1', gpu_layers=99, tensor_split=None):
    source=PROJECT/'models/Qwen3-8B'
    folder=PROJECT/'models/BLAgent-Qwen3-8B-BF16';folder.mkdir(parents=True,exist_ok=True)
    model=folder/'Qwen3-8B-BF16.gguf'
    converter=PROJECT/'.runtime/rrl-model-source/llama.cpp'
    server=PROJECT/'.runtime/llama/rrl-cuda-build/bin/llama-server'
    if subprocess.check_output(['git','-C',str(converter),'rev-parse','HEAD'],text=True).strip()!=COMMIT:
        raise ValueError('Unexpected llama.cpp converter revision')
    if not model.exists():
        env=dict(os.environ,PYTHONPATH=str(PROJECT/'.runtime/rrl-model-deps'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8')
        with (folder/'conversion.log').open('w') as log:
            subprocess.run([sys.executable,str(converter/'convert_hf_to_gguf.py'),str(source),
                            '--outfile',str(model),'--outtype','bf16'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    # Validate GGUF fully before blessing an existing conversion artifact.
    sys.path.insert(0,str(converter/'gguf-py'))
    from gguf import GGUFReader
    reader=GGUFReader(model)
    if len(reader.tensors)!=399 or any(str(t.tensor_type) not in ('0','30') for t in reader.tensors):
        raise ValueError('Expected complete unquantized BF16/F32 Qwen3 GGUF')
    record={'gguf_path':str(model.resolve()),'gguf_sha256':sha256_file(model),
            'source_files_sha256':{n:sha256_file(source/n) for n in required_model_files(source)},
            'llama_commit':COMMIT,'server_sha256':sha256_file(server),
            'converter_sha256':sha256_file(converter/'convert_hf_to_gguf.py'),
            'precision':'BF16 weights; no quantization; F16 KV cache','slots':2,'per_slot_context':32768,
            'server_version':subprocess.check_output([str(server),'--version'],stderr=subprocess.STDOUT,text=True)}
    target=folder/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=record:
        raise ValueError('Existing BLAgent server manifest differs')
    target.write_text(json.dumps(record,indent=2)+'\n')
    if start:
        endpoint='http://127.0.0.1:8099'
        try:
            props=request_json(endpoint,'/props',timeout=2)
        except Exception:
            # Never start over another listener, including one still loading.
            import socket
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',8099))==0:
                    raise ValueError('Port 8099 occupied but not ready; inspect existing process')
            cmd=[str(server),'-m',str(model),'--alias','blagent-qwen3-8b-bf16','--host','127.0.0.1','--port','8099',
                 '-c','65536','-np','2','-ngl','0' if cpu else str(gpu_layers),
                 '--split-mode','layer' if ',' in visible_gpus else 'none',
                 '-b','1024','-ub','128','-fa','off','--cache-type-k','f16','--cache-type-v','f16',
                 '--cont-batching','--no-context-shift','--cache-ram','1024','-t','8']
            if cpu:cmd+=['--no-op-offload']
            if ',' in visible_gpus:cmd+=['--tensor-split',tensor_split or ','.join('1' for _ in visible_gpus.split(','))]
            runtime=PROJECT/'results/blagent/recovery-20261118';runtime.mkdir(parents=True,exist_ok=True)
            with (runtime/'server.log').open('a') as log:
                process=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if cpu else visible_gpus),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (runtime/'server_process.json').write_text(json.dumps({'pid':process.pid,'command':cmd,
                'cuda_visible_devices':'' if cpu else visible_gpus},indent=2)+'\n')
            print('Server starting, PID',process.pid,flush=True)
            return
        if require_new:raise ValueError('Port 8099 already serves a process; queue requires its own new server')
        if Path(props['model_path']).resolve()!=model.resolve():raise ValueError('Port 8099 serves another model')
    print(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--start',action='store_true')
    p.add_argument('--cpu-smoke',action='store_true')
    p.add_argument('--require-new',action='store_true')
    p.add_argument('--visible-gpus',default='0,1')
    p.add_argument('--gpu-layers',type=int,default=99)
    p.add_argument('--tensor-split')
    args=p.parse_args();prepare(args.start,cpu=args.cpu_smoke,require_new=args.require_new,
                               visible_gpus=args.visible_gpus,gpu_layers=args.gpu_layers,tensor_split=args.tensor_split)
