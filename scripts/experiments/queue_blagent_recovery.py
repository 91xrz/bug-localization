"""Wait for GPU capacity, validate failure recovery, then run uniform BLAgent 500.

Never interrupts other experiments. Own server is shut down after the experiment.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PROJECT/'src')]
from bug_localization.batch import write_json
from bug_localization.snapshot import sha256_file
from bug_localization.blagent_runtime import request_json


def read(path):return json.loads(path.read_text())


def gpu_free(gpu_indices=(0,1)):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    values=[int(line.strip()) for line in output.splitlines()]
    return [values[i] for i in gpu_indices]


def stop_owned_server(pid):
    path=Path(f'/proc/{pid}/cmdline')
    if not path.exists():return
    command=path.read_bytes().replace(b'\0',b' ').decode()
    if 'rrl-cuda-build/bin/llama-server' not in command or 'BLAgent-Qwen3-8B-BF16' not in command:
        raise ValueError('Refusing to stop a process with different server identity')
    try:os_kill=__import__('os').kill;os_kill(pid,signal.SIGTERM)
    except ProcessLookupError:return


def run(output, minimum_free_mib=18000, visible_gpus='0,1', gpu_layers=99,
        regression_config='configs/experiments/blagent_recovery19_v2.json',
        full_config='configs/experiments/blagent500_v2.json'):
    gpu_indices=tuple(int(i) for i in visible_gpus.split(','))
    full_settings=read(PROJECT/full_config)
    server_split=full_settings.get('server_tensor_split',','.join('1' for _ in gpu_indices))
    minimums=full_settings.get('minimum_free_mib_by_gpu',[minimum_free_mib]*len(gpu_indices))
    if len(minimums)!=len(gpu_indices):raise ValueError('GPU memory thresholds differ from device count')
    full_count=len(read(PROJECT/full_settings['selection_manifest'])['samples'])
    regression_count=len(read(PROJECT/read(PROJECT/regression_config)['selection_manifest'])['samples'])
    if not gpu_indices or len(set(gpu_indices))!=len(gpu_indices) or min(gpu_indices)<0:
        raise ValueError('Invalid GPU selection')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files=['src/bug_localization/blagent.py','src/bug_localization/blagent_recovery.py',
           'src/bug_localization/blagent_runtime.py','scripts/experiments/run_blagent_parallel.py',
           'scripts/analysis/audit_blagent.py','scripts/datasets/prepare_blagent_server.py',
           'scripts/experiments/queue_blagent_recovery.py',
           full_config,regression_config]
    for path in (full_config,regression_config):
        config=read(PROJECT/path)
        if config.get('server_gpu_layers',gpu_layers)!=gpu_layers or config.get('cuda_visible_devices',visible_gpus)!=visible_gpus:
            raise ValueError('Server placement differs from experiment configuration')
        files.extend([config['base_config'],config['semantic_config'],config['selection_manifest'],config['server_manifest']])
    hashes={p:sha256_file(PROJECT/p) for p in sorted(set(files))}
    protocol={'files_sha256':hashes,'minimum_free_mib_each_gpu':minimum_free_mib,'stable_capacity_checks':3,
              'cuda_visible_devices':visible_gpus,'server_gpu_layers':gpu_layers,
              'minimum_free_mib_by_gpu':minimums,'full_samples':full_count,'regression_samples':regression_count,
              'original_partial_run':'results/blagent/crossproject500-20261118','policy':'new uniform cohort, never overwrite/mix original results'}
    protocol_path=output/'queue_protocol.json'
    if protocol_path.exists() and read(protocol_path)!=protocol:raise ValueError('Queue protocol changed; choose a new output')
    write_json(protocol_path,protocol)
    status={'status':'waiting_for_gpu','created_at':datetime.now(timezone.utc).isoformat(),
            'gpu_indices':list(gpu_indices),'server_gpu_layers':gpu_layers,
            'full_samples':full_count,'minimum_free_mib_by_gpu':minimums,
            'minimum_free_mib_each_gpu':minimum_free_mib,'note':'Other experiments keep their allocated GPU memory.'}
    pid=None
    def update(**kwargs):
        status.update(**kwargs,updated_at=datetime.now(timezone.utc).isoformat());write_json(output/'queue_status.json',status)
    try:
        stable=0
        while stable<3:
            free=gpu_free(gpu_indices);stable=stable+1 if all(f>=m for f,m in zip(free,minimums)) else 0
            update(free_mib=free,stable_capacity_checks=stable)
            if stable<3:time.sleep(30)
        for p,checksum in hashes.items():
            if sha256_file(PROJECT/p)!=checksum:raise ValueError('Queued source/config changed: '+p)
        update(status='starting_inference_server')
        with (output/'prepare.log').open('a') as log:
            subprocess.run([sys.executable,str(PROJECT/'scripts/datasets/prepare_blagent_server.py'),'--start','--require-new',
                            '--visible-gpus',visible_gpus,'--gpu-layers',str(gpu_layers),'--tensor-split',server_split],
                           cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        process_record=read(PROJECT/'results/blagent/recovery-20261118/server_process.json')
        pid=process_record['pid'];write_json(output/'owned_server.json',process_record)
        deadline=time.monotonic()+180
        while True:
            try:
                props=request_json('http://127.0.0.1:8099','/props',timeout=3)
                if props.get('total_slots')!=2:raise ValueError('Wrong server slot count')
                break
            except (OSError,ValueError):
                if time.monotonic()>deadline:raise RuntimeError('Inference server failed to become ready')
                time.sleep(5)
        regression=output/f'regression{regression_count}';full=output/f'full{full_count}'
        update(status='running_regression',regression_run=str(regression),server_pid=pid)
        command=[sys.executable,'-u',str(PROJECT/'scripts/experiments/run_blagent_parallel.py')]
        def launch(args,logname):
            with (output/logname).open('a') as log:
                return subprocess.run(command+args,cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=visible_gpus),
                                      stdout=log,stderr=subprocess.STDOUT).returncode
        code=launch(['--config',regression_config,'--output',str(regression)],'regression.log')
        if code!=0:
            update(status='regression_needs_attention',exit_code=code)
            return
        update(status=f'running_full{full_count}',full_run=str(full))
        code=launch(['--config',full_config,'--output',str(full),'--gate',str(regression)],f'full{full_count}.log')
        report=read(full/'automation_status.json')
        update(status=report['status'],exit_code=code,full_run=str(full),finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        update(status='queue_failed',error=repr(error))
        raise
    finally:
        if pid is not None:stop_owned_server(pid)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minimum-free-mib',type=int,default=18000)
    p.add_argument('--visible-gpus',default='0,1');p.add_argument('--gpu-layers',type=int,default=99)
    p.add_argument('--regression-config',default='configs/experiments/blagent_recovery19_v2.json')
    p.add_argument('--full-config',default='configs/experiments/blagent500_v2.json')
    args=p.parse_args();run((PROJECT/args.output).resolve(),args.minimum_free_mib,args.visible_gpus,args.gpu_layers,
                          args.regression_config,args.full_config)
