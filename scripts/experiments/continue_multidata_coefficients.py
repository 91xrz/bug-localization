"""Continue the authorized long experiment through scoring and final analysis.

No notifications are sent. Progress/errors are saved beside the experiment.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/experiments')]
from run_multidata_coefficients import load,read,save
from bug_localization.snapshot import sha256_file


def alive(pid):
    try:
        os.kill(pid,0)
        # A finished child can remain as a zombie until its tool session is polled.
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except (ProcessLookupError,FileNotFoundError):
        return False


def run(args):
    out=args.run.resolve();protocol,plan=load(out)
    files=[Path(__file__),ROOT/'scripts/analysis/evaluate_multidata_coefficients.py',
           ROOT/'src/bug_localization/coefficient_search.py',ROOT/'scripts/datasets/prepare_ghrb_sources.py']
    frozen={str(p):sha256_file(p) for p in files}
    save(out/'controller_protocol.json',{'files_sha256':frozen,'gpu':args.gpu,
        'prerequisite_pids':{'generation':args.generation_pid,'SWE_inputs':args.swe_pid,'sources':args.sources_pid},
        'started_unix':time.time(),'automatic_stages':['retrieve-ghrb','rerank-GHRB','rerank-SWE','development-search-and-frozen-validation']})

    def state(stage,**kwargs):
        save(out/'controller_status.json',{'status':'running','stage':stage,'updated_unix':time.time(),**kwargs})

    def wait(name,ready,pid):
        while not ready():
            state('waiting_for_'+name,pid=pid)
            if not alive(pid):
                raise RuntimeError(name+' worker ended without complete artifacts; inspect its log')
            time.sleep(20)

    def complete(dataset,folder):
        return all((out/dataset/folder/f"{r['position']:04d}.sha256").exists() for r in plan[dataset])

    def verify():
        load(out)
        if any(sha256_file(Path(p))!=checksum for p,checksum in frozen.items()):
            raise ValueError('Controller/evaluation implementation changed during run')

    def gpu_ready():
        while True:
            lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            memory={int(row.split(',')[0]):int(row.split(',')[1]) for row in lines.splitlines()}
            if memory[args.gpu]>=8000:return
            state('waiting_for_gpu_memory',gpu=args.gpu,free_mib=memory[args.gpu],required_mib=8000)
            time.sleep(30)

    def stage(name,command,gpu=True):
        verify()
        if gpu:gpu_ready()
        state(name)
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(args.gpu)
        with (out/(name+'.log')).open('a') as log:
            subprocess.run([str(ROOT/'.venv/bin/python'),'-u',*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

    wait('generation',lambda:complete('GHRB','generation'),args.generation_pid)
    sources=ROOT/'datasets/GHRB_sources/acquisition.json'
    wait('sources',lambda:sources.exists() and read(sources)['status']=='passed',args.sources_pid)
    if not complete('GHRB','inputs'):
        stage('retrieve_GHRB',['scripts/experiments/run_multidata_coefficients.py','--output',str(out),'--stage','retrieve-ghrb'])
    if not complete('GHRB','scores'):
        stage('rerank_GHRB',['scripts/experiments/run_multidata_coefficients.py','--output',str(out),'--stage','rerank','--dataset','GHRB'])
    wait('SWE_inputs',lambda:complete('SWE','inputs'),args.swe_pid)
    if not complete('SWE','scores'):
        stage('rerank_SWE',['scripts/experiments/run_multidata_coefficients.py','--output',str(out),'--stage','rerank','--dataset','SWE'])
    stage('evaluate',['scripts/analysis/evaluate_multidata_coefficients.py','--run',str(out)],gpu=False)
    verify()
    if read(out/'analysis/verification.json')['status']!='passed':
        raise ValueError('Final analysis did not pass verification')
    save(out/'controller_status.json',{'status':'completed','updated_unix':time.time(),
        'report':str(out/'analysis/report.md'),'verification_sha256':sha256_file(out/'analysis/verification.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--generation-pid',type=int,required=True);p.add_argument('--swe-pid',type=int,required=True)
    p.add_argument('--sources-pid',type=int,required=True);p.add_argument('--gpu',type=int,default=1)
    args=p.parse_args()
    with (args.run/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(args)
        except Exception as error:
            save(args.run/'controller_status.json',{'status':'failed','updated_unix':time.time(),
                'error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc()})
            raise
