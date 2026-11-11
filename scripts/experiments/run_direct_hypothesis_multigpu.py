"""Generate disjoint report caches on two GPUs, then run the paired retrieval."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bug_localization.batch import write_json
from bug_localization.embeddings import digest
from run_direct_hypothesis import Generator, setup


def shard(report):
    return int(digest(report),16)%2


def worker(index, logs):
    cfg,_,_,selected,_=setup()
    generator=Generator(cfg)
    assigned=[sample for sample in selected if shard(sample['report']['body'])==index]
    write_json(logs/f'worker{index}.status.json',{'status':'running','selected':len(assigned),'processed':0})
    for i,sample in enumerate(assigned,1):
        generator.generate(sample['report']['body'])
        write_json(logs/f'worker{index}.status.json',{'status':'running','selected':len(assigned),'processed':i})
        print(f'GPU {index}: {i}/{len(assigned)}',flush=True)
    write_json(logs/f'worker{index}.status.json',{'status':'completed','selected':len(assigned),'processed':len(assigned)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,choices=(0,1));parser.add_argument('--logs',type=Path);args=parser.parse_args()
    if args.worker is not None:
        if args.logs is None: parser.error('--logs is required for worker')
        worker(args.worker,args.logs);return
    logs=ROOT/'results/direct_report_hypothesis/automation'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    logs.mkdir(parents=True)
    write_json(logs/'status.json',{'status':'running','phase':'parallel_generation','pid':os.getpid()})
    print('Automation logs:',logs,flush=True)
    def run_worker(index):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(index)}
        with (logs/f'worker{index}.log').open('w') as out:
            subprocess.run([sys.executable,'-B','-u',str(Path(__file__)), '--worker',str(index),'--logs',str(logs)],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run_worker,i) for i in (0,1)]
            for future in futures: future.result()
        write_json(logs/'status.json',{'status':'running','phase':'retrieval','pid':os.getpid()})
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'}
        with (logs/'retrieval.log').open('w') as out:
            subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/experiments/run_direct_hypothesis.py'),'--run'],cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
        lines=(logs/'retrieval.log').read_text().splitlines()
        result=next(line[len('Results: '):] for line in lines if line.startswith('Results: '))
        summary={'status':'completed','result_directory':result,'retrieval_log':str(logs/'retrieval.log')}
        write_json(logs/'status.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
        print((logs/'retrieval.log').read_text().split('Retrieval 180/180')[-1],flush=True)
    except BaseException as error:
        write_json(logs/'status.json',{'status':'failed','error':str(error)})
        raise

if __name__=='__main__':main()
