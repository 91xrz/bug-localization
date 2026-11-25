"""Development-frozen recall-oriented A grounding control; no parameter search on validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from bug_localization.lightweight_hypothesis import definitions
from bug_localization.semantic_improvements import grounded_scores,rank
from run_b_lightweight import digest,field,metrics,read,save,summarize

DATA=None;SOURCE=None;SYMBOLS={};METHOD=None


def initialize(source,method):
    global DATA,SOURCE,METHOD
    from datasets import load_from_disk
    DATA=load_from_disk(str(ROOT/'datasets/swe_bench_formatted'))['test'];SOURCE=Path(source);METHOD=method


def one(pair):
    meta,checkpoint=pair;body=DATA[meta['dataset_row']]['body']
    assert digest(json.dumps({'body':body},sort_keys=True).encode())==meta['query_sha256']
    rows=field(SOURCE/checkpoint['result_file'],'ranking')
    depth=int(METHOD.split('_d')[1].split('_')[0]);weight=float(METHOD.split('_w')[1])
    candidates=read(SOURCE/checkpoint['assets']/'candidates.json')
    sha={r['file_path']:r['sha256'] for r in candidates['files']};symbols={}
    for row in rows[:depth]:
        path=row['file_path'];key=sha[path]
        if key not in SYMBOLS:
            raw=(ROOT/'repos'/meta['repository']/meta['before_fix_sha']/'source'/path).read_bytes()
            assert digest(raw)==key
            SYMBOLS[key]=definitions(raw.decode('utf-8',errors='replace'))
        symbols[path]=SYMBOLS[key]
    scores,_,_=grounded_scores(rows,body,symbols,weight,depth)
    paths=rank(scores,{r['file_path']:r['rank'] for r in rows})
    ranks={'A':[r['file_path'] for r in rows[:50]],METHOD:paths}
    return {**meta,'status':'ok','source_result_sha256':checkpoint['result_sha256'],
            'methods':{n:{'ranking':rs,'metrics':metrics(rs,set(checkpoint['ground_truth']))} for n,rs in ranks.items()}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run;out=run/'recall_control';out.mkdir(exist_ok=False)
    summary=read(run/'development500_summary.json');base=summary['A']
    eligible=[n for n,m in summary.items() if n.startswith('anchor_') and m['MAP@10']>=base['MAP@10']+.005
              and all(m[k]>=base[k] for k in ('MRR@10','Hit@10','Hit@50'))]
    method=min(eligible,key=lambda n:(-summary[n]['MeanRecall@50'],-summary[n]['MAP@10'],n))
    source=ROOT/read(run/'protocol.json')['config']['source_run'];selection=read(source/'selection.json')
    checkpoints=read(source/'S0/samples.json')
    save(out/'protocol.json',{'created_unix':time.time(),'method':method,
         'selection_rule':'Among original development anchor grid with MAP gain>=.005 and MRR/Hit10/Hit50 no worse, maximize Recall50; ties by MAP then name.',
         'scope':'Secondary recall objective, frozen before inspecting parent validation metrics. Not a replacement of the primary MAP-selected method.',
         'development_metrics':summary[method],
         'hashes':{str(p):digest(p.read_bytes()) for p in [Path(__file__),run/'development500_summary.json',
                   source/'S0/samples.json',ROOT/'src/bug_localization/lightweight_hypothesis.py',ROOT/'src/bug_localization/semantic_improvements.py']},
         'input_policy':'Read required fields from parent-hashed S0 result files; verify body and touched source hashes. No B, models, or future source.'})
    print('Frozen recall control: '+method,flush=True)
    start=time.monotonic();cases=[]
    with ProcessPoolExecutor(max_workers=2,initializer=initialize,initargs=(str(source),method)) as executor:
        for step,row in enumerate(executor.map(one,zip(selection['samples'],checkpoints)),1):
            save(out/'cases'/f'{row["position"]:04d}.json',row);cases.append(row)
            if step%100==0:print(f'recall control {step}/2294 elapsed={time.monotonic()-start:.1f}s',flush=True)
    dev=set(selection['development_rows'])
    save(out/'summary.json',{'all2294':summarize(cases),'development500':summarize([r for r in cases if r['dataset_row'] in dev]),
                            'remaining1794':summarize([r for r in cases if r['dataset_row'] not in dev])})
    for p,sha in read(out/'protocol.json')['hashes'].items():assert digest(Path(p).read_bytes())==sha
    save(out/'status.json',{'status':'completed','samples':len(cases),'seconds':time.monotonic()-start,'model_calls':0})


if __name__=='__main__':main()
