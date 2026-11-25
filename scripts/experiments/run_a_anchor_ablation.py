"""Frozen follow-up: isolate identifiers, paths, and report-tail access in A+."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from bug_localization.lightweight_hypothesis import IGNORE, WORD, anchors, clean_report, definitions, normalize
from bug_localization.semantic_improvements import rank
from run_b_lightweight import IDENTITY, digest, field, metrics, read, save, summarize

DATA=None; SYMBOLS={}; SOURCE=None; FROZEN=None


def initialize(source,choice):
    global DATA,SOURCE,FROZEN
    from datasets import load_from_disk
    SOURCE=Path(source);FROZEN=choice
    DATA=load_from_disk(str(ROOT/'datasets/swe_bench_formatted'))['test']


def one(item):
    meta,checkpoint=item; p=SOURCE/checkpoint['result_file']
    rows=field(p,'ranking');aggs=field(p,'aggregations');info=field(p,'query_info')
    body=DATA[meta['dataset_row']]['body']
    assert digest(json.dumps({'body':body},sort_keys=True).encode())==meta['query_sha256']
    # This follow-up is preregistered for the actual frozen combination, not tuned here.
    assert FROZEN=='combo_max_top3_w0.25_d50_w0.5'
    m3={r['file_path']:r['score'] for r in aggs['top3_mean']['ranking']}
    scores={r['file_path']:.75*r['best_chunk_similarity']+.25*m3[r['file_path']] for r in rows}
    ordered=rank(scores,limit=len(rows));pool=ordered[:50]
    normalized=normalize([{'file_path':p,'score':scores[p]} for p in ordered])
    inventory=read(SOURCE/checkpoint['assets']/'candidates.json')
    hashes={r['file_path']:r['sha256'] for r in inventory['files']}
    symbols={}
    for path in pool:
        sha=hashes[path]
        if sha not in SYMBOLS:
            raw=(ROOT/'repos'/meta['repository']/meta['before_fix_sha']/'source'/path).read_bytes()
            assert digest(raw)==sha
            SYMBOLS[sha]=definitions(raw.decode('utf-8',errors='replace'))
        symbols[path]=SYMBOLS[sha]
    full,matches=anchors(body,symbols)
    query=info['embedding_text'].split('\nQuery: ',1)[1]
    truncated,_=anchors(query,symbols)
    words={w for w in WORD.findall(clean_report(body)) if len(w)>=3 and w not in IGNORE}
    df=Counter(w for ss in symbols.values() for w in ss if w in words)
    weights={w:math.log((len(symbols)+1)/(n+1)) for w,n in df.items()}
    denominator=sum(weights.values())
    identifiers={p:sum(weights[w] for w in symbols[p]&weights.keys())/denominator if denominator else 0. for p in pool}
    pathonly={p:float(matches[p]['explicit_path_or_module']) for p in pool}
    rankings={'A':[r['file_path'] for r in rows[:50]],'aggregation_only':pool}
    for name,evidence in [('full',full),('encoded_report_only',truncated),('identifiers_only',identifiers),('paths_only',pathonly)]:
        rankings[name]=rank({p:normalized[p]+.5*evidence[p] for p in pool},{p:i for i,p in enumerate(ordered)})
    gold=set(checkpoint['ground_truth'])
    return {**meta,'status':'ok','query_truncated':info['truncated'],'source_result_sha256':checkpoint['result_sha256'],
            'methods':{n:{'ranking':rs,'metrics':metrics(rs,gold)} for n,rs in rankings.items()}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run;out=run/'anchor_ablation';out.mkdir(exist_ok=False)
    source=ROOT/read(run/'protocol.json')['config']['source_run'];frozen=read(run/'frozen_choice.json')
    assert frozen['selected']=='combo_max_top3_w0.25_d50_w0.5'
    selection=read(source/'selection.json');checkpoints=read(source/'S0/samples.json')
    save(out/'protocol.json',{'created_unix':time.time(),'frozen_parent_choice':frozen,
        'description':'No parameter search: remove path boost, remove identifiers, restrict evidence to actually encoded report, and remove all grounding. Same frozen combo50 pool and lambda0.5.',
        'hashes':{str(p):digest(p.read_bytes()) for p in [Path(__file__),run/'frozen_choice.json',source/'S0/samples.json',
                  ROOT/'src/bug_localization/lightweight_hypothesis.py',ROOT/'src/bug_localization/semantic_improvements.py']},
        'source_policy':'Parent study hashes every original result. This follow-up reuses those checkpoints, reads required fields and verifies original body and touched source hashes; no model calls.',
        'selection_policy':'Declared after development freeze, before viewing validation metrics. Diagnostic ablations only; no replacement parameter selected.'})
    started=time.monotonic();cases=[]
    with ProcessPoolExecutor(max_workers=2,initializer=initialize,initargs=(str(source),frozen['selected'])) as executor:
        for step,record in enumerate(executor.map(one,zip(selection['samples'],checkpoints)),1):
            save(out/'cases'/f'{record["position"]:04d}.json',record);cases.append(record)
            if step%100==0:print(f'anchor ablation {step}/2294 elapsed={time.monotonic()-started:.1f}s',flush=True)
    dev=set(selection['development_rows'])
    save(out/'summary.json',{'all2294':summarize(cases),
                            'development500':summarize([r for r in cases if r['dataset_row'] in dev]),
                            'remaining1794':summarize([r for r in cases if r['dataset_row'] not in dev])})
    for p,sha in read(out/'protocol.json')['hashes'].items():assert digest(Path(p).read_bytes())==sha
    save(out/'status.json',{'status':'completed','samples':len(cases),'seconds':time.monotonic()-started,
                           'new_model_calls':0,'pending_check':'Compare full rankings to parent frozen combination after parent completion.'})
    print('Completed '+str(out),flush=True)


if __name__=='__main__':main()
