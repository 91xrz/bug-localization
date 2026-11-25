"""Supplement: hide the incumbent and require support across two opposite presentations."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pilot_evidence_challenge import (OUT,PROJECT,ROOT,SAFE,ServerGenerator,anchored,freeze,get_cohort,
                                      obj,read,request,sha256_file,write_json)

PROMPT=SAFE+'''Compare A and B without assuming either is correct. Sketch the smallest behavior change
in each file that could address this report, and say which repair better explains the actual
versus expected result. Do not treat filenames or test status as proof. Quote one short, exact,
single-line code fragment from each file (not metadata or comments). Favor neither file when the
provided code cannot distinguish the two repairs. Return A, B, or unclear.'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort',choices=['replay90','transfer45'])
    name=parser.parse_args().cohort
    rows,archive,_=get_cohort(name)
    cfg=dict(read(OUT/'protocol.json')['identity']['settings'],inference_endpoint='http://127.0.0.1:8113',
             server_manifest='models/BLAgent-Qwen3-8B-BF16/manifest.json',max_new_tokens=512)
    g=ServerGenerator(PROJECT,cfg)
    freeze(ROOT/'blind_protocol.json',dict(script=sha256_file(Path(__file__)),prompt=PROMPT,identity=g.identity,
           primary_protocol_sha256=sha256_file(ROOT/'protocol.json'),
           rule='blind evidence vote with exact citations, alone and intersected with opposite-order direct vote'))
    def run(s):
        key=f'{s["position"]:04d}'
        dest=ROOT/name/'cases'/key
        t=read(dest/'decisions.json')
        record=dict(promote=False,consensus=False,status='not_triggered')
        if t['promote']['unconditional']:
            order=t['stages']['direct']['order'][::-1]
            inp=read(archive/'cases'/key/'inference_input.json')
            user='REPORT:\n'+inp['query']+'\n\n'
            for label,path in zip('AB',order):
                user+=f'FILE {label}: {path}\n'+'\n\n'.join(c['text'] for c in inp['evidence'][path])+'\n\n'
            p,status=request(copy.copy(g),dest,'blind',PROMPT,user,obj(quote_A='',quote_B='',repair_A='',
                             repair_B='',winner=['A','B','unclear']))
            anchors={l:anchored(p.get('quote_'+l,''),inp['evidence'][f]) for l,f in zip('AB',order)}
            move=(status=='ok' and p['winner'] in ('A','B') and order['AB'.index(p['winner'])]==t['challenger']
                  and all(anchors.values()) and bool(p['repair_A'].strip()) and bool(p['repair_B'].strip()))
            record=dict(order=order,answer=p,anchors=anchors,status=status,promote=move,
                        consensus=move and t['promote']['direct'])
        write_json(dest/'blind_decision.json',record)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run,rows))
    print(f'blind {name} complete',flush=True)


if __name__=='__main__':
    main()
