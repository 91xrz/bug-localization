"""Blind, code-line-addressed evidence rather than reproducing verbatim strings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pilot_evidence_challenge import (OUT,PROJECT,ROOT,SAFE,ServerGenerator,freeze,get_cohort,
                                      obj,read,request,sha256_file,write_json)

PROMPT=SAFE+'''For each file pick the numbered code line most diagnostic of whether ITS behavior
needs to change, not merely whether it is mentioned. For each, identify its role in the reported
problem: owner (implements the behavior requiring repair), observer (exposes, invokes, or tests
another implementation), or uncertain. A test can itself be an owner. Explain the causal
difference using the two lines. Choose the file with stronger implementation-level repair evidence
or unclear. Do not assume an owner exists. Return line IDs as printed.'''


def index(chunks,label):
    sections=[]
    lookup={}
    for c in chunks:
        lines=[]
        for ln in c['text'].splitlines():
            if ln.strip() and not ln.lstrip().startswith(('Path:','Class:','Signature:','Function:','#','//','/*','*')):
                key=f'{label}{len(lookup)+1}'
                lookup[key]=ln
                ln=f'{key} | {ln}'
            lines.append(ln)
        sections.append('\n'.join(lines))
    return '\n\n'.join(sections),lookup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort',choices=['replay90','transfer45'])
    name=parser.parse_args().cohort
    samples,archive,_=get_cohort(name)
    cfg=dict(read(OUT/'protocol.json')['identity']['settings'],inference_endpoint='http://127.0.0.1:8113',
             server_manifest='models/BLAgent-Qwen3-8B-BF16/manifest.json',max_new_tokens=512)
    gen=ServerGenerator(PROJECT,cfg)
    freeze(ROOT/'line_protocol.json',dict(script=sha256_file(Path(__file__)),prompt=PROMPT,identity=gen.identity,
           rule='two valid source IDs, challenger wins and is owner; strict additionally requires observer incumbent and blind direct vote',
           primary_protocol_sha256=sha256_file(ROOT/'protocol.json')))
    def run(s):
        key=f'{s["position"]:04d}'
        dest=ROOT/name/'cases'/key
        trace=read(dest/'decisions.json')
        record=dict(promote=False,strict=False,status='not_triggered')
        if trace['promote']['unconditional']:
            inp=read(archive/'cases'/key/'inference_input.json')
            order=trace['stages']['direct']['order'][::-1]
            user='REPORT:\n'+inp['query']+'\n\n'
            lookups={}
            for l,path in zip('AB',order):
                body,lookups[l]=index(inp['evidence'][path],l)
                user+=f'FILE {l}: {path}\n'+body+'\n\n'
            roles=['owner','observer','uncertain']
            p,status=request(gen,dest,'line',PROMPT,user,obj(line_A='',line_B='',role_A=roles,role_B=roles,
                            contrast='',winner=['A','B','unclear']))
            cites={l:lookups[l].get(p.get('line_'+l,'')) for l in 'AB'}
            ch='AB'[order.index(trace['challenger'])]
            inc='B' if ch=='A' else 'A'
            move=(status=='ok' and all(cites.values()) and bool(p['contrast'].strip()) and
                  p['winner']==ch and p['role_'+ch]=='owner')
            record=dict(order=order,answer=p,source_lines=cites,status=status,promote=move,
                        strict=move and p['role_'+inc]=='observer' and trace['promote']['direct'])
        write_json(dest/'line_decision.json',record)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run,samples))
    print(f'line {name} complete',flush=True)


if __name__=='__main__':
    main()
