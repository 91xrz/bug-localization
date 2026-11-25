"""Frozen incumbent/challenger review on the historical cross-project manifest."""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from historical180_joint import RUN,SEL,FIELDS,cases,locate
from pilot_evidence_challenge import DIRECT,ROOT,OUT,request,obj,promote
from pilot_line_challenge import PROMPT,index
from pilot_perspective_rank import (BL,C3,COMMON,LENS,PROJECT,ServerGenerator,compact,fuse,paired,ranking_metrics,
                                    paths_from_saved,read,sha256_file,write_json)
from prepare_challenge_transfer import TRANSFER,freeze


def prior_review(key):
    for name in ('replay90','transfer45'):
        folder=ROOT/name/'cases'/key
        if (folder/'line_decision.json').exists():
            return folder


def proposal(s,gen,src,dest,prior):
    key=src.name
    for archive in (TRANSFER,OUT.parent/'confirmation90_v1',OUT):
        f=archive/'cases'/key/'rankings.json'
        if f.exists():
            t=read(f)
            assert t['input_sha256']==sha256_file(src/'inference_input.json')
            assert t['heads']['dual']==fuse(t['heads']['owner'],t['heads']['behavior'],prior)
            return t['heads']['dual'][0],str(f)
    inp=read(src/'inference_input.json')
    head=inp['candidates']
    order=sorted(head)
    random.Random(s['query_sha256']).shuffle(order)
    owner_first=int(s['query_sha256'][:8],16)%2==0
    report=compact(gen.tokenizer,inp['query'],1100)
    evidence={p:'\n'.join(compact(gen.tokenizer,c['text'],510//len(chunks)) for c in chunks)
              for p,chunks in inp['evidence'].items()}
    branches={}
    for lens,seq in [('owner',order if owner_first else order[::-1]),('behavior',order[::-1] if owner_first else order)]:
        ids={chr(65+i):p for i,p in enumerate(seq)}
        user='REPORT:\n'+report+'\n\n'+'\n\n'.join(f'[{i}] {p}\n{evidence[p]}' for i,p in ids.items())
        schema={'type':'object','properties':{'ranking':{'type':'array','items':{'enum':list(ids)},
                'minItems':10,'maxItems':10}},'required':['ranking'],'additionalProperties':False}
        value,status=request(gen,dest,lens,COMMON+LENS[lens],user,schema)
        if status!='ok':
            raise ValueError(f'Unexpected proposal failure: {dest} {lens} {status}')
        ranks=value['ranking']
        branches[lens]=[ids[i] for i in ranks] if len(ranks)==len(ids) and set(ranks)==set(ids) else head
    merged=fuse(branches['owner'],branches['behavior'],prior)
    f=dest/'proposal.json'
    write_json(f,dict(branches=branches,dual=merged))
    return merged[0],str(f)


def review(s,gen,inp,inc,ch,dest):
    pair=[inc,ch]
    if int(s['query_sha256'][:8],16)%2:
        pair.reverse()
    user='REPORT:\n'+inp['query']+'\n\n'
    for label,path in zip('AB',pair):
        user+=f'FILE {label}: {path}\n'+'\n\n'.join(c['text'] for c in inp['evidence'][path])+'\n\n'
    d,ds=request(gen,dest,'direct',DIRECT,user,obj(winner=['A','B']))
    direct=ds=='ok' and pair['AB'.index(d['winner'])]==ch
    order=pair[::-1]
    user='REPORT:\n'+inp['query']+'\n\n'
    lookups={}
    for l,path in zip('AB',order):
        body,lookups[l]=index(inp['evidence'][path],l)
        user+=f'FILE {l}: {path}\n'+body+'\n\n'
    roles=['owner','observer','uncertain']
    p,status=request(gen,dest,'line',PROMPT,user,obj(line_A='',line_B='',role_A=roles,role_B=roles,
                     contrast='',winner=['A','B','unclear']))
    cites={l:lookups[l].get(p.get('line_'+l,'')) for l in 'AB'}
    alt='AB'[order.index(ch)]
    old='B' if alt=='A' else 'A'
    move=(status=='ok' and all(cites.values()) and bool(p['contrast'].strip()) and
          p['winner']==alt and p['role_'+alt]=='owner')
    return dict(order=order,answer=p,source_lines=cites,status=status,promote=move,
                strict=move and p['role_'+old]=='observer' and direct)


def run(s,g,judge):
    key=f'{s["position"]:04d}'
    archive=locate(s)
    if archive is None:
        raise ValueError(f'Missing baseline {key}')
    src=archive/'cases'/key
    dest=RUN/'cases'/key
    dest.mkdir(parents=True,exist_ok=True)
    status=read(src/'record.json')['status']
    inp=read(src/'inference_input.json')
    prior=read(src/'result.json')['ranking'][:10] if status=='ok' else inp['candidates']
    assert len(prior)==10 and set(prior)==set(inp['candidates'])
    old=prior_review(key)
    ch,where=proposal(s,g,src,dest,prior) if status=='ok' else (prior[0],None)
    line=dict(promote=False,strict=False,status='not_triggered')
    if ch!=prior[0]:
        if old is None:
            line=review(s,judge,inp,prior[0],ch,dest)
        else:
            tr=read(old/'decisions.json')
            assert tr['evidence_sha256']==sha256_file(src/'inference_input.json') and tr['prior']==prior and tr['challenger']==ch
            # Re-render and verify archived requests against the backend's keyed cache; never generate into the archive.
            for stage in ('direct','line'):
                f=old/stage/'call_001.json'
                if f.exists():
                    sub=dest/stage
                    sub.mkdir(exist_ok=True)
                    target=sub/f.name
                    if not target.exists():
                        target.write_bytes(f.read_bytes())
            line=review(s,judge,inp,prior[0],ch,dest)
    if old is not None:
        assert line==read(old/'line_decision.json')
    t=dict(archive=str(archive),status=status,prior=prior,challenger=ch,line=line,
           input_sha256=sha256_file(src/'inference_input.json'),proposal=where,
           previous_review=str(old) if old else None)
    write_json(dest/'review.json',t)


def evaluation():
    cs={s['position']:s for s in read(C3/'samples.json')}
    bs={s['dataset_row']:s for s in read(BL/'samples.json')}
    rows=[]
    for s in cases():
        key=f'{s["position"]:04d}'
        t=read(RUN/'cases'/key/'review.json')
        c,b=cs[s['position']],bs[s['dataset_row']]
        assert all(c[k]==s[k] for k in FIELDS)
        assert all(b[k]==s[k] for k in FIELDS[1:])
        assert c['status']==b['status']=='ok'
        for folder,record in ((C3,c),(BL,b)):
            assert sha256_file(folder/record['result_file'])==record['result_sha256']
        data=read(C3/c['result_file'])
        pool=paths_from_saved(data)
        base=t['prior']+pool[10:]
        assert set(base[:10])==set(pool[:10]) and len(base)==len(set(base))
        if t['status']=='ok':
            assert base==read(Path(t['archive'])/'cases'/key/'result.json')['ranking']
        ch=t['challenger']
        l=t['line']
        assert not l['strict'] or l['promote']
        heads=dict(original=base,C3=pool,BLAgent=read(BL/b['result_file'])['ranking'],
                   loose=promote(base,ch) if l['promote'] else base,strict=promote(base,ch) if l['strict'] else base)
        metrics={k:ranking_metrics(h,data['ground_truth'],10) for k,h in heads.items()}
        for m in ('loose','strict'):
            assert heads[m][10:]==pool[10:] and metrics[m]['recall_at_k']==metrics['C3']['recall_at_k']
        rows.append(dict(s,old=bool(t['previous_review']),status=t['status'],incumbent=base[0],challenger=ch,
                         proposed=ch!=base[0],loose=l['promote'],strict=l['strict'],metrics=metrics))
    report={}
    for name,group in [('all',rows),('first_review',[r for r in rows if not r['old']]),('reused',[r for r in rows if r['old']])]:
        metrics={k:dict(map=fmean(r['metrics'][k]['average_precision_at_k'] for r in group),
                 mrr=fmean(r['metrics'][k]['reciprocal_rank_at_k'] for r in group),
                 top1=sum(r['metrics'][k]['top_1'] for r in group),top10=sum(r['metrics'][k]['top_10'] for r in group))
                 for k in group[0]['metrics']}
        report[name]=dict(n=len(group),fallback=sum(r['status']!='ok' for r in group),proposed=sum(r['proposed'] for r in group),
                          changes={k:sum(r[k] for r in group) for k in ('loose','strict')},metrics=metrics,
                          comparisons={k:paired(group,k,'original') for k in ('loose','strict')},
                          strict_vs_BLAgent=paired(group,'strict','BLAgent'))
    assert report['all']['n']==180 and report['first_review']['n']==119 and report['reused']['n']==61
    write_json(RUN/'paired_cases.json',rows)
    write_json(RUN/'summary.json',report)
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evaluate-only',action='store_true')
    if not p.parse_args().evaluate_only:
        proto=read(OUT/'protocol.json')
        cfg=dict(proto['identity']['settings'],server_manifest='models/BLAgent-Qwen3-8B-BF16/manifest.json',
                 inference_endpoint='http://127.0.0.1:8113')
        gen=ServerGenerator(PROJECT,cfg)
        assert gen.identity==proto['identity']
        judge=ServerGenerator(PROJECT,dict(cfg,max_new_tokens=512))
        frozen=read(ROOT/'line_protocol.json')
        assert judge.identity==frozen['identity'] and PROMPT==frozen['prompt']
        freeze(RUN/'review_protocol.json',dict(selection=sha256_file(SEL),previous=sha256_file(ROOT/'line_protocol.json'),
               script=sha256_file(Path(__file__)),primary='strict vs original on first-reviewed reports; full 180 and loose secondary'))
        samples=cases()
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i,_ in enumerate(pool.map(lambda s:run(s,gen,judge),samples),1):
                if i%10==0: print(f'review {i}/180',flush=True)
    evaluation()


if __name__=='__main__':
    main()
