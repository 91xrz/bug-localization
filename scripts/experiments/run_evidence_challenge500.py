"""Run both frozen review policies on the original 500 in a detached session."""
import copy
import fcntl
import os
import signal
import socket
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean
from urllib.request import urlopen

import historical180_review as h
from pilot_c3_joint import infer,inference_input

OUT=h.ROOT.parent/'crossproject500_v1'
SEL=h.PROJECT/'configs/experiments/selections/crossproject_500_seed20261111.json'
PORT=8113
NAMES=('C3','BLAgent','original','loose','strict')
METRICS=('average_precision_at_k','reciprocal_rank_at_k','top_1','top_5','top_10','recall_at_k')


def progress(stage,**info):
    rec=dict(stage=stage,updated=time.strftime('%Y-%m-%d %H:%M:%S %z'),**info)
    h.write_json(OUT/'status.json',rec)
    print(rec,flush=True)


@contextmanager
def server(stage,slots,ctx):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',PORT))
    cmd=['.runtime/llama/rrl-cuda-build/bin/llama-server','-m','models/BLAgent-Qwen3-8B-BF16/Qwen3-8B-BF16.gguf',
         '--alias','blagent-qwen3-8b-bf16','--host','127.0.0.1','--port',str(PORT),'-c',str(ctx*slots),
         '-np',str(slots),'-ngl','99','-b','512','-ub','128','-fa','off','--cache-type-k','f16',
         '--cache-type-v','f16','--cont-batching','--no-context-shift','--cache-ram','256','-t','6']
    with (OUT/(stage+'-server.log')).open('a') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=h.PROJECT,
                              env=dict(os.environ,CUDA_VISIBLE_DEVICES='1'))
        try:
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError(f'{stage} server exited: {proc.returncode}')
                try:
                    with urlopen(f'http://127.0.0.1:{PORT}/health',timeout=2) as reply:
                        if reply.status==200: break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError('server startup timed out')
            yield
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def generator(identity):
    cfg=dict(identity['settings'],inference_endpoint=f'http://127.0.0.1:{PORT}',
             server_manifest='models/BLAgent-Qwen3-8B-BF16/manifest.json')
    g=h.ServerGenerator(h.PROJECT,cfg)
    assert g.identity==identity
    return g


def source(s):
    archive=h.locate(s)
    d=(archive or OUT/'joint')/'cases'/f'{s["position"]:04d}'
    if (d/'record.json').exists():
        rec=h.read(d/'record.json')
        assert all(s[k]==rec[k] for k in h.FIELDS)
        if rec['status']=='ok':
            assert h.sha256_file(d/'result.json')==rec['result_sha256']
        return d


def joint(s,c,g):
    d=OUT/'joint/cases'/f'{s["position"]:04d}'
    d.mkdir(parents=True,exist_ok=True)
    path=h.C3/c['result_file']
    assert h.sha256_file(path)==c['result_sha256']
    inp=h.read(h.BL/'cases'/f'{s["dataset_row"]+1:04d}'/'inputs.json')
    q,head,ev,tail=inference_input(h.read(path),inp,10)
    model=copy.copy(g)
    model.start_case(d)
    try:
        rank,reason=infer(model,q,head,ev,d)
    except ValueError as exc:
        if not any(msg in str(exc) for msg in ('exact candidate permutation','context overflow','exhausted the output budget')):
            raise
        rec=dict(s,status='failed',error=str(exc))
    else:
        h.write_json(d/'result.json',dict(ranking=rank+tail,reasoning=reason,model_calls=model.calls))
        rec=dict(s,status='ok',result_sha256=h.sha256_file(d/'result.json'))
    h.write_json(d/'record.json',rec)


def review_case(s,g,judge):
    key=f'{s["position"]:04d}'
    src=source(s)
    inp=h.read(src/'inference_input.json')
    ok=h.read(src/'record.json')['status']=='ok'
    prior=h.read(src/'result.json')['ranking'][:10] if ok else inp['candidates']
    dest=OUT/'cases'/key
    dest.mkdir(parents=True,exist_ok=True)
    old180=h.RUN/'cases'/key/'review.json'
    old=h.prior_review(key)
    if old180.exists():
        t=h.read(old180)
        assert t['input_sha256']==h.sha256_file(src/'inference_input.json') and t['prior']==prior
        ch,line=t['challenger'],t['line']
    elif old:
        t=h.read(old/'decisions.json')
        assert t['evidence_sha256']==h.sha256_file(src/'inference_input.json') and t['prior']==prior
        ch,line=t['challenger'],h.read(old/'line_decision.json')
    else:
        ch,_=h.proposal(s,g,src,dest,prior) if ok else (prior[0],None)
        line=h.review(s,judge,inp,prior[0],ch,dest) if ch!=prior[0] else dict(promote=False,strict=False,status='not_triggered')
    assert len(prior)==10 and set(prior)==set(inp['candidates']) and ch in prior
    assert not line['strict'] or line['promote']
    h.write_json(dest/'review.json',dict(prior=prior,challenger=ch,line=line,old=bool(old or old180.exists()),
                 status='ok' if ok else 'joint_fallback',input_sha256=h.sha256_file(src/'inference_input.json')))


def evaluation(samples,cs,bs):
    rows=[]
    reference={r['position']:r for r in h.read(h.RUN/'paired_cases.json')}
    for s in samples:
        i=s['position']
        c,b=cs[i],bs[s['dataset_row']]
        m={name:dict.fromkeys(METRICS,0) for name in NAMES}
        if b['status']=='ok':
            assert h.sha256_file(h.BL/b['result_file'])==b['result_sha256']
            m['BLAgent']=b['metrics']
        extra=dict(old=False,status='c3_failed',proposed=False,loose=False,strict=False)
        if c['status']=='ok':
            assert h.sha256_file(h.C3/c['result_file'])==c['result_sha256']
            saved=h.read(h.C3/c['result_file'])
            gold=saved['ground_truth']
            pool=h.paths_from_saved(saved)
            t=h.read(OUT/'cases'/f'{i:04d}'/'review.json')
            base=t['prior']+pool[10:]
            assert set(base[:10])==set(pool[:10]) and len(set(base))==len(base)
            ch=t['challenger']
            extra=dict(old=t['old'],status=t['status'],proposed=ch!=base[0],loose=t['line']['promote'],strict=t['line']['strict'])
            for k,rank in (('C3',pool),('original',base),('loose',h.promote(base,ch) if extra['loose'] else base),
                           ('strict',h.promote(base,ch) if extra['strict'] else base)):
                m[k]=h.ranking_metrics(rank,gold,10)
                assert m[k]['recall_at_k']==c['metrics']['recall_at_k']
        if i in reference:
            assert all(m[n][k]==reference[i]['metrics'][n][k] for n in NAMES for k in METRICS)
        rows.append(dict(s,**extra,bl_status=b['status'],metrics=m))
    report={}
    groups={'all500':rows,'old180':rows[:180],'added320':rows[180:],
            'not_previously_reviewed':[r for r in rows if not r['old']]}
    for group,rs in groups.items():
        report[group]=dict(n=len(rs),proposed=sum(r['proposed'] for r in rs),
            changes={k:sum(r[k] for r in rs) for k in ('loose','strict')},
            joint_fallback=sum(r['status']=='joint_fallback' for r in rs),
            c3_failed=sum(r['status']=='c3_failed' for r in rs),bl_failed=sum(r['bl_status']!='ok' for r in rs),
            metrics={n:{k:fmean(r['metrics'][n][k] for r in rs) for k in METRICS} for n in NAMES},
            paired={n:h.paired(rs,n,'original') for n in ('strict','loose')},
            strict_vs_BLAgent=h.paired(rs,'strict','BLAgent'))
    h.write_json(OUT/'paired_cases.json',rows)
    h.write_json(OUT/'summary.json',report)


def main():
    samples=h.read(SEL)['samples']
    assert len(samples)==len({(s['repository'],s['issue_id']) for s in samples})==500 and samples[:180]==h.cases()
    cs={r['position']:r for r in h.read(h.C3/'samples.json')}
    bs={r['dataset_row']:r for r in h.read(h.BL/'samples.json')}
    for s in samples:
        assert all(s[k]==cs[s['position']][k] for k in (*h.FIELDS,'query_sha256','labels_sha256'))
        assert all(s[k]==bs[s['dataset_row']][k] for k in h.FIELDS[1:])
    h.freeze(OUT/'protocol.json',dict(selection=h.sha256_file(SEL),scripts={p:h.sha256_file(Path(__file__).with_name(p))
        for p in ('run_evidence_challenge500.py','historical180_review.py','pilot_c3_joint.py','pilot_line_challenge.py')},
        line_protocol=h.sha256_file(h.ROOT/'line_protocol.json'),rule='frozen loose and strict; C3 and BL failures zero independently'))
    valid=[s for s in samples if cs[s['position']]['status']=='ok']
    missing=[s for s in valid if source(s) is None]
    progress('joint',completed=0,to_generate=len(missing),c3_valid=len(valid))
    if missing:
        with server('joint',1,32768):
            g=generator(h.read(h.RUN/'joint_protocol.json')['identity'])
            for i,s in enumerate(missing,1):
                joint(s,cs[s['position']],g)
                progress('joint',completed=i,to_generate=len(missing),position=s['position'])
    progress('review',completed=0,total=len(valid))
    with server('review',2,8192):
        g=generator(h.read(h.OUT/'protocol.json')['identity'])
        judge=generator(h.read(h.ROOT/'line_protocol.json')['identity'])
        with ThreadPoolExecutor(max_workers=2) as pool:
            for i,_ in enumerate(pool.map(lambda s:review_case(s,g,judge),valid),1):
                if i%10==0 or i==len(valid): progress('review',completed=i,total=len(valid))
    progress('evaluation')
    evaluation(samples,cs,bs)
    progress('complete',reports=len(samples))


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: main()
        except BaseException:
            progress('failed',error=traceback.format_exc())
            raise
