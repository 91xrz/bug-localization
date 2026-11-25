"""Evaluate the frozen new AB pool with the frozen evidence/prior C, fixed500."""
import argparse
from collections import defaultdict
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/retrieval'))
from run_current_baseline import load_current, rank_fixed
from run_conservative_reranking import interval
from run_c_evidence_prior_validation import rank as frozen_refine
from bug_localization.batch import write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.evidence_packets import file_packets
from bug_localization.lightweight_hypothesis import normalize
from bug_localization.reranking import LocalReranker
from bug_localization.snapshot import sha256_file

CONFIG=ROOT/'configs/experiments/ab_structure_newc500.json'
IDENTITY=('dataset_row','repository','issue_id','before_fix_sha')
METHODS=('new_AB','old_pool_Max','new_pool_Max','old_pool_new_C','new_pool_prior',
         'new_pool_new_C','new_pool_new_C_full50','old_pool_new_C_historical')
METRICS={'MAP@10':'average_precision_at_k','MRR@10':'reciprocal_rank_at_k','Top1':'top_1',
         'Top5':'top_5','Top10':'top_10','Recall@10':'recall_at_k'}


def read(path):return json.loads(path.read_bytes())


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);write_json(path,value)


def checked(path,expected):
    if sha256_file(path)!=expected:raise ValueError('Changed source: '+str(path))
    return read(path)


def reuse_packets(query,packets,old_input,old_scores):
    """Cache hits require identical query, chunk IDs, and complete packet text."""
    if query!=old_input['query']:raise ValueError('Joint query differs from cached run')
    old={p['file_path']:p for p in old_input['packets']}
    if set(old)!=set(old_scores):raise ValueError('Incomplete cached packet scores')
    cached,missing={},[]
    for packet in packets:
        path=packet['file_path']
        if path in old:
            if packet!=old[path]:raise ValueError('Same-file joint evidence changed')
            cached[path]=old_scores[path]
        else:missing.append(packet)
    return cached,missing


def max_order(order,chunks):
    ties={p:i for i,p in enumerate(order)}
    return sorted(order,key=lambda p:(-max(chunks[p]),ties[p]))


def refine(base,chunks,prior,joint,packet_weight=.25,prior_weight=4.,head=10):
    values={p:joint[p]['score'] for p in base}
    return frozen_refine({'base':base,'chunks':chunks,'prior':prior},values,packet_weight,prior_weight,head)


def verify(folder):
    p=read(folder/'protocol.json')
    for name,digest in p['files_sha256'].items():
        if sha256_file(Path(name))!=digest:raise ValueError('Frozen dependency changed: '+name)
    return p


def prepare(folder):
    cfg=read(CONFIG);baseline,baseline_sha=load_current()
    if baseline['id']!='ab_structure_v1':raise ValueError('Require frozen AB baseline')
    pair=ROOT/cfg['paired_max_run'];joint=ROOT/cfg['joint_reference'];ab=ROOT/cfg['ab_study']
    ref=read(joint/'protocol.json')
    if (cfg['packet_weight'],cfg['prior_weight'],cfg['head'])!=(ref['packet_weight'],ref['prior_weight'],10):
        raise ValueError('New C parameters changed')
    for name,digest in ref['code_sha256'].items():
        if sha256_file(ROOT/name)!=digest:raise ValueError('New C implementation changed')
    for path in [pair/'verification.json',joint/'verification.json',joint/'independent_audit.json']:
        if read(path)['status']!='passed':raise ValueError('Source not verified')
    old_model=read(joint/'model0.json')
    if read(joint/'model1.json')!=old_model:raise ValueError('Cached scorer identities differ')
    if read(pair/'model_gpu0.json')['identity']!=old_model:raise ValueError('Max/packet model identities differ')
    selection=read(ROOT/cfg['selection'])
    if len(selection['samples'])!=500:raise ValueError('Require fixed500')
    reference_rows={r['dataset_row']:r for r in read(joint/'labels.json')}
    reference_plan={r['dataset_row']:r for r in ref['sample_plan']}
    reference_orders=read(joint/'rankings.json')
    order_map={i:o for i,o in zip(reference_orders['dataset_rows'],reference_orders['orders']['combined_head10'])}
    paired_plan={r['position']:r for r in read(pair/'prepared_index.json')}
    cache_hashes=read(joint/'verification.json')['result_sha256']
    prior_hashes=read(ROOT/cfg['baseline_run']/'source_checkpoints.json')
    files=[CONFIG,Path(__file__),ROOT/cfg['selection'],ROOT/'configs/baselines/current.json',
        ROOT/'configs/baselines/ab_structure_v1.json',pair/'protocol.json',pair/'verification.json',
        pair/'prepared_index.json',joint/'protocol.json',joint/'verification.json',joint/'independent_audit.json',
        joint/'model0.json',joint/'model1.json',joint/'labels.json',joint/'rankings.json',joint/'summary.json',
        ROOT/cfg['baseline_run']/'source_checkpoints.json',
        ROOT/'scripts/retrieval/run_current_baseline.py',ROOT/'src/bug_localization/lightweight_hypothesis.py',
        *[ROOT/name for name in ref['code_sha256']]]
    folder.mkdir(parents=True,exist_ok=False)
    protocol={'config':cfg,'reranker':ref['config'],'model_identity':old_model,
        'baseline_config_sha256':baseline_sha,'started_unix':time.time(),
        'files_sha256':{str(p.resolve()):sha256_file(p) for p in files}}
    save(folder/'protocol.json',protocol);save(folder/'selection.json',selection)
    plans=[];n_reused=n_new=0;input_hashes={}
    for meta in selection['samples']:
        pos=meta['position'];rowid=meta['dataset_row'];old=reference_rows[rowid];plan=reference_plan[rowid]
        record=read(pair/'records'/f'{pos:03d}.json')
        casepath=pair/record['result_file'];case=checked(casepath,record['result_sha256'])
        pp=paired_plan[pos];prepared_path=pair/pp['path'];source=checked(prepared_path,pp['sha256'])
        if any(meta[k]!=case[k] or meta[k]!=old[k] or meta[k]!=plan[k] for k in IDENTITY):raise ValueError('Identity mismatch')
        cache_input_path=joint/plan['input_file'];cache_input=checked(cache_input_path,plan['input_sha256'])
        cache_path=joint/'cases'/f'{rowid:04d}.json'
        cache=checked(cache_path,cache_hashes[str(cache_path.relative_to(joint))])
        if cache['input_sha256']!=plan['input_sha256'] or cache['protocol_sha256']!=sha256_file(joint/'protocol.json'):
            raise ValueError('Joint cache provenance mismatch')
        ledger={str(p.resolve()):sha256_file(p) for p in (casepath,prepared_path,cache_input_path,cache_path)}
        if case['status']!='ok':
            if old['base'] or cache['scores']:raise ValueError('Upstream failure mismatch')
            inp={**meta,'status':'failed','query':'','packets':[]}
            features={**meta,'status':'failed','gold':old['gold'],'error':case['error'],'source_hashes':ledger}
        else:
            if set(case['ground_truth'])!=set(old['gold']):raise ValueError('Gold mismatch')
            prior_path=ab/'inputs'/f"{case['source_position']:04d}.json"
            features_ab=checked(prior_path,prior_hashes[str(case['source_position'])]['prepared_inputs'])
            if rank_fixed(features_ab,baseline)!=case['rankings']['new_AB']:raise ValueError('AB baseline mismatch')
            ledger[str(prior_path.resolve())]=sha256_file(prior_path)
            a,b=normalize(features_ab['a']),normalize(features_ab['b'])
            chunks=defaultdict(list)
            for row in case['scored_chunks']:chunks[row['file_path']].append(row['score'])
            union=set(case['rankings']['old_AB'])|set(case['rankings']['new_AB'])
            if set(chunks)!=union:raise ValueError('Max score inventory mismatch')
            prior={p:.5*a[p]+.5*b[p]+.5*features_ab['anchor'][p] for p in union}
            if any(abs(prior[p]-old['prior'][p])>1e-12 for p in old['base']):raise ValueError('Prior definition changed')
            old_base=max_order(case['rankings']['old_AB'],chunks);new_base=max_order(case['rankings']['new_AB'],chunks)
            if old_base!=case['rankings']['old_ABC'] or new_base!=case['rankings']['new_ABC']:
                raise ValueError('Max stage does not reproduce')
            packets=file_packets(case['scored_chunks'])
            cached,missing=reuse_packets(source['query'],packets,cache_input,{r['file_path']:r for r in cache['scores']})
            if set(cached)!=set(old_base):raise ValueError('Old/new pool cache alignment differs')
            reference=refine(old['base'],old['chunks'],old['prior'],cached)
            if reference!=order_map[rowid]:raise ValueError('Published new C does not reproduce')
            inp={**meta,'status':'ok','query':source['query'],'packets':missing}
            features={**meta,'status':'ok','gold':case['ground_truth'],'old_base':old_base,'new_base':new_base,
                'new_AB':case['rankings']['new_AB'],'chunks':dict(chunks),'prior':prior,'cached_joint':cached,
                'old_pool_new_C_historical':reference,'source_hashes':ledger,
                'packet_inputs_sha256':{p['file_path']:__import__('hashlib').sha256(json.dumps(
                    {'query':source['query'],'packet':p},sort_keys=True).encode()).hexdigest() for p in packets}}
            n_reused+=len(cached);n_new+=len(missing)
        ip=folder/'inputs'/f'{pos:03d}.json';fp=folder/'features'/f'{pos:03d}.json'
        save(ip,inp);save(fp,features)
        plans.append({**meta,'status':inp['status'],'worker':(pos-1)%2,'input_file':str(ip.relative_to(folder)),
            'input_sha256':sha256_file(ip),'feature_file':str(fp.relative_to(folder)),'feature_sha256':sha256_file(fp)})
        if pos%100==0:print('prepared',pos,'/500',flush=True)
    save(folder/'plan.json',plans)
    save(folder/'preparation.json',{'reports':500,'reused_joint_scores':n_reused,'new_joint_scores':n_new,
        'failed_positions':[r['position'] for r in plans if r['status']=='failed'],
        'policy':'All old-pool packets verified byte-identical. Newly added files require model inference.'})
    save(folder/'status.json',{'stage':'prepared'})


def worker(folder,gpu):
    protocol=verify(folder);cfg=protocol['config']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Wrong GPU visibility')
    plans=[p for p in read(folder/'plan.json') if p['worker']==gpu]
    with (folder/f'worker{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model=None
        for step,plan in enumerate(plans,1):
            inp=checked(folder/plan['input_file'],plan['input_sha256'])
            target=folder/'incremental_scores'/f"{plan['position']:03d}.json"
            recordpath=folder/'records'/f"{plan['position']:03d}.json"
            if recordpath.exists():
                checked(target,read(recordpath)['sha256']);continue
            start=time.monotonic();scored=[]
            if inp['packets']:
                if model is None:
                    free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                    memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in free.splitlines()}
                    if memory[gpu]<cfg['minimum_free_mib']:raise ValueError('Insufficient GPU memory')
                    model=LocalReranker(ROOT,protocol['reranker'])
                    if model.identity!=protocol['model_identity']:raise ValueError('Joint model identity changed')
                    save(folder/f'model_gpu{gpu}.json',{'physical_gpu':gpu,'identity':model.identity})
                values=model.score(inp['query'],[r['document'] for r in inp['packets']])
                scored=[{'file_path':p['file_path'],**v} for p,v in zip(inp['packets'],values)]
            result={**{k:plan[k] for k in ('position',*IDENTITY)},'status':plan['status'],
                'input_sha256':plan['input_sha256'],'protocol_sha256':sha256_file(folder/'protocol.json'),
                'physical_gpu':gpu,'scores':scored,'elapsed_seconds':time.monotonic()-start}
            save(target,result);save(recordpath,{'path':str(target.relative_to(folder)),'sha256':sha256_file(target)})
            save(folder/f'progress_gpu{gpu}.json',{'processed':step,'selected':len(plans),'position':plan['position']})
            if step%25==0:print(f'GPU{gpu} {step}/{len(plans)}',flush=True)


def method_metrics(order,gold):
    m=ranking_metrics(order,gold,10);at50=ranking_metrics(order,gold,50)
    return {**m,'hit50':int(at50['retrieved_gold_files']>0),'recall50':at50['recall_at_k']}


def summarize(rows,method):
    n=len(rows)
    return {'reports':n,**{label:sum(r['metrics'][method][key] for r in rows)/n for label,key in METRICS.items()},
            'Hit50':sum(r['metrics'][method]['hit50'] for r in rows)/n,
            'Recall50':sum(r['metrics'][method]['recall50'] for r in rows)/n}


def evaluate(folder):
    protocol=verify(folder);cfg=protocol['config'];plans=read(folder/'plan.json');rows=[];fresh=0;seconds=0.
    for plan in plans:
        feature=checked(folder/plan['feature_file'],plan['feature_sha256']);inp=checked(folder/plan['input_file'],plan['input_sha256'])
        rec=read(folder/'records'/f"{plan['position']:03d}.json");scored=checked(folder/rec['path'],rec['sha256'])
        if scored['input_sha256']!=plan['input_sha256'] or scored['protocol_sha256']!=sha256_file(folder/'protocol.json'):
            raise ValueError('Incremental scoring provenance changed')
        fresh+=len(scored['scores']);seconds+=scored['elapsed_seconds']
        if feature['status']=='failed':orders={m:[] for m in METHODS}
        else:
            added={r['file_path']:r for r in scored['scores']};cached=feature['cached_joint']
            if set(added)!={r['file_path'] for r in inp['packets']} or set(added)&set(cached):raise ValueError('Incremental coverage mismatch')
            joint={**cached,**added}
            if set(joint)!=set(feature['old_base'])|set(feature['new_base']):raise ValueError('Missing union scores')
            if any(not math.isfinite(s['score']) or not 0<s['input_tokens']<=4096 for s in joint.values()):raise ValueError('Invalid scorer output')
            z,p=feature['chunks'],feature['prior'];old,new=feature['old_base'],feature['new_base']
            orders={'new_AB':feature['new_AB'],'old_pool_Max':old,'new_pool_Max':new,
                'old_pool_new_C':refine(old,z,p,joint),'new_pool_prior':refine(new,z,p,joint,0.,4.),
                'new_pool_new_C':refine(new,z,p,joint),'new_pool_new_C_full50':refine(new,z,p,joint,head=50),
                'old_pool_new_C_historical':feature['old_pool_new_C_historical']}
            for name,base in [('old_pool_new_C',old),('new_pool_prior',new),('new_pool_new_C',new)]:
                if set(orders[name][:10])!=set(base[:10]) or orders[name][10:]!=base[10:]:raise ValueError('Head/tail invariant violated')
            # Independent reconstruction of the fixed three-term formula.
            for name,base in [('old_pool_new_C',old),('new_pool_new_C',new),('new_pool_new_C_full50',new)]:
                values={f:.75*max(z[f])+.25*joint[f]['score']+4*p[f] for f in base}
                h=50 if name.endswith('full50') else 10
                expected=sorted(base[:h],key=lambda f:(-values[f],base.index(f)))+base[h:]
                if orders[name]!=expected:raise ValueError('Independent score audit failed')
        if any(len(rank)!=len(set(rank)) or len(rank)>50 for rank in orders.values()):raise ValueError('Invalid output pool')
        row={**{k:plan[k] for k in ('position',*IDENTITY)},'status':feature['status'],'gold':feature['gold'],
            'orders':orders,'metrics':{m:method_metrics(o,feature['gold']) for m,o in orders.items()}}
        save(folder/'cases'/f"{plan['position']:03d}.json",row);rows.append(row)
    if len(rows)!=500:raise ValueError('Incomplete experiment')
    summary={m:summarize(rows,m) for m in METHODS}
    reference=read(ROOT/cfg['joint_reference']/'summary.json')['combined_head10']
    if any(abs(summary['old_pool_new_C_historical'][k]-reference[k])>1e-12 for k in METRICS):
        raise ValueError('Published old-pool new-C metrics not reproduced')
    save(folder/'summary.json',summary)
    save(folder/'project_summary.json',{p:{m:summarize([r for r in rows if r['repository']==p],m) for m in METHODS}
        for p in sorted({r['repository'] for r in rows})})
    stats={};fields=('average_precision_at_k','reciprocal_rank_at_k','top_1','top_10','recall_at_k')
    for left,right in cfg['primary_contrasts']:
        stats[left+'_vs_'+right]={k:interval(rows,[r['metrics'][left][k]-r['metrics'][right][k] for r in rows]) for k in fields}
    tests=[v for d in stats.values() for v in d.values()];previous=0.
    for i,t in enumerate(sorted(tests,key=lambda x:x['paired_randomization_p'])):
        previous=min(1.,max(previous,(len(tests)-i)*t['paired_randomization_p']));t['holm_p_10_tests']=previous
    save(folder/'statistics.json',stats)
    diagnostics={}
    for baseline in ('old_pool_new_C','new_pool_Max','new_AB'):
        diagnostics[baseline]={}
        for key in ('top_1','top_10'):
            diagnostics[baseline][key]={'corrected':sum(r['metrics']['new_pool_new_C'][key]>r['metrics'][baseline][key] for r in rows),
                'broken':sum(r['metrics']['new_pool_new_C'][key]<r['metrics'][baseline][key] for r in rows)}
    changes=[]
    for r in rows:
        d=r['metrics']['new_pool_new_C']['average_precision_at_k']-r['metrics']['old_pool_new_C']['average_precision_at_k']
        if abs(d)>1e-12:changes.append({k:r[k] for k in ('position','repository','issue_id','gold')}|{
            'delta_AP':d,'old_top10':r['orders']['old_pool_new_C'][:10],'new_top10':r['orders']['new_pool_new_C'][:10]})
    diagnostics['changed_AP_cases']=sorted(changes,key=lambda x:-abs(x['delta_AP']))
    save(folder/'diagnostics.json',diagnostics)
    with (folder/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        rr=[{'method':k,**v} for k,v in summary.items()];writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
    prep=read(folder/'preparation.json')
    if fresh!=prep['new_joint_scores']:raise ValueError('Not all new files scored')
    verify(folder)
    save(folder/'verification.json',{'status':'passed','reports':500,'success':sum(r['status']=='ok' for r in rows),
        'failed_positions':[r['position'] for r in rows if r['status']=='failed'],
        'reused_joint_scores':prep['reused_joint_scores'],'fresh_joint_scores':fresh,'sum_incremental_seconds':seconds,
        'model':'Qwen3-Reranker-0.6B','new_generation_or_embedding_calls':0,'new_8B_calls':0,
        'checks':['same fixed500 and exact frozen AB output','byte-identical query/evidence for every reused joint score',
                  'new files completely scored with identical model identity/config','same shared Max/joint scores across pools',
                  'independent formula reconstruction','Top10 membership and tail preservation','historical new-C metrics reproduced',
                  'failure retained as zero; no retuning']})
    save(folder/'status.json',{'stage':'completed','finished_unix':time.time()})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def run(folder):
    verify(folder)
    with (folder/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);procs=[];logs=[]
        save(folder/'status.json',{'stage':'running','started_unix':time.time()})
        try:
            for gpu in (0,1):
                log=(folder/f'gpu{gpu}.log').open('a');logs.append(log)
                procs.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'worker','--output',str(folder),'--gpu',str(gpu)],
                    cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'},
                    stdout=log,stderr=subprocess.STDOUT))
            save(folder/'workers.json',{'pids':[p.pid for p in procs]})
            while any(p.poll() is None for p in procs):
                if any(p.poll() not in (None,0) for p in procs):raise RuntimeError('Worker failed; inspect log')
                time.sleep(5)
            if any(p.returncode for p in procs):raise RuntimeError('Worker failed')
            evaluate(folder)
        finally:
            for p in procs:
                if p.poll() is None:p.terminate()
            for log in logs:log.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','worker','evaluate'))
    parser.add_argument('--output',type=Path,default=ROOT/read(CONFIG)['output']);parser.add_argument('--gpu',type=int,choices=(0,1));args=parser.parse_args()
    try:
        if args.action=='worker':worker(args.output,args.gpu)
        else:{'prepare':prepare,'run':run,'evaluate':evaluate}[args.action](args.output)
    except Exception:
        if args.output.exists():save(args.output/f'error_{args.action}_{args.gpu}.json',{'error':traceback.format_exc(),'time':time.time()})
        raise


if __name__=='__main__':main()
