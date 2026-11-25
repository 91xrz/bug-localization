"""Fresh paired C reranking of old and frozen structural AB pools, fixed500."""
import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/retrieval'))
from run_current_baseline import load_current
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.reranking import LocalReranker
from bug_localization.snapshot import sha256_file
from bug_localization.paired_statistics import inference

CONFIG=ROOT/'configs/experiments/ab_structure_c500.json'
IDENTITY=('dataset_row','repository','issue_id','before_fix_sha','query_sha256','labels_sha256')
METHODS=('old_AB','new_AB','old_ABC','new_ABC','old_ABC_historical')


def read(path):
    return json.loads(path.read_bytes())


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path,value)


def field(path,key):
    with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        marker=('\n  "'+key+'": ').encode(); start=mm.find(marker)
        if start<0:raise ValueError('Missing field '+key)
        start+=len(marker); end=mm.find(b'\n  "',start)
        raw=mm[start:end].rstrip().removesuffix(b',') if end>=0 else mm[start:].rstrip()[:-1].rstrip()
        return json.loads(raw)


def mixed_ids(paths,a,b):
    aa={r['file_path']:r for r in a};bb={r['file_path']:r for r in b}
    result={}
    for p in paths:
        ids=[aa[p]['top_chunk_ids'][0]]
        for cid in bb[p]['top_chunk_ids']:
            if len(ids)==3:break
            if cid not in ids:ids.append(cid)
        result[p]=ids
    return result


def extract_chunks(path,wanted):
    if not wanted:return {}
    found={}
    with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        for match in re.finditer(rb'\n    "chunk_id": "([a-f0-9]{64})"',mm):
            cid=match[1].decode()
            if cid not in wanted:continue
            begin=mm.rfind(b'\n  {',0,match.start())+3
            end=mm.find(b'\n  }',match.end())+4
            if begin<3 or end<4:raise ValueError('Malformed chunk JSON')
            chunk=json.loads(mm[begin:end])
            expected=hashlib.sha256(json.dumps([chunk['route'],chunk['file_path'],chunk['parent_id'],
                chunk['start_byte'],chunk['end_byte'],chunk['embedding_text']],ensure_ascii=False).encode()).hexdigest()
            if expected!=cid:raise ValueError('Chunk content hash mismatch')
            found[cid]=chunk
    if found.keys()!=wanted:raise ValueError('Missing requested chunks')
    return found


def aggregate(order,ids,scores):
    best={p:max((scores[c]['score'] for c in ids[p])) for p in order}
    if not all(math.isfinite(s) for s in best.values()):raise ValueError('Nonfinite C score')
    ranks={p:i for i,p in enumerate(order)}
    ordered=sorted(order,key=lambda p:(-best[p],ranks[p]))
    return [{'rank':i,'file_path':p,'score':best[p],
             'best_chunk_id':max(ids[p],key=lambda c:scores[c]['score'])} for i,p in enumerate(ordered,1)]


def verify(folder):
    protocol=read(folder/'protocol.json')
    for path,digest in protocol['files_sha256'].items():
        if sha256_file(Path(path))!=digest:raise ValueError('Frozen input changed: '+path)
    return protocol


def prepare(folder):
    config=read(CONFIG);baseline,_=load_current()
    if baseline['id']!='ab_structure_v1':raise ValueError('Unexpected current baseline')
    selection=read(ROOT/config['selection']); samples=selection['samples']
    if len(samples)!=500 or [s['position'] for s in samples]!=list(range(1,501)):
        raise ValueError('Require original fixed500 order')
    newroot=ROOT/config['baseline_run'];oldroot=ROOT/config['old_c_run'];recall=ROOT/config['recall_run']
    if read(newroot/'verification.json')['status']!='passed':raise ValueError('Baseline not verified')
    indexes={}
    for key,path in [('new',newroot/'samples.json'),('old',oldroot/'FULL/samples.json'),
                     ('a',recall/'S0/samples.json'),('b',recall/'H0/samples.json')]:
        indexes[key]={r['dataset_row']:r for r in read(path)}
    reranker=read(ROOT/config['reranker_config'])
    reference=read(oldroot/'protocol.json')
    if reranker!=reference['reranker']:raise ValueError('C settings changed')
    files=[CONFIG,Path(__file__),ROOT/config['selection'],ROOT/config['baseline_registry'],
        ROOT/'configs/baselines/ab_structure_v1.json',ROOT/config['reranker_config'],
        oldroot/'protocol.json',oldroot/'model.json',oldroot/'FULL/samples.json',newroot/'samples.json',
        newroot/'verification.json',recall/'S0/samples.json',recall/'H0/samples.json',
        *[ROOT/'src/bug_localization'/f'{n}.py' for n in ['reranking','aggregation','bm25','batch','embeddings','paired_statistics']],
        ROOT/'scripts/retrieval/run_current_baseline.py',ROOT/'src/bug_localization/lightweight_hypothesis.py']
    folder.mkdir(parents=True,exist_ok=False)
    save(folder/'protocol.json',{'config':config,'baseline':baseline['id'],'reranker':reranker,
         'reference_model':reference['reference_model'],
         'files_sha256':{str(p.resolve()):sha256_file(p) for p in files},'started_unix':time.time()})
    save(folder/'selection.json',selection)
    instruction=read(recall/'protocol.json')['semantic']['query_instruction']
    prefix='Instruct: '+instruction+'\nQuery: '
    prepared=[]
    for sample in samples:
        source={k:v[sample['dataset_row']] for k,v in indexes.items()}
        if any(any(s[k]!=sample[k] for k in IDENTITY) for s in source.values()):
            raise ValueError('Unpaired report identity')
        meta={**sample,'source_position':source['a']['position']}
        if source['b']['status']!='ok':
            if source['new']['status']!='failed' or source['old']['status']!='failed':raise ValueError('Failure mismatch')
            item={**meta,'status':'failed','error_type':'DependencyFailure','error':source['b'].get('error','B failed')}
        else:
            oldpath=oldroot/source['old']['result_file'];newpath=newroot/source['new']['result_file']
            for key,p in [('old',oldpath),('new',newpath)]:
                if sha256_file(p)!=source[key]['result_sha256']:raise ValueError('Saved result changed')
            old,new=read(oldpath),read(newpath)
            apath=recall/source['a']['result_file'];bpath=recall/source['b']['result_file']
            for key,p in [('a',apath),('b',bpath)]:
                if sha256_file(p)!=source[key]['result_sha256']:raise ValueError('Retrieval checksum changed')
            a,b=field(apath,'ranking'),field(bpath,'ranking')
            query=field(apath,'query_info')['embedding_text']
            if not query.startswith(prefix) or query[len(prefix):]!=old['query']:raise ValueError('Query differs from old C')
            if new['ground_truth']!=old['ground_truth']:raise ValueError('Labels mismatch')
            oldorder=old['ab_ranking'];neworder=[r['file_path'] for r in new['ranking']]
            union=list(dict.fromkeys(oldorder+neworder)); ids=mixed_ids(union,a,b)
            oldids=[cid for p in oldorder for cid in ids[p]]
            if oldids!=[r['chunk_id'] for r in old['scored_chunks']]:raise ValueError('Old evidence policy not reproduced')
            oldlookup={r['chunk_id']:r for r in old['scored_chunks']}
            wanted={cid for p in union for cid in ids[p]}-oldlookup.keys()
            chunks_path=recall/source['a']['assets']/'chunks.json'
            extra=extract_chunks(chunks_path,wanted)
            lookup={**oldlookup,**extra}
            orderedids=list(dict.fromkeys(oldids+[cid for p in neworder for cid in ids[p]]))
            documents=[]
            for cid in orderedids:
                chunk=lookup[cid]
                if cid not in ids[chunk['file_path']]:raise ValueError('Chunk/file mismatch')
                documents.append({k:chunk[k] for k in ('chunk_id','file_path','embedding_text','start_line','end_line')})
            historical={r['chunk_id']:{k:r[k] for k in ('score','probability','input_tokens')} for r in old['scored_chunks']}
            check=aggregate(oldorder,ids,historical)
            if [r['file_path'] for r in check]!=[r['file_path'] for r in old['ranking']]:raise ValueError('Historical C aggregation differs')
            item={**meta,'status':'ok','query':old['query'],'ground_truth':old['ground_truth'],
                'missing_gold':source['old']['missing_gold'],'old_order':oldorder,'new_order':neworder,
                'chunk_ids':ids,'documents':documents,'historical_scores':historical,
                'historical_ranking':old['ranking'],'historical_metrics':source['old']['metrics'],
                'reference_ab_metrics':{'old_AB':old['module_metrics']['AB'],'new_AB':new['metrics']},
                'source_hashes':{str(p.resolve()):sha256_file(p) for p in (oldpath,newpath)},
                'retrieval_sha256':{str(apath):source['a']['result_sha256'],str(bpath):source['b']['result_sha256']}}
        path=folder/'prepared'/f"{sample['position']:03d}.json";save(path,item)
        prepared.append({'position':sample['position'],'status':item['status'],'path':str(path.relative_to(folder)),'sha256':sha256_file(path)})
        if sample['position']%25==0:print('prepared',sample['position'],'/500',flush=True)
    save(folder/'prepared_index.json',prepared)
    save(folder/'status.json',{'stage':'prepared','reports':500,'failed':sum(r['status']=='failed' for r in prepared)})


def worker(folder,gpu):
    protocol=verify(folder);config=protocol['config']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Wrong GPU visibility')
    index=read(folder/'prepared_index.json');subset=[r for r in index if (r['position']-1)%2==gpu]
    with (folder/f'gpu{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model=None
        for step,entry in enumerate(subset,1):
            path=folder/entry['path']
            if sha256_file(path)!=entry['sha256']:raise ValueError('Prepared input changed')
            result_path=folder/'cases'/f"{entry['position']:03d}.json"
            record_path=folder/'records'/f"{entry['position']:03d}.json"
            if record_path.exists():
                saved=read(record_path)
                if sha256_file(result_path)!=saved['result_sha256']:raise ValueError('Checkpoint changed')
                continue
            item=read(path);start=time.monotonic()
            if item['status']=='failed':
                result={**item,'physical_gpu':gpu}
            else:
                if model is None:
                    free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                    memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in free.splitlines()}
                    if memory[gpu]<config['minimum_free_mib']:raise ValueError('Insufficient free GPU memory')
                    model=LocalReranker(ROOT,protocol['reranker'])
                    if model.identity!=protocol['reference_model']:raise ValueError('C model identity changed')
                    save(folder/f'model_gpu{gpu}.json',{'physical_gpu':gpu,'identity':model.identity})
                values=model.score(item['query'],[r['embedding_text'] for r in item['documents']])
                scores={r['chunk_id']:v for r,v in zip(item['documents'],values)}
                if len(scores)!=len(item['documents']):raise ValueError('Missing scores')
                rankings={'old_AB':item['old_order'],'new_AB':item['new_order'],
                          'old_ABC_historical':[r['file_path'] for r in item['historical_ranking']]}
                ranked={}
                for name,order in [('old_ABC',item['old_order']),('new_ABC',item['new_order'])]:
                    ranked[name]=aggregate(order,item['chunk_ids'],scores)
                    rankings[name]=[r['file_path'] for r in ranked[name]]
                metrics={k:ranking_metrics(v,item['ground_truth'],10) for k,v in rankings.items()}
                for name in ('old_AB','new_AB'):
                    if metrics[name]!=item['reference_ab_metrics'][name]:raise ValueError('AB metrics changed')
                if metrics['old_ABC_historical']!=item['historical_metrics']:raise ValueError('Historical metrics changed')
                deltas=[scores[c]['score']-v['score'] for c,v in item['historical_scores'].items()]
                result={**{k:item[k] for k in (*IDENTITY,'position','source_position','ground_truth','missing_gold')},
                    'status':'ok','physical_gpu':gpu,'query_sha256_actual':hashlib.sha256(item['query'].encode()).hexdigest(),
                    'prepared_sha256':entry['sha256'],'rankings':rankings,'scored_rankings':ranked,
                    'scored_chunks':[{**r,**v} for r,v in zip(item['documents'],values)],'metrics':metrics,
                    'pool_metrics':{name:ranking_metrics(rank,item['ground_truth'],50) for name,rank in rankings.items()},
                    'historical_score_check':{'max_absolute_delta':max(map(abs,deltas)),
                        'identical_pairs':sum(d==0 for d in deltas),'pairs':len(deltas)},
                    'elapsed_seconds':time.monotonic()-start}
            save(result_path,result)
            save(record_path,{'position':entry['position'],'status':item['status'],
                 'result_file':str(result_path.relative_to(folder)),'result_sha256':sha256_file(result_path)})
            save(folder/f'progress_gpu{gpu}.json',{'processed':step,'selected':len(subset),'last_position':entry['position'],'updated_unix':time.time()})
            print(f'GPU{gpu} {step}/{len(subset)} position={entry["position"]} {item["status"]}',flush=True)


def finalize(folder):
    protocol=verify(folder);entries=read(folder/'prepared_index.json');selection=read(folder/'selection.json')['samples']
    records={m:[] for m in METHODS};cases=[];totalpairs=0
    for entry,meta in zip(entries,selection):
        record=read(folder/'records'/f"{meta['position']:03d}.json");path=folder/record['result_file']
        if sha256_file(path)!=record['result_sha256']:raise ValueError('Result checksum changed')
        case=read(path);item=read(folder/entry['path'])
        if sha256_file(folder/entry['path'])!=entry['sha256']:raise ValueError('Prepared input changed')
        if any(case[k]!=meta[k] for k in (*IDENTITY,'position')):raise ValueError('Unpaired results')
        if case['status']=='ok':
            scores={r['chunk_id']:r for r in case['scored_chunks']}
            if set(scores)!={r['chunk_id'] for r in item['documents']}:raise ValueError('Score inventory differs')
            for doc in item['documents']:
                if any(doc[k]!=scores[doc['chunk_id']][k] for k in doc):raise ValueError('Scored document changed')
            for name,order in [('old_ABC',item['old_order']),('new_ABC',item['new_order'])]:
                expected=sorted(order,key=lambda p:(-max(scores[c]['score'] for c in item['chunk_ids'][p]),order.index(p)))
                if case['rankings'][name]!=expected:raise ValueError('Independent Max/tie audit failed')
            totalpairs+=len(scores)
        for method in METHODS:
            row={**meta,'status':case['status']}
            if case['status']=='ok':
                metric=ranking_metrics(case['rankings'][method],case['ground_truth'],10)
                if metric!=case['metrics'][method]:raise ValueError('Metric recomputation failed')
                row.update(metrics=metric,missing_gold=case['missing_gold'])
            else:row.update(error_type=case['error_type'],error=case['error'])
            records[method].append(row)
        cases.append(case)
    summary={m:aggregate_results(rows,10,500) for m,rows in records.items()}
    save(folder/'samples.json',records);save(folder/'summary.json',summary)
    save(folder/'project_summary.json',{p:{m:aggregate_results([r for r in rows if r['repository']==p],10)
         for m,rows in records.items()} for p in sorted({r['repository'] for r in selection})})
    contrasts={'new_ABC_vs_old_ABC':('new_ABC','old_ABC'),'new_ABC_vs_new_AB':('new_ABC','new_AB')}
    fields=('average_precision_at_k','reciprocal_rank_at_k','top_1','top_10','recall_at_k')
    stats={}
    for contrast,(left,right) in contrasts.items():
        stats[contrast]={}
        for metric in fields:
            delta=[(a['metrics'][metric] if a['status']=='ok' else 0)-(b['metrics'][metric] if b['status']=='ok' else 0)
                   for a,b in zip(records[left],records[right])]
            stats[contrast][metric]=inference(delta,[r['repository'] for r in selection],seed=20261123,repetitions=5000)
    tests=[v for contrast in stats.values() for v in contrast.values()];previous=0.
    for i,test in enumerate(sorted(tests,key=lambda x:x['p_two_sided_paired_randomization'])):
        previous=min(1.,max(previous,(len(tests)-i)*test['p_two_sided_paired_randomization']))
        test['p_holm_10_tests']=previous
    save(folder/'paired_statistics.json',stats)
    with (folder/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        rows=[{'method':m,'n':500,'successful':v['successful_samples'],'failed':v['failed_samples'],**v['metrics']} for m,v in summary.items()]
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(folder/'verification.json',{'status':'passed','reports':500,'success':sum(c['status']=='ok' for c in cases),
         'failed_positions':[c['position'] for c in cases if c['status']!='ok'],'fresh_scored_pairs':totalpairs,
         'historical_max_score_delta':max(c.get('historical_score_check',{}).get('max_absolute_delta',0) for c in cases),
         'checks':['same fixed500 identities','same original report query and mixed evidence policy',
                   'current AB exactly matches frozen baseline','fresh shared C scores for both arms',
                   'independent Max/tie and metrics audit','all checkpoints checksummed']})
    save(folder/'status.json',{'stage':'completed','reports':500,'finished_unix':time.time()})
    print(json.dumps({m:v['metrics'] for m,v in summary.items()},indent=2),flush=True)


def run(folder):
    verify(folder);save(folder/'status.json',{'stage':'running','started_unix':time.time()})
    with (folder/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        procs=[];logs=[]
        try:
            for gpu in (0,1):
                log=(folder/f'gpu{gpu}.log').open('a');logs.append(log)
                procs.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'worker','--output',str(folder),'--gpu',str(gpu)],
                    cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1',
                                 'TOKENIZERS_PARALLELISM':'false'},stdout=log,stderr=subprocess.STDOUT))
            save(folder/'workers.json',{'pids':[p.pid for p in procs]})
            while any(p.poll() is None for p in procs):
                if any(p.poll() not in (None,0) for p in procs):raise RuntimeError('GPU worker failed; inspect logs')
                time.sleep(5)
            if any(p.returncode for p in procs):raise RuntimeError('GPU worker failed')
            finalize(folder)
        finally:
            for p in procs:
                if p.poll() is None:p.terminate()
            for log in logs:log.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','worker','finalize'))
    parser.add_argument('--output',type=Path,default=ROOT/read(CONFIG)['output'])
    parser.add_argument('--gpu',type=int,choices=(0,1));args=parser.parse_args()
    try:
        if args.action=='worker':worker(args.output,args.gpu)
        else:{'prepare':prepare,'run':run,'finalize':finalize}[args.action](args.output)
    except Exception:
        if args.output.exists():save(args.output/f'error_{args.action}_{args.gpu}.json',{'error':traceback.format_exc(),'time':time.time()})
        raise


if __name__=='__main__':main()
