"""Offline paired fusion exploration of two already-computed full file rankings."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from bug_localization.batch import write_json, aggregate_results
from bug_localization.bm25 import ranking_metrics

ORIG=ROOT/'results/report_hypothesis/20261110_023926_805713_retrieval'
DIRECT=ROOT/'results/direct_report_hypothesis/20261110_113033_098831_retrieval'


def indexed(ranking):
    result={}
    for rank,item in enumerate(ranking,1):
        path=item['file_path']
        if item['rank'] != rank or path in result: raise ValueError('Ranking must be unique and consecutive')
        result[path]=(rank,float(item['aggregated_similarity']))
    return result


def interleave(a,b):
    seen=set();output=[]
    for i in range(max(len(a),len(b))):
        for ranking in (a,b):
            if i<len(ranking):
                path=ranking[i]['file_path']
                if path not in seen: seen.add(path);output.append(path)
    return output


def normalized(index):
    if not index: return {}
    values=[score for _,score in index.values()];lo=min(values);hi=max(values)
    return {p:(v-lo)/(hi-lo) if hi>lo else 0.0 for p,(_,v) in index.items()}


def fuse(a,b):
    ai,bi=indexed(a),indexed(b)
    paths=ai.keys()|bi.keys()
    an,bn=normalized(ai),normalized(bi)
    results={'original':[x['file_path'] for x in a], 'hypothesis':[x['file_path'] for x in b],
             'interleave_original_first':interleave(a,b),'interleave_hypothesis_first':interleave(b,a)}
    for k in (10,60):
        for w in (0.25,0.5,1,2,4):
            scores={p:(1/(k+ai[p][0]) if p in ai else 0)+(w/(k+bi[p][0]) if p in bi else 0) for p in paths}
            results[f'rrf_k{k}_w{w}']=sorted(paths,key=lambda p:(-scores[p],p))
    for w in (0.25,0.5,1,2,4):
        scores={p:(an.get(p,0)+w*bn.get(p,0))/(1+w) for p in paths}
        results[f'minmax_mean_w{w}']=sorted(paths,key=lambda p:(-scores[p],p))
    scores={p:max(an.get(p,0),bn.get(p,0)) for p in paths}
    results['minmax_max']=sorted(paths,key=lambda p:(-scores[p],p))
    return results


def main():
    prior=json.loads((ORIG/'selection.json').read_text()); current=json.loads((DIRECT/'selection.json').read_text())
    if prior['samples']!=current['samples']: raise ValueError('Selection mismatch')
    audit=json.loads((DIRECT/'audit.json').read_text())
    if not all(audit[k] for k in ('selection_verified','code_vectors_all_cache_hits','embedding_identity_matches','candidate_and_chunk_inputs_match','original_and_enriched_metrics_recomputed')): raise ValueError('Previous pairing audit failed')
    folder=ROOT/'results/direct_report_fusion'/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True);write_json(folder/'selection.json',current)
    write_json(folder/'protocol.json',{'scope':'Exploratory parameter search on the same 180 reports; no independent test or generalization claim', 'input_original':str(ORIG),'input_hypothesis':str(DIRECT),'rrf_constants':[10,60],'hypothesis_weights':[0.25,0.5,1,2,4],'score_normalization':'per-query min-max across complete file ranking; constant scores map to zero', 'interleave':'round-robin with first occurrence preserved, duplicate paths removed', 'score':'aggregated cosine similarity without +1 offset', 'tie_break':'file path ascending','ranking_depth':'complete file rankings; fused first then evaluated @10'})
    print('Results:',folder,flush=True)
    records={};pairs=[]
    for sample in current['samples']:
        pos=sample['position'];case=f'{pos:03d}'
        a=json.loads((ORIG/'cases'/case/'q0_result.json').read_text())
        b=json.loads((DIRECT/'cases'/case/'direct_result.json').read_text())
        if a['ground_truth']!=b['ground_truth']: raise ValueError('Ground truth mismatch '+case)
        ai,bi=indexed(a['ranking']),indexed(b['ranking'])
        if bi and set(ai)!=set(bi): raise ValueError('Candidate inventories differ '+case)
        if ranking_metrics([x['file_path'] for x in a['ranking']],a['ground_truth'],10)!=a['metrics']: raise ValueError('Original metric mismatch')
        if ranking_metrics([x['file_path'] for x in b['ranking']],b['ground_truth'],10)!=b['metrics']: raise ValueError('Hypothesis metric mismatch')
        results=fuse(a['ranking'],b['ranking']);gold=a['ground_truth'];missing=sorted(set(gold)-set(ai))
        sample_results={}
        for name,paths in results.items():
            metrics=ranking_metrics(paths,gold,10)
            records.setdefault(name,[]).append({**sample,'status':'ok','metrics':metrics,'missing_gold':missing})
            sample_results[name]={'metrics':metrics,'top10':paths[:10]}
        pairs.append({**sample,'results':sample_results})
        if pos%20==0: print('Processed',pos,flush=True)
        write_json(folder/'status.json',{'status':'running','processed':pos})
    summary={name:aggregate_results(rows,10,len(current['samples'])) for name,rows in records.items()}
    projects={repo:{name:aggregate_results([r for r in rows if r['repository']==repo],10) for name,rows in records.items()} for repo in sorted({r['repository'] for r in records['original']})}
    comparisons=[]
    for name,rows in records.items():
        row={'method':name,**summary[name]['metrics']}
        for baseline in ('original','hypothesis'):
            p=[(x['metrics']['average_precision_at_k'],y['metrics']['average_precision_at_k']) for x,y in zip(rows,records[baseline])]
            row.update({f'delta_vs_{baseline}':sum(a-b for a,b in p)/len(p), f'wins_vs_{baseline}':sum(a>b for a,b in p),f'ties_vs_{baseline}':sum(a==b for a,b in p),f'losses_vs_{baseline}':sum(a<b for a,b in p)})
        comparisons.append(row)
    comparisons.sort(key=lambda x:-x['MAP@10'])
    with (folder/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    write_json(folder/'summary.json',summary);write_json(folder/'project_summary.json',projects);write_json(folder/'paired_samples.json',pairs)
    write_json(folder/'status.json',{'status':'completed','processed':len(pairs),'routes':len(records)})
    print(json.dumps({'status':'completed','result_directory':str(folder),'top_methods':comparisons[:8]},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
