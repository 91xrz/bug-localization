"""Single frozen CPU pilot to separate three fusion regression hypotheses."""
import csv
import json
import math
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from bug_localization.batch import aggregate_results,write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.snapshot import sha256_file


def main():
    out=ROOT/'results/fusion_diagnosis/20261117_repair90';out.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/experiments/fusion_repair90.json').read_text())
    selection=json.loads((ROOT/config['selection_source']).read_text())['samples']
    excluded={s['position'] for s in json.loads((ROOT/config['exclude_previous_pilot']).read_text())['samples']}
    selected=[];halves={}
    for repo in sorted({s['repository'] for s in selection}):
        eligible=[s for s in selection if s['repository']==repo and s['position'] not in excluded]
        chosen=random.Random(config['seed']+':'+repo).sample(eligible,config['per_project'])
        for i,s in enumerate(chosen):halves[s['position']]='half1' if i<5 else 'half2'
        selected+=chosen
    selected.sort(key=lambda s:s['position'])
    assert len(selected)==90 and len({s['position'] for s in selected})==90
    write_json(out/'protocol.json',config)
    write_json(out/'selection.json',{'samples':selected,'halves':halves})
    run=ROOT/'results/module_ablation/20261116_rrf500'
    samples=json.loads((run/'samples.json').read_text())
    records={m:[] for m in config['methods']};pairs=[];hashes={}
    for s in selected:
        i=s['position']-1
        if samples['B'][i]['status']!='ok':
            for m in records:records[m].append({**samples['B'][i],**s})
            pairs.append({**s,'half':halves[s['position']],'status':'failed','methods':{}})
            continue
        pos=s['position'] if i<180 else s['position']-180
        paths=[Path(samples[m][i]['source_root'])/'cases'/f'{pos:03d}'/f for m,f in [('A','q0_result.json'),('B','direct_result.json')]]
        a,b=[json.loads(p.read_text()) for p in paths]
        savedpath=run/'cases'/f"{s['position']:03d}.json";saved=json.loads(savedpath.read_text())
        for p in paths:
            hashes[str(p.resolve())]=sha256_file(p)
            assert saved['source_hashes'][str(p.resolve())]==hashes[str(p.resolve())]
        hashes[str(savedpath.resolve())]=sha256_file(savedpath)
        assert all(s[k]==samples['B'][i][k]==saved[k] for k in ('position','dataset_row','repository','issue_id','before_fix_sha','query_sha256','labels_sha256'))
        ar={r['file_path']:r['rank'] for r in a['ranking']};br={r['file_path']:r['rank'] for r in b['ranking']}
        assert ar.keys()==br.keys() and a['ground_truth']==b['ground_truth']==saved['ground_truth']
        gold=a['ground_truth'];pool={r['file_path'] for r in saved['candidates']}
        score=lambda k,w: sorted(pool,key=lambda p: (-(1/(k+ar[p])+w/(k+br[p])),p))
        old=score(60,1);assert old==saved['ab_ranking']
        first=min(br,key=br.get); assert first in pool
        ranks={'B':sorted(pool,key=br.get),'RRF60_equal':old,'RRF60_B2':score(60,2),
               'RRF10_equal':score(10,1),'B_top1_then_RRF60':[first]+[p for p in old if p!=first]}
        results={}
        for m,rank in ranks.items():
            assert len(rank)==len(pool)==len(set(rank)) and set(rank)==pool
            metrics=ranking_metrics(rank,gold,10)
            if m=='B':assert metrics==samples['B'][i]['metrics']
            if m=='RRF60_equal':assert metrics==samples['A+B'][i]['metrics']
            # Independent AP reconstruction from saved top ten.
            hits=[j for j,p in enumerate(rank[:10],1) if p in set(gold)]
            assert math.isclose(metrics['average_precision_at_k'],sum(j/r for j,r in enumerate(hits,1))/len(set(gold)),abs_tol=1e-12)
            record={**s,'status':'ok','metrics':metrics,'missing_gold':sorted(set(gold)-set(ar))}
            records[m].append(record);results[m]={'metrics':metrics,'top10':rank[:10]}
        pairs.append({**s,'half':halves[s['position']],'status':'ok','ground_truth':gold,'methods':results})
    summary={cohort:{m:aggregate_results([r for r in rows if cohort=='all90' or halves[r['position']]==cohort],10)
        for m,rows in records.items()} for cohort in ('all90','half1','half2')}
    ap=lambda r,m:r['methods'].get(m,{}).get('metrics',{}).get('average_precision_at_k',0)
    comparisons=[]
    for m in records:
        stats=summary['all90'][m]['metrics'];base=summary['all90']['RRF60_equal']['metrics'];b=summary['all90']['B']['metrics']
        row={'method':m,**stats,'delta_vs_B':stats['MAP@10']-b['MAP@10'],'delta_vs_current':stats['MAP@10']-base['MAP@10']}
        for name,ref in [('B','B'),('current','RRF60_equal')]:
            delta=[ap(p,m)-ap(p,ref) for p in pairs]
            row.update({f'wins_vs_{name}':sum(d>1e-12 for d in delta),f'ties_vs_{name}':sum(abs(d)<=1e-12 for d in delta),f'losses_vs_{name}':sum(d < -1e-12 for d in delta)})
        row.update({'repairs_current_losses':sum(ap(p,'RRF60_equal')<ap(p,'B')-1e-12 and ap(p,m)>ap(p,'RRF60_equal')+1e-12 for p in pairs),
                    'harms_current_wins':sum(ap(p,'RRF60_equal')>ap(p,'B')+1e-12 and ap(p,m)<ap(p,'RRF60_equal')-1e-12 for p in pairs)})
        row['passes_screen']=(row['delta_vs_B']>=.005 and row['delta_vs_current']>=.005 and stats['MRR@10']>=base['MRR@10'] and stats['Top-10']>=b['Top-10'] and all(summary[h][m]['metrics']['MAP@10']>summary[h]['RRF60_equal']['metrics']['MAP@10'] for h in ('half1','half2')))
        comparisons.append(row)
    write_json(out/'summary.json',summary);write_json(out/'paired_samples.json',pairs);write_json(out/'comparison.json',comparisons)
    write_json(out/'project_summary.json',{repo:{m:aggregate_results([r for r in rows if r['repository']==repo],10) for m,rows in records.items()} for repo in sorted({s['repository'] for s in selected})})
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    for p in [Path(__file__),ROOT/'configs/experiments/fusion_repair90.json',run/'samples.json',ROOT/'src/bug_localization/bm25.py',ROOT/'src/bug_localization/batch.py']:
        hashes[str(p.resolve())]=sha256_file(p)
    write_json(out/'source_hashes.json',hashes)
    write_json(out/'verification.json',{'status':'passed','samples':90,'gpu_calls':0,'failures':summary['all90']['B']['failed_samples'],
        'checks':['selection frozen before pilot outcomes','baseline metrics matched','source hash checked against full run','same candidate pool','independent AP reconstruction']})
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
