"""Independent metric reconstruction and gain/loss attribution for CPU B study."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from bug_localization.bm25 import ranking_metrics


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=ROOT/'results/b_lightweight/20261122_v1')
    args=parser.parse_args(); run=args.run
    summary=read(run/'summary.json'); choice=read(run/'frozen_choice.json')['method']
    rows=[read(p) for p in sorted((run/'cases').glob('*.json'))]
    assert len(rows)==2294 and [r['position'] for r in rows]==list(range(1,2295))
    mapping={'MAP@10':'average_precision_at_k','MRR@10':'reciprocal_rank_at_k',
             'Hit@1':'top_1','Hit@10':'top_10','MeanRecall@10':'recall_at_k'}
    checks=0
    for r in rows:
        for name,m in r['methods'].items():
            rank=m['ranking']; gold=r['ground_truth']; expected=m['metrics']
            assert len(rank)==len(set(rank))
            assert name=='union_ceiling_Border' or len(rank)<=50
            at10=ranking_metrics(rank,gold,10); at50=ranking_metrics(rank,gold,50)
            for k,v in mapping.items(): assert abs(expected[k]-at10[v])<1e-12
            assert expected['Hit@50']==int(at50['retrieved_gold_files']>0)
            assert abs(expected['MeanRecall@50']-at50['recall_at_k'])<1e-12
            checks+=1
    for cohort,methods in summary.items():
        chosen_rows=rows if cohort=='all2294' else [r for r in rows if r['phase']==cohort]
        for name,ms in methods.items():
            for k,v in ms.items():
                assert abs(sum(r['methods'][name]['metrics'][k] for r in chosen_rows)/len(chosen_rows)-v)<1e-12
    diagnostics={}
    for cohort in summary:
        rs=rows if cohort=='all2294' else [r for r in rows if r['phase']==cohort]
        stats={}
        for baseline in ('B','current_RRF60'):
            d=[r['methods'][choice]['metrics']['MAP@10']-r['methods'][baseline]['metrics']['MAP@10'] for r in rs]
            stats[baseline]={'ap_wins':sum(x>1e-12 for x in d),'ap_ties':sum(abs(x)<=1e-12 for x in d),'ap_losses':sum(x < -1e-12 for x in d)}
            for k in ('Hit@10','Hit@50'):
                stats[baseline][k]={'rescued':sum(r['methods'][choice]['metrics'][k]>r['methods'][baseline]['metrics'][k] for r in rs),
                                    'lost':sum(r['methods'][choice]['metrics'][k]<r['methods'][baseline]['metrics'][k] for r in rs)}
        stats['active_anchor_reports']=sum(bool(r.get('anchor_features')) for r in rs)
        # How much of B's complementary gold is retained by the new fixed50 pool?
        success=[r for r in rs if r['status']=='ok']
        newly_available=sum(len((set(r['methods']['B']['ranking'])-set(r['methods']['A']['ranking'])) & set(r['ground_truth'])) for r in success)
        retained=sum(len((set(r['methods']['B']['ranking'])-set(r['methods']['A']['ranking'])) & set(r['ground_truth']) & set(r['methods'][choice]['ranking'])) for r in success)
        stats['B_exclusive_gold']={'available':newly_available,'retained':retained,'dropped':newly_available-retained}
        diagnostics[cohort]=stats
    examples=[]
    for r in rows:
        delta=r['methods'][choice]['metrics']['MAP@10']-r['methods']['B']['metrics']['MAP@10']
        if abs(delta)<1e-12:continue
        br={p:i for i,p in enumerate(r['methods']['B']['ranking'],1)}
        sr={p:i for i,p in enumerate(r['methods'][choice]['ranking'],1)}
        examples.append({k:r[k] for k in ('position','repository','issue_id','phase')} | {
            'delta_ap':delta,'gold':[{'path':p,'B_rank_within50':br.get(p),'selected_rank_within50':sr.get(p),
                                    'anchor':r.get('anchor_features',{}).get(p)} for p in r['ground_truth']]})
    examples.sort(key=lambda r:-abs(r['delta_ap']))
    report={'status':'passed','metric_checks':checks,'samples':len(rows),'selected':choice,
            'diagnostics':diagnostics,'largest_changes':examples[:40],
            'checks':['every saved Top10/Top50 metric independently recomputed with existing evaluator',
                      'all cohort means recomputed','unique bounded candidate lists','complementary gold retention counted'],
            'case_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((run/'cases').glob('*.json'))}}
    (run/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('case_files_sha256','largest_changes')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
