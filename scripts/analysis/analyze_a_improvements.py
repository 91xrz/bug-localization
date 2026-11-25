"""Independent rank-metric audit and robustness analyses for the frozen A study."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/experiments'))
from bug_localization.paired_statistics import inference
from run_a_improvements import select


def read(p):return json.loads(p.read_bytes())


def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def independent_metrics(paths,gold):
    relevant=np.array([int(p in gold) for p in paths[:10]],dtype=float)
    positions=np.flatnonzero(relevant)+1
    ap=float(((np.cumsum(relevant)/np.arange(1,len(relevant)+1))*relevant).sum()/len(gold))
    hits50=len(gold.intersection(paths[:50]))
    return {'MAP@10':ap,'MRR@10':float(1/positions[0]) if len(positions) else 0.,
            'Hit@1':float(len(positions)>0 and positions[0]==1),'Hit@10':float(len(positions)>0),
            'MeanRecall@10':float(relevant.sum()/len(gold)),'Hit@50':float(hits50>0),
            'MeanRecall@50':hits50/len(gold),'PoolHit':float(hits50>0),
            'PoolRecall':hits50/len(gold),'gold50':hits50,'pool_size':len(paths)}


def aggregate(rows,names):
    return {name:{k:float(np.mean([r['methods'][name]['metrics'][k] for r in rows]))
                  for k in rows[0]['methods'][name]['metrics']} for name in names}


def contrast(rows,method,baseline='A'):
    keys=('MAP@10','MRR@10','Hit@10','Hit@50','MeanRecall@50')
    return {k:float(np.mean([r['methods'][method]['metrics'][k]-r['methods'][baseline]['metrics'][k] for r in rows]))
            for k in keys}


def development_checks(dev):
    names=list(dev[0]["methods"])
    repeats=[]
    for seed in range(20261122,20261142):
        rng=np.random.default_rng(seed);train=[];test=[]
        for repo in sorted({r['repository'] for r in dev}):
            group=[r for r in dev if r['repository']==repo]
            indices=rng.permutation(len(group));cut=max(1,int(.7*len(group)))
            train.extend(group[i] for i in indices[:cut]);test.extend(group[i] for i in indices[cut:])
        selected=select(aggregate(train,names))['selected']
        repeats.append({'seed':seed,'train_n':len(train),'test_n':len(test),'selected':selected,
                        'test_delta':contrast(test,selected)})
    project_cv=[]
    for repo in sorted({r['repository'] for r in dev}):
        train=[r for r in dev if r['repository']!=repo];test=[r for r in dev if r['repository']==repo]
        selected=select(aggregate(train,names))['selected']
        project_cv.append({'held_out_project':repo,'n':len(test),'selected':selected,'delta':contrast(test,selected)})
    return {"repeated_splits":repeats,"leave_one_project_out_selection":project_cv}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--development-only',action='store_true');args=parser.parse_args()
    out=args.run;selection=read(out/'selection.json');frozen=read(out/'frozen_choice.json')
    if args.development_only:
        dev=[read(out/'cases'/f'{m["position"]:04d}.json') for m in selection['samples']
             if m['dataset_row'] in set(selection['development_rows'])]
        checks=development_checks(dev)
        checks['development_summary_sha256']=hashlib.sha256((out/'development500_summary.json').read_bytes()).hexdigest()
        save(out/'development_stability.json',checks)
        print(json.dumps({'positive_splits':sum(r['test_delta']['MAP@10']>0 for r in checks['repeated_splits']),
                          'splits':len(checks['repeated_splits']),
                          'positive_heldout_projects':sum(r['delta']['MAP@10']>0 for r in checks['leave_one_project_out_selection']),
                          'projects':len(checks['leave_one_project_out_selection'])}))
        return
    assert read(out/'verification.json')['status']=='passed'
    rows=[read(out/'cases'/f'{m["position"]:04d}.json') for m in selection['samples']]
    assert len(rows)==2294
    total=0;maxerr=0.
    for meta,r in zip(selection['samples'],rows):
        assert all(r[k]==v for k,v in meta.items())
        for name,m in r['methods'].items():
            rank=m['ranking'];assert len(rank)<=50 and len(rank)==len(set(rank))
            independent=independent_metrics(rank,set(r['ground_truth']))
            for key,value in independent.items():
                error=abs(value-m['metrics'][key]);assert error<1e-12;maxerr=max(maxerr,error)
            if name.startswith(('anchor_d50','prf_')):
                assert set(rank)==set(r['methods']['A']['ranking'])
            total+=1
    for p,sha in read(out/'protocol.json')['hashes'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha
    dev=[r for r in rows if r['phase']=='development500'];val=[r for r in rows if r['phase']=='remaining1794']
    names=list(dev[0]['methods']);summary=aggregate(dev,names)
    chosen=select(summary)
    assert chosen['selected']==frozen['selected'] and chosen['families']==frozen['families']
    assert all((out/'cases'/f'{r["position"]:04d}.json').stat().st_mtime>=frozen['frozen_unix'] for r in val)
    save(out/'independent_audit.json',{'status':'passed','cases':len(rows),'rankings_recomputed':total,
        'max_metric_error':maxerr,'frozen_selection_reproduced':True,'validation_created_after_freeze':True,
        'all_frozen_inputs_rehashed':True,'scope':'Independent metric implementation and selection checks; main runner verified source/vector and full result hashes.'})
    method=frozen['selected'];methods=sorted({method,*frozen['families'].values()}-{'A'})
    checks=(read(out/'development_stability.json') if (out/'development_stability.json').exists() else development_checks(dev))
    if 'development_summary_sha256' in checks:
        assert checks['development_summary_sha256']==hashlib.sha256((out/'development500_summary.json').read_bytes()).hexdigest()
    repeats=checks["repeated_splits"];project_cv=checks["leave_one_project_out_selection"]
    seen=set();unique=[];devhash={r['query_sha256'] for r in dev}
    for row in val:
        if row['query_sha256'] not in seen and row['query_sha256'] not in devhash:
            unique.append(row);seen.add(row['query_sha256'])
    robustness={'scope':'Repeated splits and LOPO are development sensitivity checks, not independent test sets.',
                'repeated_splits':repeats,'leave_one_project_out_selection':project_cv,
                'repeat_summary':{'positive_MAP_runs':sum(x['test_delta']['MAP@10']>0 for x in repeats),
                                  'total_runs':len(repeats),'chosen_counts':dict(Counter(x['selected'] for x in repeats)),
                                  'MAP_delta_range':[min(x['test_delta']['MAP@10'] for x in repeats),max(x['test_delta']['MAP@10'] for x in repeats)],
                                  'MAP_delta_mean':float(np.mean([x['test_delta']['MAP@10'] for x in repeats]))},
                'query_dedup_validation':{'n':len(unique),'excluded':len(val)-len(unique),'methods':{}},
                'truncation_sensitivity':{},'evidence_sensitivity':{},'combination_increment':{}}
    for name in methods:
        robustness['query_dedup_validation']['methods'][name]=inference(
            [r['methods'][name]['metrics']['MAP@10']-r['methods']['A']['metrics']['MAP@10'] for r in unique],
            [r['repository'] for r in unique],seed=20261122,repetitions=5000)
        robustness['truncation_sensitivity'][name]={str(flag):{'n':len(group),'delta':contrast(group,name)}
            for flag in (True,False) if (group:=[r for r in val if r['query_truncated']==flag])}
        if name.startswith(('anchor_','combo_')):
            def evidence_kind(row):
                evidence=row['anchor_evidence'][name]
                return ('explicit_path_or_module' if any(e['explicit_path_or_module'] for e in evidence.values())
                        else 'identifiers_only' if evidence else 'none')
            robustness['evidence_sensitivity'][name]={kind:{'n':len(group),'delta':contrast(group,name)}
                for kind in ('explicit_path_or_module','identifiers_only','none')
                if (group:=[r for r in val if evidence_kind(r)==kind])}
    dev_projects={r['repository'] for r in dev}
    transfer=[r for r in val if r['repository'] not in dev_projects]
    robustness['projects_absent_from_development']={
        'n':len(transfer),'projects':sorted({r['repository'] for r in transfer}),
        'note':'No rows from these projects selected parameters in this study; historical prior observation remains.',
        'methods':{name:inference([r['methods'][name]['metrics']['MAP@10']-r['methods']['A']['metrics']['MAP@10'] for r in transfer],
                                  [r['repository'] for r in transfer],repetitions=5000) for name in methods}}
    anchor=frozen['families']['anchor'];combo=frozen['families']['combination']
    robustness['combination_increment']={'combo':combo,'anchor':anchor,'validation_delta':contrast(val,combo,anchor),
       'MAP_inference_descriptive':inference([r['methods'][combo]['metrics']['MAP@10']-r['methods'][anchor]['metrics']['MAP@10'] for r in val],
                                             [r['repository'] for r in val],repetitions=5000)}
    robustness['bootstrap_seed_sensitivity']={str(seed):inference(
        [r['methods'][method]['metrics']['MAP@10']-r['methods']['A']['metrics']['MAP@10'] for r in val],
        [r['repository'] for r in val],seed=seed,repetitions=5000)['ci95_project_stratified_bootstrap']
        for seed in (11,23,47,89,131)}
    save(out/'robustness.json',robustness)
    deltas=sorted(val,key=lambda r:r['methods'][method]['metrics']['MAP@10']-r['methods']['A']['metrics']['MAP@10'])
    def example(row):
        return {k:row[k] for k in ('position','repository','issue_id','ground_truth')} | {
            'delta_MAP':row['methods'][method]['metrics']['MAP@10']-row['methods']['A']['metrics']['MAP@10'],
            'A_top10':row['methods']['A']['ranking'][:10],
            'selected_top10':row['methods'][method]['ranking'][:10],
            'evidence':row['anchor_evidence'].get(method,{})}
    save(out/'examples.json',{'largest_losses':[example(r) for r in deltas[:10]],
                              'largest_gains':[example(r) for r in reversed(deltas[-10:])]})
    # A+structural grounding alone is distinct from the already adopted A+B baseline.
    current=read(ROOT/'configs/baselines/ab_structure_v1.json')
    save(out/'current_baseline_context.json',{'warning':'Reference only; no new A+ result has been integrated into B or C.',
        'current_id':current['id'],'current_metrics':current['reference_metrics'],
        'selected_A_method':method,'selected_A_metrics':{c:m[method] for c,m in read(out/'summary.json').items()}})
    ablation=out/'anchor_ablation'
    if (ablation/'summary.json').exists():
        abrows=[read(ablation/'cases'/f'{r["position"]:04d}.json') for r in rows]
        for parent,child in zip(rows,abrows):
            assert child['methods']['full']['ranking']==parent['methods'][method]['ranking']
            for name,m in child['methods'].items():
                independent=independent_metrics(m['ranking'],set(parent['ground_truth']))
                assert all(abs(v-m['metrics'][k])<1e-12 for k,v in independent.items())
        av=[r for r in abrows if r['dataset_row'] not in set(selection['development_rows'])]
        tests={name:inference([r['methods']['full']['metrics']['MAP@10']-r['methods'][name]['metrics']['MAP@10'] for r in av],
                             [r['repository'] for r in av],repetitions=5000)
               for name in ('aggregation_only','encoded_report_only','identifiers_only','paths_only')}
        previous=0.
        for i,t in enumerate(sorted(tests.values(),key=lambda x:x['p_two_sided_paired_randomization'])):
            previous=min(1.,max(previous,(len(tests)-i)*t['p_two_sided_paired_randomization']))
            t['p_holm_four_ablation_tests']=previous
        save(ablation/'verification.json',{'status':'passed','full_rankings_equal_parent':len(abrows),
                                          'all_metrics_recomputed':True,'validation_MAP_full_minus_ablation':tests})
    recall=out/'recall_control'
    if (recall/'summary.json').exists():
        rmethod=read(recall/'protocol.json')['method']
        recall_rows=[read(recall/'cases'/f'{r["position"]:04d}.json') for r in rows]
        devset=set(selection['development_rows'])
        for parent,child in zip(rows,recall_rows):
            assert child['methods']['A']['ranking']==parent['methods']['A']['ranking']
            if child['dataset_row'] in devset:
                assert child['methods'][rmethod]['ranking']==parent['methods'][rmethod]['ranking']
            for m in child['methods'].values():
                independent=independent_metrics(m['ranking'],set(parent['ground_truth']))
                assert all(abs(v-m['metrics'][k])<1e-12 for k,v in independent.items())
        rv=[r for r in recall_rows if r['dataset_row'] not in devset]
        tests={key:inference([r['methods'][rmethod]['metrics'][key]-r['methods']['A']['metrics'][key] for r in rv],
                            [r['repository'] for r in rv],repetitions=5000)
               for key in ('MAP@10','MRR@10','Hit@10','Hit@50','MeanRecall@50')}
        extended=read(out/'validation_inference.json')['comparisons'];extended[rmethod]=tests
        comparisons=[t for method_tests in extended.values() for t in method_tests.values()];previous=0.
        for i,t in enumerate(sorted(comparisons,key=lambda x:x['p_two_sided_paired_randomization'])):
            previous=min(1.,max(previous,(len(comparisons)-i)*t['p_two_sided_paired_randomization']))
            t['p_holm_with_recall_control']=previous
        save(out/'extended_validation_inference.json',{'comparisons':extended,'holm_family_size':len(comparisons)})
        save(recall/'verification.json',{'status':'passed','baseline_rankings_reproduced':2294,
             'development_rankings_reproduced':500,'all_metrics_recomputed':True,'validation_inference':tests})
    print(json.dumps({'audit':'passed','selected':method,'rankings':total,'repeats':robustness['repeat_summary'],
                       'dedup_n':len(unique)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
