"""Post-hoc, label-separated diagnostics of the frozen C500 improvements.

No model calls, fitting, new thresholds, or changes to published rankings.
Leave-one-signal-out runs are explanatory counterfactuals, not selected methods.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/experiments'),str(ROOT/'scripts/analysis')]
from bug_localization.lightweight_hypothesis import normalize
from audit_c_revision import metrics, read, sha
from run_conservative_reranking import save, interval, FULL, PRIOR, IDENTITY


def test_path(path):
    parts=path.split('/')
    return any(x in {'test','tests','testing'} for x in parts[:-1]) or parts[-1].startswith('test_') or parts[-1]=='conftest.py' or parts[-1].endswith('_test.py')


def rank_features(base, features, active):
    """No labels: fixed .75z+.25u plus retained 2A/2B/2E signals."""
    score={f:.75*features[f]['z']+.25*features[f]['u']+2*sum(features[f][k] for k in active) for f in base}
    index={f:i for i,f in enumerate(base)}
    return sorted(base[:10],key=lambda f:(-score[f],index[f]))+base[10:]


def analyze(source,out):
    out.mkdir(parents=True,exist_ok=False)
    variants={'full':('a','b','e'),'without_original_A':('b','e'),'without_hypothesis_B':('a','e'),
              'without_structure_E':('a','b'),'structure_without_semantic':('e',)}
    save(out/'protocol.json',{
        'scope':'Post-hoc descriptive and fixed-weight signal removal; no fitting or model calls',
        'variants':variants,'formula':'.75z+.25u+2A+2B+2E, retained signals only; head10 unchanged',
        'not_causal_partition':'Marginal effects overlap and must not be added as independent contribution percentages',
        'cohort':'Historical500 previously used in development; not a fresh test',
        'path_diagnostic':'Test-path heuristic only; not manual code-role labels',
        'source_sha256':{name:sha(source/name) for name in ('protocol.json','labels.json','rankings.json','summary.json','verification.json','independent_audit.json')},
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'src/bug_localization/lightweight_hypothesis.py')},
    })
    rows=read(source/'labels.json');saved=read(source/'rankings.json')['orders']
    records={r['dataset_row']:r for r in read(FULL/'FULL/samples.json')}
    validation=read(source/'verification.json')
    counts=defaultdict(Counter);ap_groups=defaultdict(list);score_groups=defaultdict(list)
    outputs={k:[] for k in variants};details=[];source_hashes={}
    old_corrected={n:set() for n in ('prior_head10','packet_head10','combined_head10')}
    old_broken={n:set() for n in old_corrected}
    for i,row in enumerate(rows):
        dataset_row=row['dataset_row'];base=row['base'];gold=set(row['gold'])
        result_path=source/'cases'/f'{dataset_row:04d}.json'
        if sha(result_path)!=validation['result_sha256'][str(result_path.relative_to(source))]:
            raise ValueError('Frozen score file changed')
        scores={r['file_path']:r['score'] for r in read(result_path)['scores']}
        features={}
        if base:
            inputs_path=PRIOR/'inputs'/f"{records[dataset_row]['position']:04d}.json"
            inp=read(inputs_path);source_hashes[str(inputs_path)]=sha(inputs_path)
            if any(inp['sample'][k]!=row[k] for k in IDENTITY if k!='position'):
                raise ValueError('Prior sample identity differs')
            a,b=normalize(inp['a']),normalize(inp['b'])
            for f in base:
                features[f]={'z':max(row['chunks'][f]),'u':scores[f],'a':a[f],'b':b[f],'e':inp['anchor'][f]}
                if abs(.5*(a[f]+b[f]+inp['anchor'][f])-row['prior'][f])>1e-12:
                    raise ValueError('Initial score decomposition does not reproduce')
        for name,active in variants.items():
            order=rank_features(base,features,active)
            if set(order[:10])!=set(base[:10]) or order[10:]!=base[10:]:
                raise ValueError('Candidate membership changed')
            outputs[name].append(order)
        if outputs['full'][-1]!=saved['combined_head10'][i]:
            raise ValueError('Factorized formula fails to reproduce the frozen combination')
        before=metrics(base,gold);after=metrics(outputs['full'][-1],gold)
        transition=('kept_correct' if after['Top1'] else 'broken') if before['Top1'] else ('corrected' if after['Top1'] else 'still_wrong')
        ap_groups[transition].append(after['MAP@10']-before['MAP@10'])
        item={k:row[k] for k in ('dataset_row','repository','issue_id','before_fix_sha')}
        item.update(transition=transition,delta_ap=after['MAP@10']-before['MAP@10'],gold=row['gold'])
        if base:
            winner=outputs['full'][-1][0];incumbent=base[0]
            hit={n:bool(saved[n][i] and saved[n][i][0] in gold) for n in old_corrected}
            for n,correct in hit.items():
                if not before['Top1'] and correct:old_corrected[n].add(dataset_row)
                if before['Top1'] and not correct:old_broken[n].add(dataset_row)
            if transition in ('corrected','broken'):
                counts[transition+'_component_hits'][f"prior={hit['prior_head10']},packet={hit['packet_head10']}"]+=1
                counts[transition+'_promoted_original_rank'][base.index(winner)+1]+=1
                counts[transition+'_original_top1_new_rank'][outputs['full'][-1].index(incumbent)+1]+=1
                counts[transition+'_path_transitions'][f"{'test' if test_path(incumbent) else 'other'}_to_{'test' if test_path(winner) else 'other'}"]+=1
                delta={k:features[winner][k]-features[incumbent][k] for k in features[winner]}
                terms={'original_margin':delta['z'], 'packet_residual':.25*(delta['u']-delta['z']),
                       'original_retrieval':2*delta['a'],'hypothesis_retrieval':2*delta['b'],'structure':2*delta['e']}
                terms['total_margin']=sum(terms.values())
                terms['prior_total']=2*(delta['a']+delta['b']+delta['e'])
                if terms['total_margin']<=0:
                    raise ValueError('Promoted top1 must beat the old top1 under stable tie ordering')
                score_groups[transition].append(terms)
                item.update(old_top1=incumbent,new_top1=winner,new_top1_original_rank=base.index(winner)+1,
                    old_top1_new_rank=outputs['full'][-1].index(incumbent)+1,component_top1_correct=hit,
                    score_terms=terms,features={f:features[f] for f in {winner,incumbent}})
            counts['all_original_top1_kind']['test' if test_path(incumbent) else 'other']+=1
            counts['all_combined_top1_kind']['test' if test_path(winner) else 'other']+=1
        details.append(item)
    base_metrics=[metrics(o,r['gold']) for o,r in zip(saved['FULL_max'],rows)]
    variant_metrics={n:[metrics(o,r['gold']) for o,r in zip(order,rows)] for n,order in outputs.items()}
    summary={}
    for n,ms in variant_metrics.items():
        summary[n]={k:sum(m[k] for m in ms)/len(rows) for k in ms[0]}
        summary[n].update(corrected_top1=sum(a['Top1']>b['Top1'] for a,b in zip(ms,base_metrics)),
                          broken_top1=sum(a['Top1']<b['Top1'] for a,b in zip(ms,base_metrics)))
    stats={}
    for name in variants:
        if name=='full':continue
        stats['full_minus_'+name]=interval(rows,[a['MAP@10']-b['MAP@10'] for a,b in zip(variant_metrics['full'],variant_metrics[name])])
    previous=0.
    for i,name in enumerate(sorted(stats,key=lambda k:stats[k]['paired_randomization_p'])):
        previous=max(previous,min(1.,(len(stats)-i)*stats[name]['paired_randomization_p']))
        stats[name]['holm_p_exploratory_removals']=previous
    contributions={k:{'n':len(v),'sum_delta_ap':sum(v),'contribution_to_500_delta_map':sum(v)/500,
                       'mean_delta_ap_within_group':sum(v)/len(v)} for k,v in ap_groups.items()}
    terms_summary={g:{k:{'mean':sum(r[k] for r in rs)/len(rs),'positive':sum(r[k]>1e-12 for r in rs),
                       'zero':sum(abs(r[k])<=1e-12 for r in rs),'negative':sum(r[k]<-1e-12 for r in rs)}
                     for k in rs[0]} for g,rs in score_groups.items()}
    component_overlap={}
    for n in ('prior_head10','packet_head10'):
        new=variant_metrics['full'];old=[metrics(o,r['gold']) for o,r in zip(saved[n],rows)]
        component_overlap[n]={'new_correct_top1':sum(a['Top1']>b['Top1'] for a,b in zip(new,old)),
                              'new_broken_top1':sum(a['Top1']<b['Top1'] for a,b in zip(new,old)),
                              'retained_original_rescues':len(old_corrected[n]&old_corrected['combined_head10']),
                              'lost_original_rescues':sorted(old_corrected[n]-old_corrected['combined_head10'])}
    result={'n':500,'rank_membership_invariant':True,'top1_counts':{k:dict(v) for k,v in counts.items()},
            'ap_transition_contributions':contributions,'top1_score_margin_terms':terms_summary,
            'component_overlap':component_overlap,'fixed_weight_removal_summary':summary,
            'fixed_weight_removal_statistics':stats,'model_calls':0,
            'limits':['Test-path classification is heuristic','Score terms explain the implemented decision, not semantic reasoning inside the model',
                      'Removing fixed signals changes calibration; this is post-hoc diagnosis, not independently validated model selection',
                      'Packet scoring also changes formatting and numerical batching; cross-chunk reasoning is not isolated causally']}
    save(out/'analysis.json',result);save(out/'cases.json',details);save(out/'prior_sources_sha256.json',source_hashes)
    save(out/'counterfactual_rankings.json',{'dataset_rows':[r['dataset_row'] for r in rows],'orders':outputs})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.run.resolve(),args.output.resolve())
