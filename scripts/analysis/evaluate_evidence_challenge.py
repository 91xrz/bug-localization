"""Evaluate frozen decisions separately from inference for both report cohorts."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from pilot_perspective_rank import BL,C3,paired,paths_from_saved,ranking_metrics,read,sha256_file,write_json
from pilot_evidence_challenge import ROOT,get_cohort,promote


def evaluate(name):
    samples,arc,proposals=get_cohort(name)
    out=ROOT/name
    assert read(out/'selection.json')['samples']==samples
    cs={s['position']:s for s in read(C3/'samples.json')}
    bs={s['dataset_row']:s for s in read(BL/'samples.json')}
    rows=[]
    costs=defaultdict(Counter)
    stages=Counter()
    for s in samples:
        key=f'{s["position"]:04d}'
        case=out/'cases'/key
        trace=read(case/'decisions.json')
        blind=read(case/'blind_decision.json')
        trace['promote'].update(blind=blind['promote'],blind_consensus=blind['consensus'])
        if blind['status']!='not_triggered':
            trace['stages']['blind']=blind
        line=read(case/'line_decision.json')
        trace['promote'].update(line=line['promote'],line_strict=line['strict'])
        if line['status']!='not_triggered':
            trace['stages']['line']=line
        c,b=cs[s['position']],bs[s['dataset_row']]
        assert all(c[k]==s[k]==b[k] for k in ('repository','issue_id','before_fix_sha','dataset_row'))
        assert c['status']==b['status']=='ok'
        data=read(C3/c['result_file'])
        pool=paths_from_saved(data)
        gold=data['ground_truth']
        valid=trace['archive_status']=='ok'
        original=read(arc/'cases'/key/'result.json')['ranking'] if valid else pool
        assert original[:10]==trace['prior'] and original[10:]==pool[10:]
        assert set(original[:10])==set(pool[:10])
        assert trace['evidence_sha256']==sha256_file(arc/'cases'/key/'inference_input.json')
        assert trace['proposal_sha256']==sha256_file(proposals/'cases'/key/'rankings.json')
        ch=trace['challenger']
        assert trace['promote']['unconditional']==(valid and original[0]!=ch)
        metrics={m:ranking_metrics(promote(original,ch) if move else original,gold,10)
                 for m,move in trace['promote'].items()}
        assert all(m['recall_at_k']==metrics['original']['recall_at_k'] for m in metrics.values())
        metrics['BLAgent']=ranking_metrics(read(BL/b['result_file'])['ranking'],gold,10)
        metrics['C3']=ranking_metrics(pool,gold,10)
        rows.append({**s,'incumbent':original[0],'challenger':ch,'promote':trace['promote'],
                     'archive_status':trace['archive_status'],'metrics':metrics})
        for stage,info in trace['stages'].items():
            stages[f'{stage}:{info["status"]}']+=1
            path=case/stage/'call_001.json'
            if path.exists():
                call=read(path)
                costs[stage].update(calls=1,input_tokens=call['input_tokens'],output_tokens=call.get('output_tokens',0))
            if stage=='evidence' and info['answer'].get('decision')=='overturn' and not all(info['anchors'].values()):
                stages['invalid_override_citations']+=1
    summary={}
    for method in rows[0]['metrics']:
        metrics=[r['metrics'][method] for r in rows]
        result={k:fmean(m[k] for m in metrics) for k in ('average_precision_at_k','reciprocal_rank_at_k')}
        result['top1']=sum(m['top_1'] for m in metrics)
        if method in rows[0]['promote']:
            result['changed']=sum(r['promote'][method] for r in rows)
            result.update(paired(rows,method,'original'))
        summary[method]=result
    report=dict(cohort=name,n=len(rows),baseline_fallbacks=sum(r['archive_status']!='ok' for r in rows),
                summary=summary,extra_review_cost=costs,stage_counts=stages,
                verified_vs_BLAgent=paired(rows,'verified','BLAgent'))
    write_json(out/'paired_cases.json',rows)
    write_json(out/'summary.json',report)
    return report


if __name__=='__main__':
    for name in sys.argv[1:] or ('replay90','transfer45'):
        print(json.dumps(evaluate(name),indent=2))
