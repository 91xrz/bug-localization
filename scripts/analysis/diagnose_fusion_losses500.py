"""Audit rank-level regression mechanisms without changing any model or ranking."""
import json
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from bug_localization.bm25 import ranking_metrics
from bug_localization.batch import write_json
from bug_localization.snapshot import sha256_file


def main():
    out=ROOT/'results/fusion_diagnosis/20261117_losses500';out.mkdir(parents=True,exist_ok=False)
    run=ROOT/'results/module_ablation/20261116_rrf500'
    samples=json.loads((run/'samples.json').read_text()); selection=json.loads((run/'selection.json').read_text())['samples']
    hashes={};cases=[];counts=Counter()
    for s in selection:
        i=s['position']-1
        if samples['B'][i]['status']!='ok':continue
        local=s['position'] if i<180 else s['position']-180
        files=[Path(samples[m][i]['source_root'])/'cases'/f'{local:03d}'/f for m,f in [('A','q0_result.json'),('B','direct_result.json')]]
        a,b=[json.loads(p.read_text()) for p in files]
        saved=json.loads((run/'cases'/f"{s['position']:03d}.json").read_text())
        for p in files:
            h=sha256_file(p);hashes[str(p.resolve())]=h
            assert saved['source_hashes'][str(p.resolve())]==h
        gold=set(a['ground_truth']);assert gold==set(b['ground_truth'])
        ar={r['file_path']:r['rank'] for r in a['ranking']};br={r['file_path']:r['rank'] for r in b['ranking']}
        assert ar.keys()==br.keys()
        pool={r['file_path'] for r in saved['candidates']}
        expected=sorted(pool,key=lambda p: (-(1/(60+ar[p])+1/(60+br[p])),p))
        assert expected==saved['ab_ranking']
        fr={p:i for i,p in enumerate(expected,1)}
        for method,rank in [('A',list(ar)),('B',list(br)),('A+B',expected)]:
            assert ranking_metrics(rank,gold,10)==samples[method][s['position']-1]['metrics']
        delta=samples['A+B'][i]['metrics']['average_precision_at_k']-samples['B'][i]['metrics']['average_precision_at_k']
        if abs(delta)<1e-12:counts['ties']+=1;continue
        if delta>0:counts['wins']+=1;continue
        counts['losses']+=1
        bgold={p for p in gold if br.get(p,999999)<=10}
        escaped=[p for p in bgold if fr.get(p,999999)>10]
        removed=bgold-pool
        counts['pool_removal_of_B_top10_gold']+=bool(removed)
        counts['some_gold_exits_top10']+=bool(escaped)
        counts['only_rank_or_within_top10_recall_change']+=not bool(escaped)
        counts['loses_all_top10_hits']+=bool(bgold) and not any(fr.get(p,999999)<=10 for p in gold)
        top1=list(br)[0]
        counts['B_correct_top1_demoted']+=top1 in gold and fr.get(top1,999999)>1
        counts['B_correct_top1_to_nongold_top1']+=top1 in gold and expected[0] not in gold
        details=[]
        for p in sorted(bgold,key=br.get):
            if fr[p]<=br[p]:continue
            blockers=[q for q in expected[:fr[p]-1] if q not in gold and br[q]>br[p]]
            details.append({'file':p,'A_rank':ar[p],'B_rank':br[p],'fusion_rank':fr[p],
                'nongold_overtakers':[{'file':q,'A_rank':ar[q],'B_rank':br[q],'fusion_rank':fr[q],
                  'rrf_A':1/(60+ar[q]),'rrf_B':1/(60+br[q])} for q in blockers[:10]]})
        cases.append({**s,'delta_AP':delta,'A_AP':samples['A'][i]['metrics']['average_precision_at_k'],
            'B_AP':samples['B'][i]['metrics']['average_precision_at_k'],'fusion_AP':samples['A+B'][i]['metrics']['average_precision_at_k'],
            'gold':sorted(gold),'gold_exiting_top10':escaped,'correct_B_top1_demoted':top1 in gold and fr.get(top1,999999)>1,
            'degraded_gold':details,'source_original':str(files[0]),'source_hypothesis':str(files[1])})
    cases.sort(key=lambda c:c['delta_AP'])
    write_json(out/'cases.json',cases)
    write_json(out/'summary.json',{'counts':dict(counts),'sum_negative_AP':sum(c['delta_AP'] for c in cases),
        'project_losses':dict(Counter(c['repository'] for c in cases)),
        'scope':'Rank-level attribution relative to repair-file labels; no claim that all non-gold files are semantically irrelevant'})
    hashes[str((run/'samples.json').resolve())]=sha256_file(run/'samples.json')
    hashes[str(Path(__file__).resolve())]=sha256_file(Path(__file__))
    write_json(out/'source_hashes.json',hashes)
    print(json.dumps(json.loads((out/'summary.json').read_text()),indent=2))
    print(json.dumps(cases[:4],ensure_ascii=False,indent=2))

if __name__=='__main__': main()
