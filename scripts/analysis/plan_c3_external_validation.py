"""Freeze a prospective A/C3 protocol and outcome-blind candidate pool; do not evaluate it."""
import ast
from collections import Counter
import hashlib
import math
from pathlib import Path
import random
from statistics import NormalDist
import sys
import numpy as np
import pyarrow.parquet as pq
from datasets import load_from_disk
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src'))
from bug_localization.batch import write_json
from bug_localization.reranking import read_json
from bug_localization.snapshot import sha256_file


def frozen_write(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError('Refusing to replace frozen planning artifact: ' + str(path))
    write_json(path, value)


def norm(body): return ' '.join((body or '').split()).casefold()

def main():
    out=PROJECT/'results/c3_failure_attribution/20261112/next_validation';out.mkdir(parents=True,exist_ok=True)
    previous=[load_from_disk(str(PROJECT/'datasets'/name))['test'] for name in ('swe_bench_formatted','lca_py_formatted')]
    used_projects={r['repo_name'].casefold() for d in previous for r in d}
    used_bodies={norm(r['body']) for d in previous for r in d}
    used_issues={str(r['issue_url']).rstrip('/').casefold() for d in previous for r in d}
    evidence={}
    for p in sorted((PROJECT/'results').rglob('selection.json')):
        s=read_json(p)
        if isinstance(s,dict) and 'samples' in s:
            used_projects.update(r['repository'].casefold() for r in s['samples'] if 'repository' in r)
            evidence[str(p)]=sha256_file(p)
    raw=PROJECT/'datasets/LCA_bug_localization/py/train-00000-of-00001.parquet'
    cols=['id','text_id','repo_owner','repo_name','repo_language','base_sha','head_sha','issue_url','pull_url','issue_body','changed_files']
    rows=pq.read_table(raw,columns=cols).to_pylist(); candidates=[]; rejected=Counter();seen_issues=set();seen_bodies=set()
    for r in rows:
        repo=(r['repo_owner']+'/'+r['repo_name']).casefold();issue=r['issue_url'].rstrip('/').casefold();body=norm(r['issue_body'])
        if repo in used_projects: rejected['previous_project']+=1;continue
        if r['repo_language']!='Python': rejected['not_Python_primary_language']+=1;continue
        if not body: rejected['empty_report']+=1;continue
        if body in used_bodies or issue in used_issues: rejected['historical_body_or_issue']+=1;continue
        if body in seen_bodies or issue in seen_issues: rejected['duplicate_body_or_issue_in_pool']+=1;continue
        labels=ast.literal_eval(r['changed_files'])
        # Keep original labels, including tests/missing/non-Python paths: no outcome-driven cleaning.
        if not labels or not any(str(p).endswith('.py') for p in labels): rejected['no_Python_file_label']+=1;continue
        seen_bodies.add(body);seen_issues.add(issue)
        candidates.append({'raw_id':str(r['id']),'text_id':r['text_id'],'repository':repo,'issue_url':r['issue_url'],
            'pull_url':r['pull_url'],'before_fix_sha':r['base_sha'],'after_fix_sha':r['head_sha'],
            'body_sha256':hashlib.sha256(r['issue_body'].encode()).hexdigest(),
            'labels_sha256':hashlib.sha256(r['changed_files'].encode()).hexdigest(),
            'local_repository_available':(PROJECT/'datasets/LCA_bug_localization/repositories/py'/(r['repo_owner']+'__'+r['repo_name'])).exists()})
    dev_a=read_json(PROJECT/'results/mixed_recall_reranking/20261111_CD_500/paired_samples.json')['cohorts']['all500']['A']
    dev_c=read_json(PROJECT/'results/chunk_ablation_reranking/20261112_C2_C3_500/C3/samples.json')
    ext=PROJECT/'results/c3_supplement/20261112_lca_py_holdout'
    deviations={}
    for label,a,c in [('development500',dev_a,dev_c),('external47',read_json(ext/'A/samples.json'),read_json(ext/'C3/samples.json'))]:
        d=[y.get('metrics',{}).get('average_precision_at_k',0)-x.get('metrics',{}).get('average_precision_at_k',0) for x,y in zip(a,c)]
        deviations[label]=float(np.std(d,ddof=1))
    sigma=max(deviations.values());z=NormalDist().inv_cdf(.975)+NormalDist().inv_cdf(.8)
    power=[]
    for mde in (.01,.02,.03):
        for rho in (0,.05,.1,.2):
            power.append({'absolute_MAP_gain':mde,'assumed_within_project_ICC':rho,'max_reports_per_project':5,
                'paired_sd':sigma,'design_effect':1+4*rho,'approximate_required_reports':math.ceil((z*sigma/mde)**2*(1+4*rho))})
    target=800;pool_by_repo={}
    for c in candidates:pool_by_repo.setdefault(c['repository'],[]).append(c)
    for repo,cs in pool_by_repo.items():random.Random('20261112:'+repo).shuffle(cs)
    project_order=sorted(pool_by_repo);random.Random(20261112).shuffle(project_order)
    reserve=[]
    for round_index in range(5):
        for repo in project_order:
            if len(reserve)>=target:break
            if len(pool_by_repo[repo])>round_index:reserve.append(pool_by_repo[repo][round_index])
    method_files=[PROJECT/'configs/retrieval/reranker.json', PROJECT/'configs/experiments/report_hypothesis_crossproject180.json',
        PROJECT/'scripts/experiments/run_direct_hypothesis.py',PROJECT/'src/bug_localization/chunk_ablation.py',
        PROJECT/'src/bug_localization/mixed_recall.py',PROJECT/'src/bug_localization/reranking.py',
        PROJECT/'src/bug_localization/embeddings.py',PROJECT/'src/bug_localization/ast_chunking.py',
        PROJECT/'results/report_hypothesis/20261110_023926_805713_retrieval/effective_config.json',
        PROJECT/'results/c3_supplement/20261112_lca_py_holdout/generator_identity.json',
        PROJECT/'results/c3_supplement/20261112_lca_py_holdout/reranker_identity.json']
    protocol={'status':'method_and_sampling_rules_frozen_not_ready_to_run','created_date':'2026-11-12',
        'primary_metric':'MAP@10','primary_contrast':'C3 minus A','minimum_practical_absolute_gain':.02,
        'alpha_two_sided':.05,'target_power':.8,'target_reports':target,'max_reports_per_project':5,'minimum_distinct_projects':160,
        'planning_paired_sd':sigma,'planning_ICC':.1,'planning_note':'Normal approximation with design effect 1+(m-1)*ICC; neither empirical power guarantee nor evidence that ICC=0.1. 800 is a round-up of ~749. Failed reports remain in all-report estimand, not removed as attrition.',
        'primary_inference':'Report-weighted paired mean; two-sided project-level paired sign randomization (100000 replicates), project-cluster bootstrap 95% CI (20000). Sign randomization assumes exchangeable symmetric cluster contrasts. Single primary comparison: no multiplicity correction on primary. Secondary 5 metrics Holm corrected; projects exploratory.',
        'success_rule':'Statistical evidence requires primary p<0.05 and positive effect. Practical adequacy judged separately against +0.02 and CI; power is for rejecting zero at true +0.02, not for proving a lower CI bound above +0.02.',
        'failure_policy':'Every frozen report stays. Method failures count zero; A runs independently if hypothesis fails. No fallback, replacements, silent query truncation beyond frozen encoder policy, or outcome filtering. Infrastructure retries logged and fixed at at most one identical retry.',
        'methods':{'A':'original Top50 + original Top3 chunks','C3':'hypothesis Top40 then original unique padding to50 + original Top1 chunk then hypothesis unique padding to3'},
        'fixed_inputs':'Original body only; original-report reranker query; same AST64 and model/runtime/token settings as LCA47; preserve full original labels.',
        'sample_source':'LCA py TRAIN only as prospective external evaluation candidates; DEV overlaps train/test and must not be concatenated. These labels are not equivalent to the manually verified test split.',
        'sample_status':'reserve identities frozen for acquisition, NOT final evaluation selection; zero model scoring performed',
        'source_and_label_gate':'Acquire verified base commits read-only. Before predictions, independent human review confirms issue-to-PR linkage, base SHA and label validity. Preserve raw labels; do not repair them using model outcomes. If eligibility/data quality changes the pool, version the protocol and freeze final manifest before any scoring. Log all exclusions; retain negative/hard cases.',
        'unseen_gate':'Exclude every previously evaluated project, canonical issue, normalized report body, duplicate issue/body, and any newly discovered historical overlap. Do not equate unpublished model pretraining exposure with experimental non-use.',
        'quality_audit':'Blind to A/C3 predictions; at least 50 randomly selected reserve reports across projects reviewed before go/no-go; discrepancies trigger documented audit, not selective replacement.',
        'stopping_rule':'No peeking, interim strategy selection, or optional stopping; analyze once the frozen sample is fully attempted. New strategy uses a future untouched cohort.',
        'timing':'Record cold preparation separately; online A/C3 time and token counts per report; same device/runtime; paired randomized method order; fixed 30-report latency subset chosen before outcomes, three repeats. Currency only with supplied price.',
        'seed':20261112,'method_files_sha256':{str(p):sha256_file(p) for p in method_files},
        'raw_candidate_source_sha256':sha256_file(raw),'planned_execution':'Do not start until source/label gates, sample size, disjointness and final manifest pass. No Java migration or cost optimization in this round.'}
    frozen_write(out/'candidate_pool.json',{'candidates':candidates,'rejected_counts':dict(rejected),'historical_selection_sha256':evidence})
    frozen_write(out/'acquisition_reserve.json',{'status':'not_test_results_not_final_selection','requested':target,'selected':len(reserve),
        'projects':len({r['repository'] for r in reserve}),'max_per_project':max(Counter(r['repository'] for r in reserve).values()),'samples':reserve})
    frozen_write(out/'power_analysis.json',{'observed_paired_sd':deviations,'sensitivity':power,
        'sigma_sensitivity_at_2pp_ICC_0_1':{str(s):math.ceil((z*s/.02)**2*1.4) for s in (.1384,.1651,.2,.25)},
        'warning':'47 external observations yield uncertain variance; clustering and dataset shift can make this approximation optimistic.'})
    dest=PROJECT/'configs/experiments/c3_external_python_next.json'
    if dest.exists() and read_json(dest)!=protocol:raise ValueError('Refuse overwrite frozen protocol')
    write_json(dest,protocol)
    frozen_write(out/'readiness.json',{'status':'blocked_on_source_acquisition_and_blinded_label_review','candidate_reports':len(candidates),
        'candidate_projects':len(pool_by_repo),'capacity_at_max5':sum(min(5,len(c)) for c in pool_by_repo.values()),
        'reserve_reports':len(reserve),'reserve_projects':len({r['repository'] for r in reserve}),
        'local_repositories_for_reserve':sum(c['local_repository_available'] for c in reserve),'no_model_evaluation_performed':True})
    print(read_json(out/'readiness.json'))

if __name__=='__main__':main()
