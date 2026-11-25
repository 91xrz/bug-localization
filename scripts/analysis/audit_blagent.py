"""Independently audit complete BLAgent runs and compare the frozen 500 cohort."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.embeddings import digest
from bug_localization.snapshot import sha256_file


def read(path): return json.loads(path.read_text())


def audit(folder):
    selection=read(folder/'selection.json')
    records=read(folder/'samples.json')
    summary=read(folder/'summary.json')
    if len(records)!=len(selection['samples']) or summary['pending_samples']:
        raise ValueError('Incomplete experiment; cannot report final metrics')
    replay_path=folder/'replay_source.json'
    if replay_path.exists():
        replay=read(replay_path)
        source=Path(replay['source'])
        if sha256_file(source/'provenance.json')!=replay['source_provenance_sha256']:
            raise ValueError('Replay source provenance changed')
        for relative, checksum in replay['files'].items():
            if sha256_file(folder/relative)!=checksum or sha256_file(source/relative)!=checksum:
                raise ValueError('Replayed raw model response was modified')
    usage=Counter(); tool_count=0
    for rec, sample in zip(records,selection['samples']):
        for field in ('position','dataset_row','repository','issue_id','before_fix_sha'):
            if rec[field]!=sample[field]: raise ValueError('Sample identity mismatch')
        case=folder/'cases'/f'{rec["position"]:04d}'
        calls=sorted(case.glob('call_*.json'))
        for path in calls:
            call=read(path)
            if call['request_sha256']!=digest(call['request']): raise ValueError('Call request hash mismatch')
            usage['input_tokens']+=call.get('input_tokens',0)
            usage['output_tokens']+=call.get('output_tokens',0)
            usage['calls']+=1
            # No structured ground-truth fields are passed into model requests.
            if set(call['request'])-{'response_schema'}!={'messages','temperature','stop','stage','max_new_tokens','identity'}:
                raise ValueError('Unexpected inference request fields')
            if 'response_schema' in call['request']:
                candidates=read(case/'retrieval.json')['candidates']
                expected={'type':'object','properties':{p:{'type':'number'} for p in candidates},'additionalProperties':False}
                if call['request']['stage']=='EAR':
                    from bug_localization.blagent_recovery import permutation_schema
                    schema=call['request']['response_schema']
                    paths=schema['properties']['ranked_files']['enum'][0]
                    if schema!=permutation_schema(paths) or not set(paths).issubset(read(case/'phase1.json')['ranking'][:5]):
                        raise ValueError('EAR schema is not exactly the allowed candidate permutations')
                elif call['request']['stage']!='SAS_repair' or call['request']['response_schema']!=expected:
                    raise ValueError('Correction schema differs from the exact candidate pool')
        if rec['status']=='failed':
            if not (case/'error.json').exists(): raise ValueError('Failure missing diagnostic')
            continue
        result=read(case/'result.json')
        if sha256_file(case/'result.json')!=rec['result_sha256']: raise ValueError('Result checksum mismatch')
        inputs=read(case/'inputs.json');retrieval=read(case/'retrieval.json')
        if result['query']!=inputs['query'] or result['candidate_files']!=inputs['candidate_files']:
            raise ValueError('Inference/evaluation inputs differ')
        ranking=result['ranking']; phase1=result['phase1_ranking']
        pool=list(dict.fromkeys(retrieval['rankings'][0][:15]+retrieval['rankings'][1][:15]))
        if pool!=result['candidates'] or not set(ranking).issubset(pool): raise ValueError('Candidate membership differs')
        if len(ranking)!=len(set(ranking)) or ranking[5:]!=phase1[5:] or set(ranking[:5])!=set(phase1[:5]):
            raise ValueError('EAR is not a prefix permutation')
        for key, paths in (('metrics',ranking),('phase1_metrics',phase1)):
            if result[key]!=ranking_metrics(paths,result['gold_files'],10): raise ValueError('Metric mismatch')
        if result['metrics']!=rec['metrics']: raise ValueError('Record metrics mismatch')
        inspected=read(case/'tool_calls.json')
        if not inspected or any(c['file_path'] not in pool for c in inspected): raise ValueError('Invalid tool inspection')
        tool_count+=len(inspected)
        stages=[read(p)['request']['stage'] for p in calls]
        if stages[:2]!=['query','query'] or stages[-1]!='EAR' or 'SAS' not in stages:
            raise ValueError('Required pipeline stages missing')
        if (case/'ear_strategy.json').exists():
            strategy=read(case/'ear_strategy.json')
            comparisons=read(case/'ear_comparisons.json')
            for comparison in comparisons:
                paths=comparison['paths'];ranked=comparison.get('ranking',[])
                if len(ranked)!=len(paths) or set(paths)!=set(ranked) or not set(paths).issubset(phase1[:5]):
                    raise ValueError('Invalid EAR subranking')
            if strategy['mode']=='pairwise_merge' and set(strategy['all_files_preserved'])!=set(phase1[:5]):
                raise ValueError('EAR overflow handling lost candidates')
        if (case/'tool_errors.json').exists():
            for event in read(case/'tool_errors.json'):
                if event['error']=='outside_candidates' and event['requested_path'].strip().strip('"').strip("'").strip() in pool:
                    raise ValueError('Valid path incorrectly denied')
    recalculated=aggregate_results(records,10,len(selection['samples']))
    if recalculated['metrics']!=summary['metrics']: raise ValueError('Aggregate metrics mismatch')
    result={'status':'passed','samples':len(records),'successful':summary['successful_samples'],
            'failed':summary['failed_samples'],'metrics':summary['metrics'],'usage':dict(usage),
            'actual_skeleton_calls':tool_count,'selection_sha256':selection['selection_sha256']}
    if replay_path.exists():
        result['replayed_model_calls']=len(replay['files'])
        result['new_model_calls']=usage['calls']-len(replay['files'])
        result['timing_note']='Contains unchanged cached model responses; case elapsed times are not fresh inference latency.'
    if len(records)==500:
        full=read(PROJECT/'configs/experiments/selections/crossproject_500_seed20261111.json')
        if full!=selection and full['selection_sha256']!=selection['selection_sha256']:
            raise ValueError('Not the frozen 500 cohort')
        # Verify full candidate inventories, queries and labels against shared BM25 inputs.
        for rec in records:
            case=folder/'cases'/f'{rec["position"]:04d}'
            path=case/'inputs.json'
            if not path.exists(): continue
            inputs=read(path)
            baseline=read(PROJECT/'results/recall_baselines/20261112_L0_S0_H0_500/L0'/f'{rec["position"]:03d}'/'result.json')
            if inputs['query']!=baseline['query_text'] or inputs['candidate_files']!=baseline['candidate_files']:
                raise ValueError('Shared baseline inputs/candidates mismatch')
            if rec['status']=='ok' and read(case/'result.json')['gold_files']!=baseline['ground_truth']:
                raise ValueError('Shared baseline labels mismatch')
        prior=read(PROJECT/'results/hypothesis_reranking/20261111_153819_040009_500/paired_samples.json')['cohorts']['all500']
        baselines={'original_reranked':prior['original_reranked'],
                   'hypothesis_reranked':prior['hypothesis_reranked'],
                   'C3':read(PROJECT/'results/chunk_ablation_reranking/20261112_C2_C3_500/C3/samples.json'),
                   'A+B+C_RRF':read(PROJECT/'results/module_ablation/20261116_rrf500/samples.json')['A+B+C']}
        new=read(PROJECT/'configs/experiments/selections/crossproject_new320_seed20261111.json')
        key=lambda r:(r['dataset_row'],r['repository'],r['issue_id'],r['before_fix_sha'])
        for label, rows in baselines.items():
            if len(rows)!=500 or {key(r) for r in rows}!={key(r) for r in records}:
                raise ValueError('Baseline does not have the identical 500 sample identities: '+label)
        new_keys={key(r) for r in new['samples']}
        cohorts={'all500':records,'new320':[r for r in records if key(r) in new_keys],
                 'previous180':[r for r in records if key(r) not in new_keys]}
        result['cohorts']={}
        for name, rows in cohorts.items():
            group={'BLAgent_adaptation':aggregate_results(rows,10,len(rows))['metrics']}
            for label, other in baselines.items():
                lookup={key(r):r for r in other}
                paired=[lookup[key(r)] for r in rows]
                group[label]=aggregate_results(paired,10,len(paired))['metrics']
                ap=lambda r:r['metrics']['average_precision_at_k'] if r['status']=='ok' else 0
                diffs=[ap(a)-ap(b) for a,b in zip(rows,paired)]
                group['BLAgent_vs_'+label]={'wins':sum(d>1e-12 for d in diffs),
                    'ties':sum(abs(d)<=1e-12 for d in diffs),'losses':sum(d< -1e-12 for d in diffs)}
            result['cohorts'][name]=group
        result['projects']={repo:aggregate_results([r for r in records if r['repository']==repo],10)['metrics']
                            for repo in sorted({r['repository'] for r in records})}
        fields=['cohort','method','MAP@10','MRR@10','Top-1','Top-5','Top-10','MeanRecall@10']
        with (folder/'comparison.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for cohort, methods in result['cohorts'].items():
                for label, metrics in methods.items():
                    if 'MAP@10' in metrics:
                        writer.writerow({'cohort':cohort,'method':label,**metrics})
    if len(records)==2294:
        full=read(PROJECT/'configs/experiments/selections/swe_full2294_20261118.json')
        if selection!=full or [r['dataset_row'] for r in records]!=list(range(2294)):
            raise ValueError('Full SWE cohort must contain every original row exactly once')
        development=read(PROJECT/'configs/experiments/selections/crossproject_500_seed20261111.json')
        development_rows={r['dataset_row'] for r in development['samples']}
        result['cohorts']={name:aggregate_results(rows,10,len(rows))['metrics'] for name,rows in {
            'all2294':records,'development500':[r for r in records if r['dataset_row'] in development_rows],
            'remaining1794':[r for r in records if r['dataset_row'] not in development_rows]}.items()}
        result['projects']={repo:aggregate_results([r for r in records if r['repository']==repo],10)['metrics']
                            for repo in sorted({r['repository'] for r in records})}
        result['cohort_note']='The remaining 1794 are not claimed to be an independent blind test.'
    write_json(folder/'verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();print(json.dumps(audit((PROJECT/args.run).resolve()),indent=2))
