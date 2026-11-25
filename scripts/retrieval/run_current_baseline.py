"""Replay the registered fixed initial-ranking baseline, without parameter search."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from bug_localization.batch import aggregate_results
from bug_localization.bm25 import ranking_metrics
from bug_localization.lightweight_hypothesis import rank_variants

REGISTRY = ROOT/'configs/baselines/current.json'
IDENTITY = ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha',
            'query_sha256', 'labels_sha256')


def read(path):
    return json.loads(path.read_bytes())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def load_current():
    registry = read(REGISTRY)
    path = ROOT/registry['initial_ranking']['config']
    if sha256(path) != registry['initial_ranking']['sha256']:
        raise ValueError('Registered baseline configuration changed')
    config = read(path)
    for relative, expected in config['files_sha256'].items():
        if sha256(ROOT/relative) != expected:
            raise ValueError('Frozen baseline dependency changed: '+relative)
    if (config['alpha'], config['anchor_weight'], config['top_n_files'], config['method']) != (
            .5, .5, 50, 'score_a0.5_e0.5'):
        raise ValueError('Unexpected settings for the frozen baseline version')
    return config, registry['initial_ranking']['sha256']


def rank_fixed(inputs, config):
    """Use prepared, label-free A/B rankings and shared-union anchor features."""
    a, b, anchor = inputs['a'], inputs['b'], inputs['anchor']
    for rows in (a,b):
        if not rows or [r['rank'] for r in rows] != list(range(1,len(rows)+1)):
            raise ValueError('Require complete consecutive rankings')
        if len({r['file_path'] for r in rows}) != len(rows):
            raise ValueError('Duplicate candidate file')
    union = {r['file_path'] for rows in (a[:50],b[:50]) for r in rows}
    if set(anchor) != union or any(not 0 <= v <= 1 for v in anchor.values()):
        raise ValueError('Anchor scores must cover exactly A50 union B50 and lie in [0,1]')
    return rank_variants(a,b,anchor,[config['alpha']],[config['anchor_weight']])[config['method']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New output directory; never overwrite a frozen run')
    parser.add_argument('--check-only', action='store_true', help='Validate the registered configuration and provenance')
    args = parser.parse_args()
    config, config_sha = load_current()
    if args.check_only:
        print(json.dumps({'status':'passed','baseline':config['id'],'method':config['method'],
                          'scope':config['scope']},ensure_ascii=False))
        return
    study = ROOT/config['reference_study']
    output = args.output or ROOT/config['reference_output']
    output.mkdir(parents=True, exist_ok=False)
    selection = read(study/'selection.json')
    audit = read(study/'independent_audit.json')
    if audit['status'] != 'passed' or read(study/'verification.json')['status'] != 'passed':
        raise ValueError('Reference study has not passed verification')
    if read(study/'frozen_choice.json')['method'] != config['method']:
        raise ValueError('Frozen development selection does not match the registered baseline')
    save(output/'protocol.json', {'baseline':config,'config_sha256':config_sha,
         'registry_sha256':sha256(REGISTRY),'mode':'fixed replay; no parameter selection or model calls'})
    save(output/'selection.json',selection)
    samples, checkpoints = [], {}
    for meta in selection['samples']:
        case_path = study/'cases'/f"{meta['position']:04d}.json"
        if sha256(case_path) != audit['case_files_sha256'][case_path.name]:
            raise ValueError('Reference case checksum changed')
        case = read(case_path)
        if any(meta[k] != case[k] for k in IDENTITY):
            raise ValueError('Case identity mismatch')
        reference = case['methods'][config['method']]
        input_path = study/'inputs'/case_path.name
        hashes = {'case':sha256(case_path)}
        if case['status'] == 'ok':
            inputs = read(input_path)
            if any(meta[k] != inputs['sample'][k] for k in IDENTITY):
                raise ValueError('Prepared input identity mismatch')
            paths = rank_fixed(inputs, config)
            inventory = {r['file_path'] for r in inputs['a']}
            missing = sorted(set(case['ground_truth'])-inventory)
            hashes['prepared_inputs'] = sha256(input_path)
            status = 'ok'
        elif case['status'] == 'B_failed':
            paths, missing, status = [], [], 'failed'
        else:
            raise ValueError('Unknown reference status')
        if paths != reference['ranking']:
            raise ValueError('Fixed baseline differs from validated study')
        at10 = ranking_metrics(paths, case['ground_truth'],10)
        at50 = ranking_metrics(paths, case['ground_truth'],50)
        for target, actual in {
            'MAP@10':at10['average_precision_at_k'], 'MRR@10':at10['reciprocal_rank_at_k'],
            'Hit@1':at10['top_1'], 'Hit@10':at10['top_10'],
            'MeanRecall@10':at10['recall_at_k'], 'Hit@50':int(at50['retrieved_gold_files']>0),
            'MeanRecall@50':at50['recall_at_k']}.items():
            if abs(actual-reference['metrics'][target]) > 1e-12:
                raise ValueError('Frozen metrics disagree: '+target)
        result_file = f"cases/{meta['position']:04d}/result.json"
        result = {**meta,'baseline_id':config['id'],'status':status,'ground_truth':case['ground_truth'],
                  'ranking':[{'rank':i,'file_path':p} for i,p in enumerate(paths,1)],
                  'metrics':at10,'metrics_at_50':at50,'missing_gold':missing,
                  'source_hashes':hashes}
        if status == 'failed':
            result.update(error_type='DependencyFailure',error=case.get('error'))
        save(output/result_file,result)
        samples.append({k:v for k,v in result.items() if k not in ('ranking','source_hashes')} |
                       {'result_file':result_file,'result_sha256':sha256(output/result_file)})
        checkpoints[str(meta['position'])] = hashes
    save(output/'samples.json',samples)
    development = set(selection['development_rows'])
    cohorts = {'all2294':samples,
               'development500':[r for r in samples if r['dataset_row'] in development],
               'remaining1794':[r for r in samples if r['dataset_row'] not in development]}
    summary = {}
    for name, rows in cohorts.items():
        item = aggregate_results(rows,10,len(rows))
        item['candidate_metrics'] = {
            'Hit@50':sum(r['metrics_at_50']['retrieved_gold_files']>0 for r in rows)/len(rows),
            'MeanRecall@50':sum(r['metrics_at_50']['recall_at_k'] for r in rows)/len(rows)}
        summary[name] = item
    save(output/'summary.json',summary)
    save(output/'source_checkpoints.json',checkpoints)
    load_current()  # Reject any dependency change during the replay.
    save(output/'verification.json',{'status':'passed','baseline_id':config['id'],'reports':len(samples),
         'successful':sum(r['status']=='ok' for r in samples),'failed':sum(r['status']=='failed' for r in samples),
         'model_calls':0,'checks':['registered configuration and dependencies pinned by SHA256',
         'every reference case verified against independent audit', 'every successful fixed ranking exactly reproduced',
         'Top10/Top50 metrics recomputed', 'failure retained without fallback']})
    print(json.dumps({'status':'passed','baseline':config['id'],'output':str(output),'reports':len(samples)},ensure_ascii=False))


if __name__ == '__main__':
    main()
