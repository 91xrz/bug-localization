"""Audit IGSIL file-level outputs and compare with the frozen 500 baselines."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics

R = PROJECT / 'results'
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def compare(run, output):
    run, output = Path(run), Path(output)
    selection = read(PROJECT / 'configs/experiments/selections/crossproject_500_seed20261111.json')
    actual = read(run / 'selection.json')
    if selection['selection_sha256'] != actual['selection_sha256'] or selection['samples'] != actual['samples']:
        raise ValueError('IGSIL selection differs from the frozen 500')
    records = read(run / 'samples.json')
    if len(records) != 500:
        raise ValueError('IGSIL run is not complete')
    baseline = read(R / 'external_comparison/20261116_blaze_rrl_500/comparison.json')
    if baseline['selection_sha256'] != selection['selection_sha256']:
        raise ValueError('Comparison baseline selection differs')
    controls = {
        'BLAZE RRF': read(R / 'blaze-repair-20261116/merged500/blaze_rrf/samples.json'),
        'C3': read(R / 'chunk_ablation_reranking/20261112_C2_C3_500/C3/samples.json'),
        'RRL': read(R / 'rrl/crossproject500-cpu-retry-20261116/samples.json') +
               read(R / 'rrl/crossproject500-gpu1-remaining-20261116/samples.json')}
    fields = ('dataset_row', 'repository', 'issue_id', 'before_fix_sha')
    all_counts = Counter(); audited = 0
    for i, (entry, record) in enumerate(zip(selection['samples'], records), 1):
        if record['position'] != i or any(record[k] != entry[k] for k in fields):
            raise ValueError('IGSIL identity mismatch at ' + str(i))
        if record['status'] == 'ok':
            case = read(run / record['result_file'])
            ref = read(R / f'recall_baselines/20261112_L0_S0_H0_500/L0/{i:03d}/result.json')
            if case['query'] != ref['query_text'] or case['candidate_files'] != ref['candidate_files'] or case['gold_files'] != ref['ground_truth']:
                raise ValueError('IGSIL input/label mismatch at ' + str(i))
            if (not 0 < len(case['ranking']) <= 5 or len(set(case['ranking'])) != len(case['ranking']) or
                    not set(case['ranking']).issubset({x['file_path'] for x in case['candidate_files']})):
                raise ValueError('IGSIL output is not a unique valid Top-5 ranking')
            if ranking_metrics(case['ranking'], case['gold_files'], 10) != record['metrics']:
                raise ValueError('IGSIL metric mismatch at ' + str(i))
            audited += 1
        else:
            all_counts[(record['error_type'], record['error'])] += 1
    summary = aggregate_results(records, 10, 500)
    pairs = {}
    for name, rows in controls.items():
        if len(rows) != 500:
            raise ValueError('Incomplete control: ' + name)
        counts, common = Counter(), Counter()
        for rec, other in zip(records, rows):
            if any(rec[k] != other[k] for k in fields):
                raise ValueError('Control identity differs: ' + name)
            a = rec['metrics']['average_precision_at_k'] if rec['status'] == 'ok' else 0
            b = other['metrics']['average_precision_at_k'] if other['status'] == 'ok' else 0
            label = 'win' if a-b > 1e-12 else 'loss' if a-b < -1e-12 else 'tie'
            counts[label] += 1
            if rec['status'] == other['status'] == 'ok': common[label] += 1
        pairs['IGSIL vs ' + name] = {'all500_failed_zero': dict(counts), 'both_success': dict(common),
                                    'both_success_count': sum(common.values())}
    output.mkdir(parents=True, exist_ok=False)
    result = {'selection_sha256': selection['selection_sha256'], 'igsil': summary,
              'baseline_500': baseline['cohorts']['all500'], 'paired': pairs,
              'audited_successful_reports': audited,
              'failure_counts': [{'type': k[0], 'error': k[1], 'count': n} for k,n in all_counts.most_common()],
              'limitations': ['Author file-level Top-5 output is not padded to 10',
                              'Qwen 7B Q4_K_M llama.cpp variant; not original vLLM precision',
                              'Shared candidate protocol retains test files unlike author script',
                              '500 samples are a development set, not an independent blind test']}
    write_json(output / 'comparison.json', result)
    print(json.dumps({'IGSIL': summary, 'paired': pairs, 'audited_successful_reports': audited}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    compare(a.run, a.output)
