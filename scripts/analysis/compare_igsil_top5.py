"""Recompute comparable @5 metrics and paired AP for the IGSIL experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import write_json
from bug_localization.bm25 import ranking_metrics

R = PROJECT / 'results'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main(output):
    selection = read(PROJECT / 'configs/experiments/selections/crossproject_500_seed20261111.json')
    ig = R / 'igsil/crossproject500-20261116'
    if read(ig / 'selection.json')['samples'] != selection['samples']:
        raise ValueError('IGSIL frozen selection mismatch')
    cpu = R / 'rrl/crossproject500-cpu-retry-20261116'
    gpu = R / 'rrl/crossproject500-gpu1-remaining-20261116'
    bl = R / 'blaze-repair-20261116/merged500'
    c3 = R / 'chunk_ablation_reranking/20261112_C2_C3_500/C3'
    groups = {'IGSIL': read(ig / 'samples.json'), 'BLAZE RRF': read(bl / 'blaze_rrf/samples.json'),
              'RRL': read(cpu / 'samples.json') + read(gpu / 'samples.json'), 'C3': read(c3 / 'samples.json')}
    fields = ('dataset_row', 'repository', 'issue_id', 'before_fix_sha')
    calculated = {name: [] for name in groups}
    audit = {'report_identity_checked': 0, 'gold_and_metrics_checked': {name: 0 for name in groups}}
    for i, entry in enumerate(selection['samples'], 1):
        gold = read(R / f'recall_baselines/20261112_L0_S0_H0_500/L0/{i:03d}/result.json')['ground_truth']
        for name, rows in groups.items():
            rec = rows[i-1]
            if any(rec[k] != entry[k] for k in fields):
                raise ValueError(f'{name} identity mismatch at {i}')
            if rec['status'] != 'ok':
                calculated[name].append(None)
                continue
            path = (ig / rec['result_file'] if name == 'IGSIL' else
                    Path(rec['result_file']) if name == 'BLAZE RRF' else
                    (cpu if i <= 3 else gpu) / rec['result_file'] if name == 'RRL' else c3 / rec['result_file'])
            case = read(path)
            actual_gold = case['ground_truth'] if name == 'C3' else case['gold_files']
            if actual_gold != gold:
                raise ValueError(f'{name} gold mismatch at {i}')
            raw = case['routes']['blaze_rrf']['ranking'] if name == 'BLAZE RRF' else case['ranking']
            ranking = [x['file_path'] if isinstance(x, dict) else x for x in raw]
            if ranking_metrics(ranking, gold, 10) != rec['metrics']:
                raise ValueError(f'{name} @10 metrics mismatch at {i}')
            calculated[name].append(ranking_metrics(ranking, gold, 5))
            audit['gold_and_metrics_checked'][name] += 1
        audit['report_identity_checked'] += 1
    names = {'MAP@5': 'average_precision_at_k', 'MRR@5': 'reciprocal_rank_at_k',
             'Top-1': 'top_1', 'Top-5': 'top_5', 'MeanRecall@5': 'recall_at_k'}
    cohorts = {'all500': list(range(500)), 'all_four_success': [i for i in range(500)
               if all(calculated[name][i] is not None for name in groups)]}
    results = {}
    for cohort, indices in cohorts.items():
        results[cohort] = {}
        for name, rows in calculated.items():
            valid = [rows[i] for i in indices if rows[i] is not None]
            results[cohort][name] = {'samples': len(indices), 'successful': len(valid),
                'metrics': {label: sum(x[field] for x in valid) / len(indices) for label, field in names.items()}}
    paired = {}
    for name in ('BLAZE RRF', 'RRL', 'C3'):
        for cohort, indices in cohorts.items():
            count = Counter()
            for i in indices:
                a = calculated['IGSIL'][i]['average_precision_at_k'] if calculated['IGSIL'][i] else 0
                b = calculated[name][i]['average_precision_at_k'] if calculated[name][i] else 0
                count['win' if a-b > 1e-12 else 'loss' if a-b < -1e-12 else 'tie'] += 1
            paired[cohort + ': IGSIL vs ' + name] = {k: count[k] for k in ('win', 'tie', 'loss')}
    output.mkdir(parents=True, exist_ok=False)
    result = {'selection_sha256': selection['selection_sha256'], 'cohorts': results, 'paired': paired, 'audit': audit,
              'protocol': 'AP@5 divides by all gold files; failed reports count zero in all500'}
    write_json(output / 'comparison.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.output if a.output.is_absolute() else PROJECT / a.output)
