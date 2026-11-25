"""Recompute paired enrichment metrics and export the exploratory parameter table."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from bug_localization.bm25 import ranking_metrics
from bug_localization.batch import aggregate_results, write_json
from bug_localization.report_views import weighted_rrf

folder = Path(sys.argv[1])
records = json.loads((folder / 'samples.json').read_text())
config = json.loads((folder / 'effective_config.json').read_text())['experiment']
summary = json.loads((folder / 'summary.json').read_text())
selection = json.loads((folder / 'selection.json').read_text())
count = len(selection['samples'])
rows = []
quality = {f: {basis: 0 for basis in ('explicit', 'inferred', 'missing')} for f in ('OB', 'EB', 'S2R')}
rejected = 0
generation_failures = 0
for path in folder.glob('*_queries.json'):
    query = json.loads(path.read_text())
    generation_failures += int('generation_error' in query)
    for field, item in query['sections'].items():
        quality[field][item['basis']] += 1
    rejected += len(query.get('rejected_sections', []))
for name, samples in records.items():
    truncated = 0
    assert len(samples) == count, (name, 'incomplete')
    for sample in samples:
        if sample['status'] != 'ok':
            continue
        case = folder / 'cases' / f'{sample["position"]:03d}'
        result = json.loads((case / (name + '_result.json')).read_text())
        retrieval = case / (name + '_retrieval.json')
        if retrieval.exists():
            truncated += int(json.loads(retrieval.read_text())['query_info']['truncated'])
        metrics = ranking_metrics([r['file_path'] for r in result['ranking']], result['ground_truth'], 10)
        assert metrics == result['metrics'] == sample['metrics']
        if name.startswith('rrf_'):
            weight, constant = name[5:].split('_c')
            rankings = {q: json.loads((case / (source + '_result.json')).read_text())['ranking']
                        for q, source in [('q0', 'q0'), ('enriched', config.get('fusion_query', 'OB_EB_S2R'))]}
            assert result['ranking'] == weighted_rrf(rankings, {'q0': 1, 'enriched': float(weight)}, int(constant))
    recomputed = aggregate_results(samples, 10, count)
    assert recomputed == summary[name]
    paired = [(s['metrics']['average_precision_at_k'], b['metrics']['average_precision_at_k'])
              for s, b in zip(samples, records['q0']) if s['status'] == b['status'] == 'ok']
    rows.append({'query': name, **recomputed['metrics'],
        'delta_MAP': recomputed['metrics']['MAP@10'] - summary['q0']['metrics']['MAP@10'],
        'truncated_queries': truncated if not name.startswith('rrf_') else None,
        'wins': sum(a > b for a, b in paired), 'ties': sum(a == b for a, b in paired),
        'losses': sum(a < b for a, b in paired)})
with (folder / 'comparison.csv').open('w', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
project_rows = []
for repository in sorted({r['repository'] for r in records['q0']}):
    reference = aggregate_results([r for r in records['q0'] if r['repository'] == repository], 10)
    for name, samples in records.items():
        group = [r for r in samples if r['repository'] == repository]
        project = aggregate_results(group, 10)
        project_rows.append({'repository': repository, 'query': name,
                             'samples': len(group), 'failed_samples': project['failed_samples'],
                             **project['metrics'],
                             'delta_MAP': project['metrics']['MAP@10'] - reference['metrics']['MAP@10']})
with (folder / 'project_comparison.csv').open('w', newline='') as out:
    writer = csv.DictWriter(out, fieldnames=list(project_rows[0]))
    writer.writeheader()
    writer.writerows(project_rows)
write_json(folder / 'verification.json', {'metrics_and_fusion_verified': True, 'scope': config['scope'], 'comparison': rows, 'section_quality': quality, 'rejected_sections': rejected, 'generation_failures': generation_failures})
for row in sorted(rows, key=lambda r: -r['MAP@10']):
    print(f"{row['query']:28s} MAP={row['MAP@10']:.4f} delta={row['delta_MAP']:+.4f} W/T/L={row['wins']}/{row['ties']}/{row['losses']}")
