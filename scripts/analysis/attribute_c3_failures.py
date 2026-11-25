"""Offline candidate coverage and review packets; never infer semantic relevance from file labels."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from pathlib import Path
import random
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.reranking import read_json
from bug_localization.batch import write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.snapshot import sha256_file
from bug_localization.paired_statistics import align


def classify_file(path, inventory, candidates, ranks, k=10):
    if path not in inventory:
        return 'source_inventory_gap'
    if path not in candidates:
        return 'recall_gap'
    if ranks[path] > k:
        return 'selection_or_reranking_unresolved'
    return 'retrieved_at_k'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    from datasets import load_from_disk
    import pyarrow.parquet as pq
    data = {n: load_from_disk(str(PROJECT / 'datasets' / d))['test'] for n, d in [('development500', 'swe_bench_formatted'), ('external47', 'lca_py_formatted')]}
    old_a = PROJECT / read_json(PROJECT / 'configs/experiments/hypothesis_reranker180.json')['original_reranker_run']
    control = PROJECT / 'results/hypothesis_reranking/20261111_153819_040009_500'
    c3root = PROJECT / 'results/chunk_ablation_reranking/20261112_C2_C3_500/C3'
    ext = PROJECT / 'results/c3_supplement/20261112_lca_py_holdout'
    dev = read_json(PROJECT / 'results/mixed_recall_reranking/20261111_CD_500/paired_samples.json')['cohorts']['all500']
    groups = {'development500': {'A': dev['A'], 'C3': read_json(c3root / 'samples.json')},
              'external47': {m: read_json(ext / m / 'samples.json') for m in ('A', 'C3')}}
    raw_lca = {str(r['id']): r for r in pq.read_table(PROJECT / 'datasets/LCA_bug_localization/py/test-00000-of-00001.parquet').to_pylist()}
    source_hashes, rows, case_cache = {}, [], {}
    file_counts, report_counts = defaultdict(Counter), defaultdict(Counter)
    for cohort, methods in groups.items():
        align(methods)
        for idx, (ar, cr) in enumerate(zip(methods['A'], methods['C3']), 1):
            sample = data[cohort][ar['dataset_row']]
            row = {k: ar[k] for k in ('position','dataset_row','repository','issue_id','before_fix_sha')}
            row.update(cohort=cohort, issue_url=sample['issue_url'], methods={})
            for method, record in [('A', ar), ('C3', cr)]:
                root = (old_a if idx <= 180 else control / 'new320/original_reranking') if method == 'A' else c3root
                if cohort == 'external47': root = ext / method
                if record['status'] != 'ok':
                    row['methods'][method] = {'status': 'failed', 'ap': 0., 'error': record.get('error'), 'stage': 'execution_failure'}
                    report_counts[(cohort, method)]['execution_failure'] += 1
                    continue
                path = root / record['result_file']; saved = read_json(path)
                if any(saved[k] != record[k] for k in ('dataset_row','repository','issue_id','before_fix_sha','query_sha256','labels_sha256')):
                    raise ValueError('Saved result identity differs')
                if set(saved['ground_truth']) != set(ast.literal_eval(sample['updated_files'])):
                    raise ValueError('Dataset labels differ')
                metrics = ranking_metrics([r['file_path'] for r in saved['ranking']], saved['ground_truth'], 10)
                if metrics != saved['metrics'] or metrics != record['metrics']:
                    raise ValueError('Saved metrics differ from recomputation')
                source_hashes[str(path)] = sha256_file(path)
                if 'result_sha256' in record and source_hashes[str(path)] != record['result_sha256']:
                    raise ValueError('Result hash mismatch')
                inventory_paths = [Path(p) for p in saved['source_files_sha256'] if Path(p).name == 'candidates.json']
                if len(inventory_paths) != 1: raise ValueError('Ambiguous inventory')
                invkey = str(inventory_paths[0])
                invpath = inventory_paths[0]
                if not invpath.is_absolute():
                    invpath = PROJECT / 'results/report_hypothesis/20261110_023926_805713_retrieval' / invpath
                source_hashes[str(invpath)] = sha256_file(invpath)
                if source_hashes[str(invpath)] != saved['source_files_sha256'][invkey]: raise ValueError('Inventory changed')
                inventory = {r['file_path'] for r in read_json(invpath)['files']}
                # Historical A keeps an unscored tail; only scored files belong to candidate-50.
                candidates = {r['file_path'] for r in saved['scored_chunks']}
                if len(candidates) != min(50, len(inventory)):
                    raise ValueError('Candidate-50 inventory mismatch')
                ranks = {r['file_path']: r['rank'] for r in saved['ranking']}
                gold = saved['ground_truth']
                files = [{'file_path': g, 'rank': ranks.get(g) if g in candidates else None, 'unscored_tail_rank': ranks.get(g) if g not in candidates else None, 'stage': classify_file(g, inventory, candidates, ranks),
                    'selected_chunk_ids': [r['chunk_id'] for r in saved['scored_chunks'] if r['file_path'] == g]} for g in gold]
                counts = Counter(r['stage'] for r in files); file_counts[(cohort, method)].update(counts)
                if not any(g in candidates for g in gold): stage = 'no_gold_candidate'
                elif any(g not in candidates for g in gold): stage = 'partial_gold_candidate_coverage'
                elif any(ranks[g] > 10 for g in gold): stage = 'all_gold_candidates_but_some_below_top10'
                else: stage = 'all_gold_within_top10'
                report_counts[(cohort, method)][stage] += 1
                row['methods'][method] = {'status': 'ok', 'ap': record['metrics']['average_precision_at_k'], 'stage': stage,
                    'gold_files': files, 'gold_count': len(gold), 'candidate_gold_count': sum(g in candidates for g in gold),
                    'top10_gold_count': sum(g in ranks and ranks[g] <= 10 for g in gold), 'result_path': str(path)}
                case_cache[(cohort, idx, method)] = saved
            row['delta_ap'] = row['methods']['C3']['ap'] - row['methods']['A']['ap']
            rows.append(row)
    write_json(out / 'attribution.json', {'definition': 'File-level stages at Top-10 are mutually exclusive; semantic selection vs reranking remains unresolved until code review. Source inventory gaps and execution failures are separate. All-gold Top-10 does not imply perfect AP.',
        'cohorts': {cohort: {m: {'reports': dict(report_counts[(cohort, m)]), 'gold_files_successful_attempts': dict(file_counts[(cohort, m)])} for m in ('A','C3')} for cohort in groups}, 'rows': rows})
    selected = []
    rng = random.Random(20261112)
    for cohort in groups:
        pool = [r for r in rows if r['cohort'] == cohort and all(s['status']=='ok' for s in r['methods'].values())]
        k = 5 if cohort == 'development500' else 2
        gain = sorted([r for r in pool if r['delta_ap'] > 1e-12], key=lambda r: (-r['delta_ap'], r['position']))[:k]
        loss = sorted([r for r in pool if r['delta_ap'] < -1e-12], key=lambda r: (r['delta_ap'], r['position']))[:k]
        for label, group in [('large_gain', gain), ('large_loss', loss)]:
            selected.extend({**r, 'sampling_stratum': label} for r in group)
        used = {r['position'] for r in gain + loss}
        ordinary = [r for r in pool if r['position'] not in used and abs(r['delta_ap']) <= .05]
        strata = [('ordinary_tied_imperfect', lambda r: abs(r['delta_ap']) < 1e-12 and r['methods']['A']['ap'] < 1),
                  ('ordinary_tied_perfect', lambda r: abs(r['delta_ap']) < 1e-12 and r['methods']['A']['ap'] == 1),
                  ('ordinary_small_change', lambda r: abs(r['delta_ap']) >= 1e-12)]
        for (label, predicate), count in zip(strata, [4,3,3] if cohort=='development500' else [2,1,1]):
            options = [r for r in ordinary if predicate(r)]; rng.shuffle(options)
            selected.extend({**r, 'sampling_stratum': label} for r in options[:count])
    packets = out / 'review_packets'; packets.mkdir(exist_ok=True)
    template = []
    for r in selected:
        cohort, pos = r['cohort'], r['position']; sample = data[cohort][r['dataset_row']]
        if cohort == 'external47': hypothesis_path = ext / 'new320/hypothesis' / f'{pos:03d}_queries.json'
        elif pos <= 180: hypothesis_path = PROJECT / 'results/direct_report_hypothesis/20261110_113033_098831_retrieval' / f'{pos:03d}_queries.json'
        else: hypothesis_path = control / 'new320/hypothesis' / f'{pos-180:03d}_queries.json'
        hyp = read_json(hypothesis_path); source_hashes[str(hypothesis_path)] = sha256_file(hypothesis_path)
        packet = {**r, 'report_body': sample['body'], 'hypothesis_query': hyp['query'], 'hypothesis_raw': hyp.get('raw'),
            'hypothesis_path': str(hypothesis_path), 'selected_evidence': {}}
        for method in ('A','C3'):
            saved = case_cache[(cohort, pos, method)]; gold = set(saved['ground_truth'])
            top_non_gold = next((x['file_path'] for x in saved['ranking'] if x['file_path'] not in gold), None)
            packet['selected_evidence'][method] = {'query_used': saved['query'],
                'gold_chunks': [x for x in saved['scored_chunks'] if x['file_path'] in gold],
                'top_non_gold_path': top_non_gold,
                'top_non_gold_chunks': [x for x in saved['scored_chunks'] if x['file_path'] == top_non_gold]}
        if cohort == 'external47': packet['retrospective_reference_patch'] = raw_lca[str(sample['issue_id'])]['diff']
        key = f'{cohort}_{pos:03d}'
        write_json(packets / (key + '.json'), packet)
        template.append({'case_id': key, 'sampling_stratum': r['sampling_stratum'], 'repository': r['repository'],
            'delta_ap': r['delta_ap'], 'reviewer': '', 'review_status': 'pending_human_review',
            'hypothesis_adds_missing_functional_description': 'unresolved', 'hypothesis_is_incorrect': 'unresolved',
            'A_relevant_selected_chunk_ids': [], 'C3_relevant_selected_chunk_ids': [],
            'selection_failure_files': [], 'reranking_failure_files': [], 'evidence_and_rationale': '',
            'caution': 'A gold file or changed-line overlap alone is not proof of an informative block. Do not infer hallucination merely from performance regression.'})
    write_json(out / 'review_selection.json', {'seed':20261112, 'sampling':'Largest AP gains/losses plus random ordinary tied-imperfect, tied-perfect and small-change cases; diagnostic sample, not prevalence estimate.', 'samples': selected})
    review_path = out / 'human_review_template.json'
    if review_path.exists() and read_json(review_path) != template:
        raise ValueError('Refusing to overwrite a changed human-review template')
    write_json(review_path, template)
    write_json(out / 'source_hashes.json', source_hashes)
    with (out / 'case_index.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','sampling_stratum','repository','delta_ap']);w.writeheader()
        w.writerows({k:r[k] for k in w.fieldnames} for r in template)
    print(out, 'reports',len(rows),'review packets',len(selected))

if __name__ == '__main__': main()
