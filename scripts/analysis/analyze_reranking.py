"""Verify saved reranking scores/metrics and report paired results per project."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.snapshot import sha256_file


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(folder, expected_selection=None):
    summary, effective = read(folder / 'summary.json'), read(folder / 'effective_config.json')
    samples, selection = read(folder / 'samples.json'), read(folder / 'selection.json')
    if summary['status'] != 'completed':
        raise ValueError('Only fully successful completed runs can pass this audit')
    if expected_selection and selection['samples'] != read(expected_selection)['samples']:
        raise ValueError('Run differs from frozen expected selection')
    source, k = Path(effective['source_run']), summary['evaluation_k']
    for name, checksum in effective['source_metadata_sha256'].items():
        if sha256_file(source / name) != checksum:
            raise ValueError('Source metadata changed: ' + name)
    projects, changes, verification = {}, [], {}
    for route, pair in samples.items():
        if len(pair['baseline']) != len(selection['samples']) or len(pair['reranked']) != len(selection['samples']):
            raise ValueError('Incomplete paired records')
        scored_pairs, maximum_tokens, pool_recalls = 0, 0, []
        for meta, baseline, reranked in zip(selection['samples'], pair['baseline'], pair['reranked']):
            for record in (baseline, reranked):
                if record['status'] != 'ok' or any(record[key] != meta[key] for key in
                        ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha')):
                    raise ValueError('Paired sample identity/status mismatch')
            case = read(folder / reranked['result_file'])
            if effective['reranker'].get('source_query_route') == 'direct_hypothesis' and effective.get('reranker_query_source') == 'original_report':
                query_path = case.get('reranker_query_file')
                if not query_path or query_path not in case['source_files_sha256']:
                    raise ValueError('Missing original-report query provenance')
                original_query = read(Path(query_path))
                wrapper = 'Instruct: ' + effective['source_config']['semantic']['query_instruction'] + '\nQuery: '
                if original_query['query_info']['embedding_text'] != wrapper + case['query']:
                    raise ValueError('Reranker did not use the exact original-report query')
            for name, checksum in case['source_files_sha256'].items():
                if sha256_file(source / name) != checksum:
                    raise ValueError('Source artifact changed: ' + name)
            rows = case['ranking']
            original = sorted(rows, key=lambda row: row['semantic_rank'])
            if len({r['file_path'] for r in rows}) != len(rows):
                raise ValueError('Duplicate ranked files')
            if [r['semantic_rank'] for r in original] != list(range(1, len(rows) + 1)):
                raise ValueError('Incomplete baseline ranking')
            groups = defaultdict(list)
            for chunk in case['scored_chunks']:
                groups[chunk['file_path']].append(chunk['score'])
                maximum_tokens = max(maximum_tokens, chunk['input_tokens'])
            expected_head = original[:effective['reranker']['top_n_files']]
            if set(groups) != {r['file_path'] for r in expected_head}:
                raise ValueError('Wrong reranking candidate pool')
            if any(len(scores) > effective['reranker']['chunks_per_file'] for scores in groups.values()):
                raise ValueError('Too many chunks per file')
            head = sorted(expected_head, key=lambda r: (-max(groups[r['file_path']]), r['semantic_rank']))
            expected = head + original[len(head):]
            if [r['file_path'] for r in rows] != [r['file_path'] for r in expected]:
                raise ValueError('Saved ranking does not match independent max-score aggregation')
            for rank, row in enumerate(rows, 1):
                expected_score = max(groups[row['file_path']]) if row['file_path'] in groups else None
                if row['rank'] != rank or row['score'] != expected_score:
                    raise ValueError('Incorrect saved score or rank')
            for ranking, record, metric_key in ((original, baseline, 'baseline_metrics'), (rows, reranked, 'metrics')):
                metrics = ranking_metrics([r['file_path'] for r in ranking], case['ground_truth'], k)
                if metrics != record['metrics'] or metrics != case[metric_key]:
                    raise ValueError('Saved metrics differ from independent recomputation')
            pool_recalls.append(len(set(groups) & set(case['ground_truth'])) / len(set(case['ground_truth'])))
            scored_pairs += len(case['scored_chunks'])
            changes.append({**meta, 'route': route,
                'delta_ap': reranked['metrics']['average_precision_at_k'] - baseline['metrics']['average_precision_at_k'],
                'gold_ranks': [{'file_path': r['file_path'], 'before': r['semantic_rank'], 'after': r['rank']}
                               for r in rows if r['file_path'] in case['ground_truth']]})
        for name in ('baseline', 'reranked'):
            if aggregate_results(pair[name], k)['metrics'] != summary['routes'][route][name]['metrics']:
                raise ValueError('Saved overall metrics differ from recomputation')
        projects[route] = {}
        for repo in sorted({r['repository'] for r in pair['baseline']}):
            stats = {name: aggregate_results([r for r in pair[name] if r['repository'] == repo], k)
                     for name in ('baseline', 'reranked')}
            stats['delta'] = {key: stats['reranked']['metrics'][key] - value
                              for key, value in stats['baseline']['metrics'].items()}
            deltas = [r['delta_ap'] for r in changes if r['route'] == route and r['repository'] == repo]
            stats['paired_ap'] = {'wins': sum(x > 1e-12 for x in deltas), 'ties': sum(abs(x) <= 1e-12 for x in deltas),
                                  'losses': sum(x < -1e-12 for x in deltas)}
            projects[route][repo] = stats
        verification[route] = {'verified_samples': len(pair['baseline']), 'scored_pairs': scored_pairs,
            'max_input_tokens': maximum_tokens, 'mean_candidate_pool_recall': sum(pool_recalls) / len(pool_recalls),
            'source_hashes_verified': True, 'rankings_and_metrics_independently_recomputed': True}
    write_json(folder / 'verification.json', {'routes': verification, 'expected_selection_verified': bool(expected_selection)})
    write_json(folder / 'project_summary.json', projects)
    write_json(folder / 'case_changes.json', sorted(changes, key=lambda row: row['delta_ap']))
    with (folder / 'project_comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['route', 'repository', 'variant', 'samples',
            f'MAP@{k}', f'MRR@{k}', 'Top-1', 'Top-5', 'Top-10', f'MeanRecall@{k}'])
        writer.writeheader()
        for route, repos in projects.items():
            for repo, stats in repos.items():
                for name in ('baseline', 'reranked'):
                    writer.writerow({'route': route, 'repository': repo, 'variant': name,
                        'samples': stats[name]['selected_samples'], **stats[name]['metrics']})
    print(json.dumps({'verification': verification, 'project_map': {
        route: {repo: {'before': stats['baseline']['metrics'][f'MAP@{k}'],
                       'after': stats['reranked']['metrics'][f'MAP@{k}'], **stats['paired_ap']}
                for repo, stats in repos.items()} for route, repos in projects.items()}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--selection', type=Path, help='Optional frozen expected selection')
    args = parser.parse_args()
    analyze(PROJECT / args.run, PROJECT / args.selection if args.selection else None)
