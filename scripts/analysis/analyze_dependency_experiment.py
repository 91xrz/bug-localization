"""Audit unchanged focal chunks, dependency provenance and paired @10 results."""

import argparse
import ast
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.snapshot import sha256_file


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


@lru_cache(maxsize=128)
def source_info(path):
    raw_file = path.read_bytes()
    raw = raw_file.decode('utf-8-sig').encode('utf-8')
    offsets, position = [], 0
    for line in raw.splitlines(keepends=True):
        offsets.append(position)
        position += len(line)
    offsets.append(position)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        root = ast.parse(raw.decode('utf-8'), filename=str(path))
    functions, assignments, calls = set(), set(), []
    for node in ast.walk(root):
        if not hasattr(node, 'lineno') or node.end_lineno is None:
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add((start, end, node.name))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            assignments.add((start, end))
        elif isinstance(node, ast.Call):
            calls.append((start, ast.unparse(node.func)))
    return {'raw': raw, 'sha256': hashlib.sha256(raw_file).hexdigest(),
            'functions': functions, 'assignments': assignments, 'calls': calls}


def analyze(folder, reference=None):
    from transformers import AutoTokenizer

    summary, selection, config = (read(folder / name) for name in ('summary.json', 'selection.json', 'effective_config.json'))
    if (set(summary['routes']) != {'blagent_ast', 'ast_dependency'} or summary['fusion']
            or summary['file_aggregation'] != 'max'):
        raise ValueError('Use the frozen two-route Max experiment without fusion')
    records = {r: read(folder / r / 'samples.json') for r in summary['routes']}
    n = len(selection['samples'])
    for route, data in records.items():
        if len(data) != n or any(v['status'] != 'ok' for v in data):
            raise ValueError('Resolve failed or missing samples before comparing results')
        if aggregate_results(data, 10)['metrics'] != summary['routes'][route]['metrics']:
            raise ValueError('Summary disagrees with independent per-report aggregation')
        keys = ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha')
        if [tuple(v[k] for k in keys) for v in data] != [tuple(v[k] for k in keys) for v in selection['samples']]:
            raise ValueError('Sample pairing changed')
    if reference:
        if read(reference / 'selection.json')['samples'] != selection['samples']:
            raise ValueError('Historical AST96 selection differs')
        if read(reference / 'effective_config.json')['model'] != config['model']:
            raise ValueError('Historical model identity differs')
    original_records = read(reference / 'blagent_ast/samples.json') if reference else None
    tokenizer = AutoTokenizer.from_pretrained(str(PROJECT / config['semantic']['model_path']),
                                              local_files_only=True, trust_remote_code=False)
    tokenizer.model_max_length = 10**9
    settings = config['semantic']
    totals = {'chunks': 0, 'enriched_chunks': 0, 'function_chunks': 0, 'function_previews': 0,
              'truncated_function_previews': 0, 'field_previews': 0, 'cross_file_function_previews': 0,
              'resolved_candidates': 0, 'unresolved_calls': 0, 'budget_skipped_candidates': 0,
              'unchanged_focal_chunks': 0, 'unchanged_embedding_chunks': 0,
              'provenance_verified_function_previews': 0, 'provenance_verified_field_previews': 0,
              'historical_matching_metrics': 0, 'historical_matching_full_rankings': 0}
    context_tokens, input_tokens, parse_status, samples = [], [], {}, []
    costs = {r: {'elapsed_seconds': 0, 'new_vectors': 0, 'cache_hits': 0} for r in records}
    max_score_error = 0
    for i in range(n):
        original, enhanced = records['blagent_ast'][i], records['ast_dependency'][i]
        base_path, dep_path = folder / original['result_file'], folder / enhanced['result_file']
        base_result, dep_result = read(base_path), read(dep_path)
        if (base_result['candidate_fingerprint'] != dep_result['candidate_fingerprint'] or
                base_result['ground_truth'] != dep_result['ground_truth']):
            raise ValueError('Candidate or label mismatch')
        inventory = read(base_path.parent / 'candidates.json')['files']
        hashes = {v['file_path']: v['sha256'] for v in inventory}
        source_root = PROJECT / 'repos' / original['repository'] / original['before_fix_sha'] / 'source'
        base_payload = read(base_path.with_name('blagent_ast_chunks.json'))
        dep_payload = read(dep_path.with_name('ast_dependency_chunks.json'))
        a, b = base_payload['chunks'], dep_payload['chunks']
        if len(a) != len(b) or base_payload['original_blocks'] != dep_payload['original_blocks']:
            raise ValueError('Dependency route changed source boundaries or chunk count')
        tokens = tokenizer([c['embedding_text'] for c in b], add_special_tokens=True)['input_ids']
        for old, new, ids in zip(a, b, tokens):
            if any(old[k] != new[k] for k in ('file_path', 'code', 'prefix', 'parent_id', 'start_byte', 'end_byte', 'start_line', 'end_line')):
                raise ValueError('Focal source or structural prefix changed')
            context = new['dependency_context']
            assert new['original_chunk_id'] == old['chunk_id']
            assert new['original_embedding_sha256'] == hashlib.sha256(old['embedding_text'].encode('utf-8')).hexdigest()
            assert new['embedding_text'] == old['embedding_text'] + ('\n\n' + context if context else '')
            assert len(ids) == new['token_count'] <= settings['max_tokens']
            actual_context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
            assert actual_context_tokens == new['dependency_context_tokens'] <= settings['dependency_max_tokens']
            assert len(new['dependency_functions']) <= settings['dependency_max_functions']
            assert len({v['id'] for v in new['dependency_functions']}) == len(new['dependency_functions'])
            assert len(new['dependency_fields']) <= settings['dependency_max_fields']
            totals['chunks'] += 1
            totals['unchanged_focal_chunks'] += 1
            totals['enriched_chunks'] += bool(context)
            totals['function_chunks'] += bool(new['dependency_functions'])
            totals['unchanged_embedding_chunks'] += new['embedding_text'] == old['embedding_text']
            totals['resolved_candidates'] += new['dependency_resolved_candidates']
            totals['unresolved_calls'] += new['dependency_unresolved_calls']
            totals['budget_skipped_candidates'] += new['dependency_skip_counts'].get('insufficient_token_budget', 0)
            status = new['dependency_resolution_status']
            parse_status[status] = parse_status.get(status, 0) + 1
            context_tokens.append(actual_context_tokens)
            input_tokens.append(len(ids))
            for target in new['dependency_functions']:
                assert target['file_path'] in hashes and target['source_sha256'] == hashes[target['file_path']]
                info = source_info(source_root / target['file_path'])
                assert info['sha256'] == target['source_sha256']
                assert (target['start_byte'], target['end_byte'], target['qualname'].split('.')[-1]) in info['functions']
                caller = source_info(source_root / new['file_path'])
                assert any(old['start_byte'] <= pos < old['end_byte'] and expr == target['call_expression']
                           for pos, expr in caller['calls'])
                assert not (target['file_path'] == new['file_path'] and
                            target['start_byte'] < new['end_byte'] and target['end_byte'] > new['start_byte'])
                assert len(tokenizer.encode(target['preview'], add_special_tokens=False)) <= settings['dependency_function_max_tokens']
                totals['provenance_verified_function_previews'] += 1
                totals['function_previews'] += 1
                totals['truncated_function_previews'] += target['preview_truncated']
                totals['cross_file_function_previews'] += target['file_path'] != new['file_path']
            for target in new['dependency_fields']:
                info = source_info(source_root / target['file_path'])
                assert info['sha256'] == target['source_sha256'] == hashes[target['file_path']]
                assert (target['start_byte'], target['end_byte']) in info['assignments']
                totals['provenance_verified_field_previews'] += 1
                totals['field_previews'] += 1
        for route, result in [('blagent_ast', base_result), ('ast_dependency', dep_result)]:
            check = result['verification']
            assert check['all_chunk_scores_verified'] and check['all_file_scores_verified']
            assert check['same_chunk_scores_for_all_aggregations']
            assert all(v['top_10_matches_local_order'] for v in check['aggregations'].values())
            max_score_error = max(max_score_error, check['max_cosine_error'])
            costs[route]['elapsed_seconds'] += records[route][i]['elapsed_seconds']
            costs[route]['new_vectors'] += result['new_embeddings']
            costs[route]['cache_hits'] += result['cache_hits']
        if original_records:
            old = read(reference / original_records[i]['result_file'])
            totals['historical_matching_metrics'] += base_result['metrics'] == old['metrics']
            totals['historical_matching_full_rankings'] += [v['file_path'] for v in base_result['ranking']] == [v['file_path'] for v in old['ranking']]
            assert read(base_path.parent / 'query.json')['embedding_text'] == read((reference / original_records[i]['result_file']).parent / 'query.json')['embedding_text']
        samples.append({'issue_id': original['issue_id'], 'baseline_AP@10': original['metrics']['average_precision_at_k'],
            'dependency_AP@10': enhanced['metrics']['average_precision_at_k'],
            'AP_delta': enhanced['metrics']['average_precision_at_k'] - original['metrics']['average_precision_at_k'],
            'baseline_RR@10': original['metrics']['reciprocal_rank_at_k'],
            'dependency_RR@10': enhanced['metrics']['reciprocal_rank_at_k'],
            **dep_result['dependency_statistics']})
    indices = np.random.default_rng(20261026).integers(0, n, size=(20000, n))
    effects = {}
    for label, field in [('MAP@10', 'average_precision_at_k'), ('MRR@10', 'reciprocal_rank_at_k')]:
        delta = np.array([b['metrics'][field] - a['metrics'][field]
                          for a, b in zip(records['blagent_ast'], records['ast_dependency'])])
        effects[label] = {'delta': float(delta.mean()), 'ci95': np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(),
            'wins': int((delta > 1e-12).sum()), 'ties': int((abs(delta) <= 1e-12).sum()), 'losses': int((delta < -1e-12).sum())}
    output = {'run': str(folder), 'selected_samples': n, 'selection_sha256': selection['selection_sha256'],
        'scope': 'Exploratory previously observed single-project reports; not held-out or cross-project evidence',
        'metrics': {r: v['metrics'] for r, v in summary['routes'].items()}, 'effects': effects, 'audit': totals,
        'dependency_parse_status_chunk_instances': parse_status, 'max_cosine_error': max_score_error,
        'max_context_tokens': max(context_tokens), 'max_input_tokens': max(input_tokens),
        'mean_added_context_tokens_on_enriched_chunks': sum(context_tokens) / max(1, totals['enriched_chunks']),
        'enriched_fraction': totals['enriched_chunks'] / totals['chunks'], 'costs': costs,
        'cost_caution': 'Baseline warm; new-vector counts depend on single-sample cache. No cold-start speed ratio claimed.',
        'bootstrap': {'seed': 20261026, 'replicates': 20000, 'unit': 'report',
                      'caution': 'Shared project/snapshots; descriptive intervals only'}, 'per_report': samples}
    write_json(folder / 'dependency_analysis.json', output)
    with (folder / 'dependency_samples.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    print(json.dumps({k: v for k, v in output.items() if k != 'per_report'}, indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference', type=Path, help='Historical AST96 run for unchanged-baseline audit')
    args = parser.parse_args()
    analyze(args.run, args.reference)
