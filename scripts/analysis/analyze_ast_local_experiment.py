"""Audit and summarize paired AST64 experiments; no model calls."""
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.ast_local_experiment import LOCAL_ROUTES
from bug_localization.batch import aggregate_results, write_json


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def analyze(folder):
    summary = read(folder / 'summary.json')
    selection = read(folder / 'selection.json')
    routes = ('blagent_ast',) + tuple(r for r in LOCAL_ROUTES if r in summary['routes'])
    if set(summary['routes']) != set(routes) or summary['fusion'] or summary['file_aggregation'] != 'max':
        raise ValueError('Expected baseline and registered local routes, Max aggregation and no fusion')
    keys = ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha')
    expected = [tuple(s[k] for k in keys) for s in selection['samples']]
    records = {r: read(folder / r / 'samples.json') for r in routes}
    for r, data in records.items():
        if len(data) != len(expected) or any(x['status'] != 'ok' for x in data):
            raise ValueError('Missing or failed records: ' + r)
        if [tuple(s[k] for k in keys) for s in data] != expected:
            raise ValueError('Selection mismatch: ' + r)
        if aggregate_results(data, 10)['metrics'] != summary['routes'][r]['metrics']:
            raise ValueError('Summary mismatch: ' + r)
    output = {'selection_sha256': selection['selection_sha256'], 'n': len(expected), 'routes': {}, 'effects': {}}
    for r in routes:
        count = changed = sources = 0
        kinds = Counter()
        maximum_error = 0
        for i, rec in enumerate(records[r]):
            result_path = folder / rec['result_file']
            result = read(result_path)
            base = read(folder / records['blagent_ast'][i]['result_file'])
            if (result['candidate_fingerprint'], result['ground_truth']) != (base['candidate_fingerprint'], base['ground_truth']):
                raise ValueError('Candidate or label mismatch')
            v = result['verification']
            assert v['all_chunk_scores_verified'] and v['all_file_scores_verified'] and v['same_chunk_scores_for_all_aggregations']
            assert all(a['top_10_matches_local_order'] for a in v['aggregations'].values())
            maximum_error = max(maximum_error, v['max_cosine_error'])
            chunks = read(result_path.with_name(r + '_chunks.json'))['chunks']
            assert len(chunks) == result['indexed_chunks']
            baseline_chunks = read((folder / records['blagent_ast'][i]['result_file']).with_name('blagent_ast_chunks.json'))['chunks']
            original = {}
            for c in baseline_chunks:
                original.setdefault(c['file_path'], []).append(c['code'])
            original = {path: ''.join(parts) for path, parts in original.items()}
            if r not in ('blagent_ast', 'ast_scope'):
                assert len(chunks) == len(baseline_chunks)
                for chunk, prior in zip(chunks, baseline_chunks):
                    for key in ('file_path', 'start_byte', 'end_byte', 'code', 'prefix'):
                        assert chunk[key] == prior[key]
            by_file = {}
            for c in chunks:
                by_file.setdefault(c['file_path'], []).append(c)
            for group in by_file.values():
                raw = ''.join(c['code'] for c in group).encode('utf-8')
                assert raw == original[group[0]['file_path']].encode('utf-8')
                offset = 0
                for c in group:
                    assert c['start_byte'] == offset and c['end_byte'] - offset == len(c['code'].encode('utf-8'))
                    assert c['token_count'] <= 512 and c['prefix_tokens'] <= 64
                    offset = c['end_byte']
                    items = c.get('local_context_sources', [])
                    count += 1
                    changed += bool(items)
                    sources += len(items)
                    for item in items:
                        assert 0 <= item['start_byte'] < item['end_byte'] <= c['start_byte']
                        fragment = raw[item['start_byte']:item['end_byte']].decode('utf-8').strip()
                        assert item['kind'] + ': ' + fragment in c['embedding_text']
                        kinds[item['kind']] += 1
        output['routes'][r] = {'metrics': summary['routes'][r]['metrics'], 'chunks': count,
            'context_chunks': changed, 'context_sources': sources, 'context_kinds': dict(kinds),
            'new_embeddings': sum(read(folder / x['result_file'])['new_embeddings'] for x in records[r]),
            'elapsed_seconds': sum(x['elapsed_seconds'] for x in records[r]), 'max_cosine_error': maximum_error}
    indices = np.random.default_rng(20261028).integers(0, len(expected), size=(20000, len(expected)))
    for r in routes[1:]:
        effects = {}
        for title, field in [('MAP@10','average_precision_at_k'), ('MRR@10','reciprocal_rank_at_k')]:
            delta = np.array([a['metrics'][field] - b['metrics'][field] for a, b in zip(records[r], records['blagent_ast'])])
            effects[title] = {'delta': float(delta.mean()), 'bootstrap_95': np.quantile(delta[indices].mean(axis=1), [.025,.975]).tolist(),
                              'wins': int((delta > 1e-12).sum()), 'ties': int((abs(delta) <= 1e-12).sum()), 'losses': int((delta < -1e-12).sum())}
        output['effects'][r] = effects
    write_json(folder / 'ast_local_analysis.json', output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    analyze(Path(sys.argv[1]))
