"""Verify report-only hypothesis inputs and paired historical baseline rankings."""
import json
from pathlib import Path
import runpy
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import write_json
from bug_localization.embeddings import digest
from bug_localization.report_hypothesis import parse_hypothesis, build_queries, report_payload, FIELDS
from bug_localization.snapshot import sha256_file

folder = Path(sys.argv[1]).resolve()
runpy.run_path(str(Path(__file__).with_name('analyze_report_enrichment.py')), run_name='__main__')
config = json.loads((folder / 'effective_config.json').read_text())['experiment']
source = PROJECT / config['enrichment_source']
selection = json.loads((folder / 'selection.json').read_text())
previous_selection = json.loads((source / 'selection.json').read_text())
assert selection['samples'] == previous_selection['samples'], 'Different paired evaluation sample lists'
reference = source if (source / 'cases').is_dir() else None
checks, unique = [], {}
for sample in selection['samples']:
    name = f'{sample["position"]:03d}'
    query = json.loads((folder / (name + '_queries.json')).read_text())
    previous = json.loads((source / (name + '_queries.json')).read_text())
    assert query['payload'] == report_payload(previous['report'], previous)
    assert query['sections'] == previous['sections']
    assert query['repository_knowledge_provided'] is False
    assert query['hypothesis_is_speculative'] is True
    assert sha256_file(PROJECT / query['source_file']) == query['source_sha256']
    assert query['key'] == digest({'identity': query['identity'], 'payload': query['payload']})
    try:
        hypothesis = parse_hypothesis(query['raw'], query['finished'])
        assert 'generation_error' not in query
    except ValueError:
        assert query.get('generation_error')
        hypothesis = {f: '' for f in FIELDS}
    assert query['hypothesis'] == hypothesis
    expected_queries = build_queries(previous['report'], previous['sections'], hypothesis)
    if config.get('query_routes'):
        expected_queries = {name: expected_queries[name] for name in config['query_routes']}
    assert query['queries'] == expected_queries
    controls = {}
    if reference is not None:
        for route in ('q0', 'OB_EB_S2R'):
            current = json.loads((folder / 'cases' / name / (route + '_result.json')).read_text())
            prior = json.loads((source / 'cases' / name / (route + '_result.json')).read_text())
            assert current['ground_truth'] == prior['ground_truth']
            current_paths = [r['file_path'] for r in current['ranking']]
            prior_paths = [r['file_path'] for r in prior['ranking']]
            assert set(current_paths) == set(prior_paths), (name, route, 'candidate ranking set changed')
            old_scores = {r['file_path']: r['score'] for r in prior['ranking']}
            controls[route] = {
                'full_ranking_matches': current_paths == prior_paths,
                'top10_ranking_matches': current_paths[:10] == prior_paths[:10],
                'metrics_match': current['metrics'] == prior['metrics'],
                'current_AP': current['metrics']['average_precision_at_k'],
                'prior_AP': prior['metrics']['average_precision_at_k'],
                'max_absolute_score_difference': max((abs(r['score'] - old_scores[r['file_path']]) for r in current['ranking']), default=0)}
        current_candidates = json.loads((folder / 'cases' / name / 'candidates.json').read_text())
        prior_candidates = json.loads((source / 'cases' / name / 'candidates.json').read_text())
        assert current_candidates == prior_candidates
        current_chunks = json.loads((folder / 'cases' / name / 'chunks.json').read_text())
        prior_chunks = json.loads((source / 'cases' / name / 'chunks.json').read_text())
        assert [(c['file_path'], c['embedding_text']) for c in current_chunks] == [(c['file_path'], c['embedding_text']) for c in prior_chunks]
    unique[query['key']] = query
    checks.append({'position': sample['position'], 'issue_id': sample['issue_id'],
                   'report_only_input_verified': True, 'historical_controls': controls, 'code_inputs_match': True if reference else None})
write_json(folder / 'hypothesis_verification.json', {
    'verified': True, 'cases': checks, 'historical_reference_available': reference is not None, 'unique_generations': len(unique),
    'input_tokens_unique': sum(q['input_tokens'] for q in unique.values()),
    'output_tokens_unique': sum(q['output_tokens'] for q in unique.values()),
    'invalid_generations_unique': sum('generation_error' in q for q in unique.values()),
    'historical_control_summary': {route: {field: sum(c['historical_controls'][route][field] for c in checks) for field in ('full_ranking_matches', 'top10_ranking_matches', 'metrics_match')} for route in ('q0', 'OB_EB_S2R')} if reference else None,
    'note': 'Code/query inputs checked against history. Device/batch numerical differences are recorded, not treated as identical rankings. Method comparisons use paired controls from the current run. Hypotheses remain unverified speculation.'})
print('Report-only inputs verified:', len(checks), '; historical reference available:', reference is not None)
