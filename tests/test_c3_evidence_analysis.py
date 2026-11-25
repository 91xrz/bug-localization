import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('c3_evidence_analysis', PROJECT / 'scripts/analysis/analyze_c3_evidence_pilot.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class EvidenceAnalysisTests(unittest.TestCase):
    def test_failed_case_counts_zero_but_pending_suppresses_metrics(self):
        ok = {'status': 'ok', 'metrics': analysis.ranking_metrics(['a.py'], ['a.py'], 10), 'missing_gold': []}
        failed = {'status': 'failed', 'repository': 'r', 'issue_id': 2, 'error_type': 'ValueError', 'error': 'bad output'}
        completed = analysis.summary([ok, failed], 2)
        self.assertEqual(completed['metrics']['MAP@10'], .5)
        self.assertEqual(completed['failed_samples'], 1)
        self.assertNotIn('metrics_successful_only', completed)
        pending = analysis.summary([ok], 2)
        self.assertIsNone(pending['metrics'])
        self.assertEqual(pending['pending_samples'], 1)

    def test_head_permutation_cannot_change_tail_or_candidate_membership(self):
        original = ['a.py', 'b.py', 'c.py', 'd.py']
        analysis.check_inventory(['b.py', 'a.py', 'c.py', 'd.py'], original, 2)
        for bad in [['b.py', 'a.py', 'd.py', 'c.py'], ['c.py', 'a.py', 'b.py', 'd.py'],
                    ['a.py', 'a.py', 'c.py', 'd.py']]:
            with self.assertRaises(ValueError):
                analysis.check_inventory(bad, original, 2)

    def test_evidence_audit_checks_actual_text_not_only_ids(self):
        saved = {'ranking': [{'file_path': 'a.py'}, {'file_path': 'b.py'}],
                 'scored_chunks': [{'file_path': 'a.py', 'chunk_id': 'a', 'embedding_text': 'dtype=X.dtype'}]}
        observed = {'query': 'integer loses fractions', 'candidates': ['a.py'],
                    'evidence': {'a.py': [{'chunk_id': 'a', 'text': 'dtype=X.dtype'}]},
                    'original_ranking': ['a.py', 'b.py'], 'tail': ['b.py']}
        observed['catalog'] = {'files': {'F01': 'a.py'}, 'chunks': {
            'C001': {'file_id': 'F01', 'chunk_id': 'a', 'text': 'dtype=X.dtype', 'line_count': 1}}}
        observed['user_content'] = ('BUG REPORT:\ninteger loses fractions\n\nCANDIDATE EVIDENCE:\n'
                                    'FILE F01: a.py\n\nEXCERPT C001 (local lines 1-1):\n1: dtype=X.dtype')
        observed['user_content_sha256'] = analysis.digest(observed['user_content'])
        report = {'query': observed['query']}
        analysis.check_input(observed, saved, report, 1)
        observed['evidence']['a.py'][0]['text'] = 'dtype=float'
        with self.assertRaisesRegex(ValueError, 'raw C3 evidence'):
            analysis.check_input(observed, saved, report, 1)

    def test_project_randomization_does_not_treat_reports_as_independent(self):
        # Ten same-direction reports from just two projects have only four
        # project-level label swaps; a report-wise test would be overconfident.
        result = analysis.cluster_inference([1.] * 10, ['a'] * 5 + ['b'] * 5, repetitions=100)
        self.assertEqual(result['n_projects'], 2)
        self.assertEqual(result['randomization_draws'], 4)
        self.assertEqual(result['p_two_sided_project_randomization'], .5)
        self.assertEqual(result['ci95_project_cluster_bootstrap'], [1., 1.])
        ties = analysis.cluster_inference([0., 0.], ['a', 'b'], repetitions=100)
        self.assertEqual(ties['p_two_sided_project_randomization'], 1.)

    def test_costs_include_failed_calls_and_do_not_impute_unknown_tokens(self):
        rows = [{'elapsed_seconds': 3.}, {'elapsed_seconds': 4.}]
        calls = {1: [{'status': 'ok', 'input_tokens': 10, 'output_tokens': 2, 'elapsed_seconds': 2.}],
                 2: [{'status': 'failed', 'input_tokens': 20, 'elapsed_seconds': 3.}]}
        result = analysis.cost_report(rows, calls)
        self.assertEqual(result['model_calls'], 2)
        self.assertEqual(result['failed_model_calls'], 1)
        self.assertEqual(result['input_tokens'], 30)
        self.assertEqual(result['output_tokens'], 2)
        self.assertEqual(result['calls_missing_output_tokens'], 1)
        self.assertEqual(result['summed_case_seconds'], 7.)

    def test_source_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.json'
            report = root / 'inputs.json'
            source.write_text(json.dumps({'ranking': [], 'scored_chunks': [], 'ground_truth': []}))
            report.write_text(json.dumps({'query': 'report'}))
            record = {'source_sha256': {str(p): analysis.sha256_file(p) for p in (source, report)}}
            analysis.source_data(record)
            report.write_text(json.dumps({'query': 'altered report'}))
            with self.assertRaisesRegex(ValueError, 'Source checksum'):
                analysis.source_data(record)

    def test_repository_coverage_does_not_confuse_top50_recall_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inventory = root / 'candidates.json'
            analysis.write_json(inventory, {'files': [{'file_path': p} for p in ('a.py', 'b.py', 'c.py')]})
            saved = {'ranking': [{'file_path': 'a.py'}], 'ground_truth': ['b.py', 'new.py'],
                     'source_files_sha256': {str(inventory): analysis.sha256_file(inventory)}}
            reference = root / 'reference.json'
            analysis.write_json(reference, dict(saved))
            saved.update(reference_file=str(reference), reference_sha256=analysis.sha256_file(reference))
            coverage = analysis.source_inventory(saved)
            self.assertEqual(coverage['missing_gold'], ['new.py'])
            self.assertEqual(coverage['gold_available_but_outside_c3_ranking'], ['b.py'])
            self.assertEqual(coverage['available_file_count'], 3)
            self.assertEqual(coverage['c3_ranking_file_count'], 1)
            analysis.write_json(inventory, {'files': [{'file_path': 'a.py'}]})
            with self.assertRaisesRegex(ValueError, 'inventory checksum'):
                analysis.source_inventory(saved)

    def test_request_identity_and_user_content_are_both_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            identity = {'settings': {'max_calls': 2, 'max_new_tokens': 2048}}
            protocol = {'generator_identity': identity, 'prompt': 'system'}
            user = 'bug and exact code evidence'
            observed = {'user_content_sha256': analysis.digest(user)}
            request = {'messages': [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': user}],
                       'temperature': 0, 'identity': analysis.digest(identity), 'max_new_tokens': 2048}
            call = {'request': request, 'request_sha256': analysis.digest(request), 'status': 'ok',
                    'response': '{}', 'finished': True}
            path = case / 'call_001.json'
            path.write_text(json.dumps(call))
            analysis.audit_calls(case, protocol, observed)
            request['messages'][1]['content'] = 'different evidence'
            call['request_sha256'] = analysis.digest(request)
            path.write_text(json.dumps(call))
            with self.assertRaisesRegex(ValueError, 'initial user content'):
                analysis.audit_calls(case, protocol, observed)

    def test_holm_adjustment_is_monotone_and_capped(self):
        rows = [{'p': .04}, {'p': .001}, {'p': .8}]
        analysis.holm(rows, 'p', 'adjusted')
        self.assertEqual([r['adjusted'] for r in rows], [.08, .003, .8])

    def test_posthoc_rank_only_uses_last_output_even_when_earlier_rank_is_better(self):
        sample = {'position': 1, 'dataset_row': 0, 'repository': 'r', 'issue_id': 1,
                  'before_fix_sha': 'a', 'query_sha256': 'q', 'labels_sha256': 'l', 'status': 'failed'}
        payload = {'catalog': {'files': {'F01': 'a.py', 'F02': 'b.py'}},
                   'original_ranking': ['a.py', 'b.py', 'c.py'], 'tail': ['c.py']}
        calls = [{'response': '{"ranked_ids":["F01","F02"]}', 'request_sha256': 'first'},
                 {'response': '{"ranked_ids":["F02","F01"]}', 'request_sha256': 'last'}]
        result = analysis.last_output_rank(sample, calls, payload, ['a.py'])
        self.assertEqual(result['ranking'], ['b.py', 'a.py', 'c.py'])
        self.assertEqual(result['metrics']['average_precision_at_k'], .5)
        self.assertEqual(result['last_request_sha256'], 'last')
        self.assertTrue(result['primary_validation_ignored'])
        self.assertEqual(sample['status'], 'failed')

    def test_posthoc_rank_only_never_repairs_or_falls_back_from_invalid_final_output(self):
        sample = {'position': 1, 'dataset_row': 0, 'repository': 'r', 'issue_id': 1,
                  'before_fix_sha': 'a', 'query_sha256': 'q', 'labels_sha256': 'l', 'status': 'failed'}
        payload = {'catalog': {'files': {'F01': 'a.py', 'F02': 'b.py'}},
                   'original_ranking': ['a.py', 'b.py'], 'tail': []}
        first = {'response': '{"ranked_ids":["F01","F02"]}'}
        for last in [{'response': '{"ranked_ids":["F01","F01"]}'},
                     {'response': '{"ranked_ids":["F01"]}'},
                     {'raw_response': '{"ranked_ids":['}, {'error': 'context overflow'}]:
            result = analysis.last_output_rank(sample, [first, last], payload, ['a.py'])
            self.assertEqual(result['status'], 'failed')
            self.assertNotIn('metrics', result)
            self.assertEqual(result['last_call_number'], 2)

    def test_posthoc_can_read_last_failed_calls_raw_output_without_repair(self):
        sample = {'position': 1, 'dataset_row': 0, 'repository': 'r', 'issue_id': 1,
                  'before_fix_sha': 'a', 'query_sha256': 'q', 'labels_sha256': 'l', 'status': 'failed'}
        payload = {'catalog': {'files': {'F01': 'a.py'}}, 'original_ranking': ['a.py'], 'tail': []}
        calls = [{'status': 'failed', 'raw_response': '{"ranked_ids":["F01"]}',
                  'error': 'Inference truncated or exhausted the output budget'}]
        result = analysis.last_output_rank(sample, calls, payload, ['a.py'])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['output_field'], 'raw_response')
        self.assertEqual(sample['status'], 'failed')

    def test_failure_classification_preserves_multiple_independent_causes(self):
        errors = ['R2 quote must be an exact nonempty report substring',
                  "F01 cites another file's excerpt C002",
                  'F01 reference range outside C001 lines 1-2',
                  'ranked_ids must contain every file ID exactly once',
                  'Inference truncated or exhausted the output budget']
        self.assertEqual(set(analysis.failure_categories(errors)), {
            'non_exact_report_quote', 'cross_file_reference', 'reference_out_of_bounds',
            'incomplete_or_invalid_ranking', 'output_budget_or_truncation'})

    def test_saved_ranking_must_be_the_actual_model_permutation(self):
        observed = {'catalog': {'files': {'F01': 'a.py', 'F02': 'b.py'}}, 'tail': ['c.py']}
        response = {'reasoning': 'observed mechanism', 'ranked_ids': ['F02', 'F01']}
        result = {'reasoning': response['reasoning'], 'ranking': ['a.py', 'b.py', 'c.py']}
        with self.assertRaisesRegex(ValueError, 'actual model response'):
            analysis.check_response(response, observed, result)
        result['ranking'] = ['b.py', 'a.py', 'c.py']
        analysis.check_response(response, observed, result)

    def test_citations_cannot_reference_another_files_code(self):
        observed = {'query': 'integer loses fractions', 'catalog': {
            'files': {'F01': 'a.py', 'F02': 'b.py'},
            'chunks': {'C001': {'file_id': 'F01', 'line_count': 3}}}, 'tail': []}
        response = {'reasoning': 'mechanism', 'ranked_ids': ['F01', 'F02'],
                    'constraints': [{'id': 'R1', 'kind': 'observed', 'quote': 'loses fractions'}],
                    'assessments': [
                        {'file_id': 'F01', 'state': 'unknown', 'constraint_ids': [], 'evidence': [], 'reason': 'missing context'},
                        {'file_id': 'F02', 'state': 'support', 'constraint_ids': ['R1'], 'reason': 'casts to integer',
                         'evidence': [{'chunk_id': 'C001', 'start': 1, 'end': 2}]}]}
        result = {'reasoning': 'mechanism', 'ranking': ['a.py', 'b.py']}
        with self.assertRaisesRegex(ValueError, 'another file'):
            analysis.check_response(response, observed, result)

    def test_complete_paired_run_is_recomputed_and_pending_is_not_reported(self):
        from bug_localization.evidence_reranking import build_input
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = {'cohort': 'development', 'samples': [
                {'position': i, 'dataset_row': i - 1, 'repository': f'p{i}', 'issue_id': i,
                 'before_fix_sha': 'a' * 40, 'query_sha256': f'q{i}', 'labels_sha256': f'l{i}'}
                for i in (1, 2)]}
            selection_path = root / 'selection.json'
            analysis.write_json(selection_path, selection)
            identity = {'settings': {'max_calls': 2, 'max_new_tokens': 2048}}
            for variant in ('baseline', 'evidence'):
                run = root / variant
                run.mkdir()
                analysis.write_json(run / 'selection.json', selection)
                analysis.write_json(run / 'protocol.json', {'generator_identity': identity,
                    'prompt': variant, 'head': 2, 'selection_sha256': analysis.sha256_file(selection_path)})
                for sample in selection['samples']:
                    i = sample['position']
                    case = run / 'cases' / f'{i:04d}'
                    case.mkdir(parents=True)
                    saved = {**sample, 'ranking': [{'file_path': p} for p in ('a.py', 'b.py', 'c.py')],
                             'ground_truth': ['a.py'], 'candidates': ['a.py', 'b.py', 'c.py'],
                             'scored_chunks': [{'file_path': p, 'chunk_id': p, 'embedding_text': 'code ' + p}
                                               for p in ('a.py', 'b.py')]}
                    inventory_folder = root / f'inventory{i}'
                    inventory_folder.mkdir(exist_ok=True)
                    inventory_path = inventory_folder / 'candidates.json'
                    analysis.write_json(inventory_path, {'files': [{'file_path': p} for p in ('a.py', 'b.py', 'c.py')]})
                    saved['source_files_sha256'] = {str(inventory_path): analysis.sha256_file(inventory_path)}
                    source_path = root / f'source{i}.json'
                    report_path = root / f'report{i}.json'
                    analysis.write_json(source_path, saved)
                    analysis.write_json(report_path, {'query': f'bug{i}'})
                    payload = build_input(saved, f'bug{i}', 2)
                    analysis.write_json(case / 'inference_input.json', payload)
                    ranked_ids = ['F02', 'F01'] if variant == 'baseline' else ['F01', 'F02']
                    ranking = [payload['catalog']['files'][k] for k in ranked_ids] + payload['tail']
                    response = {'reasoning': 'mechanism', 'ranked_ids': ranked_ids}
                    request = {'messages': [{'role': 'system', 'content': variant},
                                            {'role': 'user', 'content': payload['user_content']}],
                               'identity': analysis.digest(identity), 'temperature': 0, 'max_new_tokens': 2048}
                    analysis.write_json(case / 'call_001.json', {'request': request,
                        'request_sha256': analysis.digest(request), 'status': 'ok', 'finished': True,
                        'response': json.dumps(response), 'input_tokens': 10, 'output_tokens': 5})
                    metrics = analysis.ranking_metrics(ranking, ['a.py'], 10)
                    result = {'ranking': ranking, 'ground_truth': ['a.py'], 'metrics': metrics,
                              'reasoning': 'mechanism', 'structured_output': response, 'model_calls': 1}
                    result_path = case / 'result.json'
                    analysis.write_json(result_path, result)
                    analysis.write_json(case / 'record.json', {**sample, 'status': 'ok', 'metrics': metrics,
                        'source_sha256': {str(p): analysis.sha256_file(p) for p in (source_path, report_path)},
                        'result_file': str(result_path.relative_to(run)), 'result_sha256': analysis.sha256_file(result_path),
                        'missing_gold': [], 'elapsed_seconds': 1.})
            report = analysis.analyze(root / 'baseline', root / 'evidence', selection_path, root / 'analysis', repetitions=100)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['runs']['baseline']['summary']['metrics']['MAP@10'], .5)
            self.assertEqual(report['runs']['evidence']['summary']['metrics']['MAP@10'], 1.)
            self.assertEqual(report['project_cluster_statistics']['average_precision_at_k']['delta'], .5)
            self.assertEqual(report['posthoc_rank_only_diagnostic']['label'], 'posthoc_rank_only_diagnostic')
            self.assertTrue(report['posthoc_rank_only_diagnostic']['primary_results_unchanged'])
            (root / 'evidence/cases/0002/record.json').unlink()
            pending = analysis.analyze(root / 'baseline', root / 'evidence', selection_path, root / 'pending', repetitions=100)
            self.assertEqual(pending['status'], 'pending')
            self.assertIsNone(pending['paired_report_statistics'])
            self.assertIsNone(pending['runs']['evidence']['summary']['metrics'])
            self.assertIsNone(pending['posthoc_rank_only_diagnostic'])


if __name__ == '__main__':
    unittest.main()
