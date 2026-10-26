import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.report_views import QueryGenerator, build_views, report_input, weighted_rrf, filter_evidence
from bug_localization.multiview import load_settings


class ReportViewsTests(unittest.TestCase):
    def setUp(self):
        self.report = {'body': '调用 foo() 时出现 ValueError。\n实际返回空；期望返回列表。'}
        self.value = {'identifiers': [{'source': 'body', 'text': 'foo()'}, {'source': 'body', 'text': 'ValueError'}],
                      'trigger': [{'source': 'body', 'text': '调用 foo()'}],
                      'actual': [{'source': 'body', 'text': '实际返回空'}],
                      'expected': [{'source': 'body', 'text': '期望返回列表'}]}
        self.config = {'endpoint': 'http://127.0.0.1:8080/v1', 'model': 'test', 'model_revision': 'fixture-only',
                       'max_output_tokens': 2048, 'max_report_characters': 24000, 'timeout_seconds': 1}

    def test_input_whitelist(self):
        row = {**self.report, 'title': 'secret PR solution', 'updated_files': '[secret]', 'patch': 'secret'}
        self.assertEqual(report_input(row, ['body']), self.report)

    def test_views_have_literal_evidence_and_unicode_offsets(self):
        views, evidence = build_views(self.report, self.value)
        self.assertEqual(views['q0'], self.report['body'])
        self.assertEqual(views['q1'], 'foo()\nValueError')
        self.assertIn('Expected: 期望返回列表', views['q2'])
        for entries in evidence.values():
            for e in entries:
                self.assertEqual(self.report[e['source']][e['start']:e['end']], e['text'])

    def test_title_order_when_trusted(self):
        report = report_input({'title': 'title', 'body': 'body'}, ['title', 'body'])
        self.assertEqual(build_views(report, {key: [] for key in self.value})[0]['q0'], 'title\nbody')

    def test_hallucinations_fail_closed(self):
        for key in self.value:
            value = {**self.value, key: [{'source': 'body', 'text': 'invented.py'}]}
            with self.assertRaises(ValueError):
                build_views(self.report, value)

    def test_untrusted_field_cannot_supply_evidence(self):
        self.value['identifiers'][0]['source'] = 'title'
        with self.assertRaises(ValueError):
            build_views(self.report, self.value)

    def test_invalid_shapes_and_extra_fields(self):
        for value in ([], {**self.value, 'root_cause': 'guess'}, {**self.value, 'actual': 'guess'},
                      {**self.value, 'actual': [{'source': 'body', 'text': '', 'extra': 1}]}):
            with self.assertRaises(ValueError):
                build_views(self.report, value)

    def test_empty_fields_do_not_invent_queries(self):
        views, _ = build_views(self.report, {key: [] for key in self.value})
        self.assertEqual((views['q1'], views['q2']), ('', ''))
        self.assertEqual(views['q0'], self.report['body'])

    def test_duplicate_quotes_do_not_get_extra_weight(self):
        self.value['identifiers'].append(self.value['identifiers'][0])
        views, evidence = build_views(self.report, self.value)
        self.assertEqual(len(evidence['identifiers']), 2)
        self.assertEqual(views['q1'].count('foo()'), 1)

    def test_invalid_evidence_is_dropped_without_fuzzy_repair(self):
        self.value['identifiers'].append({'source': 'body', 'text': 'invented.py'})
        self.value['expected'] = [{'source': 'body', 'text': '期望返回一个列表'}]
        accepted, rejected = filter_evidence(self.value, self.report)
        views, _ = build_views(self.report, accepted)
        self.assertEqual(len(rejected), 2)
        self.assertNotIn('invented.py', views['q1'])
        self.assertNotIn('Expected:', views['q2'])

    def test_earliest_rejected_response_reused_without_another_call(self):
        with tempfile.TemporaryDirectory() as folder, patch('bug_localization.report_views.build_opener') as factory:
            generator = QueryGenerator(Path(folder), self.config)
            from bug_localization.embeddings import digest
            key = digest({'generator': generator.identity, 'report': self.report})
            rejected = generator.folder / 'rejected'
            rejected.mkdir()
            response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(self.value)}}]}
            (rejected / (key + '_001.json')).write_text(json.dumps({'identity': generator.identity, 'report': self.report,
                                                                  'raw_response': response}), encoding='utf-8')
            saved = generator.generate(self.report, cache_only=True)
            self.assertTrue(saved['recovered_first_response'])
            self.assertEqual(saved['views']['q1'], 'foo()\nValueError')
            factory.assert_not_called()

    def test_weighted_rrf_missing_and_ties(self):
        rankings = {'q0': [{'rank': 1, 'file_path': 'a'}, {'rank': 2, 'file_path': 'b'}],
                    'q1': [{'rank': 1, 'file_path': 'b'}], 'q2': []}
        result = weighted_rrf(rankings, {'q0': 1, 'q1': 2, 'q2': 1})
        self.assertEqual(result[0]['file_path'], 'b')
        self.assertAlmostEqual(result[0]['score'], 1 / 62 + 2 / 61)
        self.assertEqual(result[1]['channels']['q1']['contribution'], 0)
        self.assertEqual(result[0]['channels']['q2']['rank'], None)

    def test_original_cannot_be_disabled_and_weights_finite(self):
        for weights in ({'q0': 0}, {'q0': -1}, {'q0': float('nan')}, {'q0': True}):
            with self.assertRaises(ValueError):
                weighted_rrf({'q0': []}, weights)

    def test_rank_validation_and_zero_weight_absence(self):
        with self.assertRaises(ValueError):
            weighted_rrf({'q0': [{'rank': 2, 'file_path': 'a'}]}, {'q0': 1})
        with self.assertRaises(ValueError):
            weighted_rrf({'q0': [{'rank': 1, 'file_path': 'a'}, {'rank': 2, 'file_path': 'a'}]}, {'q0': 1})
        rows = weighted_rrf({'q0': [{'rank': 1, 'file_path': 'a'}], 'q1': [{'rank': 1, 'file_path': 'b'}]}, {'q0': 1, 'q1': 0})
        self.assertEqual([r['file_path'] for r in rows], ['a'])

    def test_one_request_then_cache_and_model_change_invalidates(self):
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(self.value)}}], 'model': 'test'}
        with tempfile.TemporaryDirectory() as folder, patch('bug_localization.report_views.build_opener') as factory:
            factory.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            generator = QueryGenerator(Path(folder), self.config)
            first = generator.generate(self.report)
            second = generator.generate(self.report, cache_only=True)
            self.assertFalse(first['cache_hit'])
            self.assertTrue(second['cache_hit'])
            self.assertEqual(first['views'], second['views'])
            self.assertEqual(factory.return_value.open.call_count, 1)
            request = factory.return_value.open.call_args.args[0]
            payload = json.loads(request.data)
            self.assertEqual(json.loads(payload['messages'][1]['content']), self.report)
            changed = QueryGenerator(Path(folder), {**self.config, 'model_revision': 'changed'})
            with self.assertRaises(ValueError):
                changed.generate(self.report, cache_only=True)

    def test_truncated_response_is_not_cached(self):
        response = {'choices': [{'finish_reason': 'length', 'message': {'content': json.dumps(self.value)}}]}
        with tempfile.TemporaryDirectory() as folder, patch('bug_localization.report_views.build_opener') as factory:
            factory.return_value.open.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            generator = QueryGenerator(Path(folder), self.config)
            with self.assertRaises(ValueError):
                generator.generate(self.report)
            self.assertEqual(list(generator.folder.glob('*.json')), [])

    def test_report_limit_no_silent_truncation(self):
        with tempfile.TemporaryDirectory() as folder, patch('bug_localization.report_views.build_opener') as factory:
            generator = QueryGenerator(Path(folder), {**self.config, 'max_report_characters': 1})
            with self.assertRaises(ValueError):
                generator.generate(self.report)
            factory.assert_not_called()

    def test_custom_weights_require_development_reference(self):
        config = json.loads((PROJECT / 'configs/experiments/report_multiview.json').read_text(encoding='utf-8-sig'))
        config['weights']['q1'] = 2
        with self.assertRaisesRegex(ValueError, 'development'):
            load_settings(PROJECT, config)


if __name__ == '__main__':
    unittest.main()
