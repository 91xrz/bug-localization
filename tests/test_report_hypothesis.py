"""Report-only boundary, hypothesis semantics, and cache-only behavior."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bug_localization.embeddings import digest
from bug_localization.report_hypothesis import FIELDS, HypothesisGenerator, parse_hypothesis, build_queries, report_payload


class HypothesisTests(unittest.TestCase):
    report = 'GET sends a content-length header.'

    def source(self):
        return {'report': self.report, 'sections': {
            'OB': {'text': self.report, 'basis': 'explicit', 'evidence': [self.report]},
            'EB': {'text': '', 'basis': 'missing', 'evidence': []},
            'S2R': {'text': '', 'basis': 'missing', 'evidence': []}},
            'ground_truth': ['secret.py'], 'ranking': [{'file_path': 'secret.py'}],
            'raw': 'unverified enrichment', 'candidate_code': 'source must not enter prompt'}

    def hypothesis(self):
        return {'conflict': 'The request includes an unwanted header.',
                'context': 'HTTP request preparation.',
                'functionality': 'Construct headers from request method and body length.',
                'logic': 'Determine whether an absent body requires a content length field.'}

    def test_prompt_payload_excludes_labels_code_rankings_and_raw(self):
        payload = report_payload(self.report, self.source())
        self.assertEqual(set(payload), {'original_report', 'OB_EB_S2R'})
        self.assertNotIn('secret.py', json.dumps(payload))
        self.assertNotIn('unverified enrichment', json.dumps(payload))

    def test_hypotheses_need_not_be_literal_quotes(self):
        value = parse_hypothesis(json.dumps(self.hypothesis()), True)
        queries = build_queries(self.report, self.source()['sections'], value)
        self.assertEqual(queries['q0'], self.report)
        self.assertEqual(queries['functionality'], value['functionality'])
        self.assertNotIn(value['functionality'], self.report)
        self.assertNotIn(value['conflict'], queries['hypothesis_all'])

    def test_invalid_output_not_accepted(self):
        for raw, finished in [('not json', True), (json.dumps(self.hypothesis()), False),
                              ('{"context": "only one"}', True)]:
            with self.assertRaises(ValueError):
                parse_hypothesis(raw, finished)

    def test_empty_hypothesis_preserves_original_without_faking_enhancement(self):
        queries = build_queries(self.report, self.source()['sections'], {f: '' for f in FIELDS})
        self.assertEqual(queries['hypothesis_all'], '')
        self.assertEqual(queries['original_then_hypothesis'], self.report)
        self.assertEqual(queries['hypothesis_then_original'], self.report)

    def test_duplicate_reports_reuse_only_identical_enrichment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'model').mkdir()
            (root / 'queries').mkdir()
            source = {**self.source(), 'key': 'source-key'}
            for name in ('001_queries.json', '002_queries.json'):
                (root / 'queries' / name).write_text(json.dumps(source))
            config = {'generation_model': 'model', 'enrichment_source': 'queries',
                      'max_new_tokens': 1024, 'max_input_tokens': 12000}
            generator = HypothesisGenerator(root, config)
            self.assertEqual(len(generator.sources), 1)
            self.assertIsNone(generator.model)
            source['sections']['OB']['text'] = 'Different enrichment'
            (root / 'queries/002_queries.json').write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'Conflicting enrichment'):
                HypothesisGenerator(root, config)

    def test_cache_only_never_loads_model_and_records_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            generator = object.__new__(HypothesisGenerator)
            source = self.source()
            generator.sources = {digest(self.report): source}
            generator.identity = {'prompt': 'test'}
            generator.folder = Path(folder)
            with self.assertRaisesRegex(ValueError, 'Missing hypothesis cache'):
                generator.generate(self.report, cache_only=True)
            payload = report_payload(self.report, source)
            key = digest({'identity': generator.identity, 'payload': payload})
            saved = {'identity': generator.identity, 'payload': payload, 'key': key,
                     'raw': 'invalid', 'finished': True}
            (Path(folder) / (key + '.json')).write_text(json.dumps(saved))
            result = generator.generate(self.report, cache_only=True)
            self.assertIn('generation_error', result)
            self.assertEqual(result['queries']['q0'], self.report)
            self.assertEqual(result['queries']['hypothesis_all'], '')
            saved['payload'] = {'original_report': 'tampered'}
            (Path(folder) / (key + '.json')).write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                generator.generate(self.report, cache_only=True)


if __name__ == '__main__':
    unittest.main()
