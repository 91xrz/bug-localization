import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bug_localization.report_enrichment import validate, build_queries

class EnrichmentTests(unittest.TestCase):
    def example(self):
        return {'OB': {'text': 'request fails', 'basis': 'explicit', 'evidence': ['fails']},
                'EB': {'text': 'request should succeed', 'basis': 'inferred', 'evidence': ['request fails']},
                'S2R': {'text': '', 'basis': 'missing', 'evidence': []}}

    def test_grounding_and_missing(self):
        value = self.example()
        queries = build_queries('request fails', value)
        self.assertEqual(queries['q0'], 'request fails')
        self.assertEqual(queries['S2R'], '')
        self.assertNotIn('succeed', queries['explicit_only'])
        self.assertIn('succeed', queries['OB_EB_S2R'])
        value['OB']['evidence'] = ['not in report']
        with self.assertRaises(ValueError):
            validate(value, 'request fails')

    def test_inferred_ob_rejected(self):
        value = self.example()
        value['OB']['basis'] = 'inferred'
        with self.assertRaises(ValueError):
            validate(value, 'request fails')

    def test_unmarked_completion_rejected(self):
        value = self.example()
        value['S2R']['text'] = 'Invented command'
        with self.assertRaises(ValueError):
            validate(value, 'request fails')

if __name__ == '__main__':
    unittest.main()

class OfflineFilteringTests(unittest.TestCase):
    def test_keeps_first_response_and_drops_unsupported_section(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from bug_localization.embeddings import digest
        from bug_localization.report_enrichment_audit import cached_with_evidence_filter
        with tempfile.TemporaryDirectory() as folder:
            enricher = SimpleNamespace(identity={'model': 'test'}, folder=Path(folder))
            report = 'request fails'
            sections = EnrichmentTests().example()
            sections['OB']['evidence'] = ['invented quote']
            key = digest({'identity': enricher.identity, 'report': report})
            raw = json.dumps(sections)
            (Path(folder) / (key + '.json')).write_text(json.dumps({
                'identity': enricher.identity, 'report': report, 'finished': True, 'raw': raw}))
            result = cached_with_evidence_filter(enricher, report)
            self.assertEqual(result['raw'], raw)
            self.assertEqual(result['queries']['OB'], '')
            self.assertEqual(result['rejected_sections'][0]['field'], 'OB')
            self.assertIn('succeed', result['queries']['EB'])

    def test_malformed_response_retains_original_and_empty_enrichment(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from bug_localization.embeddings import digest
        from bug_localization.report_enrichment_audit import cached_with_evidence_filter
        with tempfile.TemporaryDirectory() as folder:
            enricher = SimpleNamespace(identity={}, folder=Path(folder))
            key = digest({'identity': {}, 'report': 'bug'})
            (Path(folder) / (key + '.json')).write_text(json.dumps({
                'identity': {}, 'report': 'bug', 'finished': True, 'raw': 'invalid json'}))
            result = cached_with_evidence_filter(enricher, 'bug')
            self.assertEqual(result['queries']['q0'], 'bug')
            self.assertEqual(result['queries']['OB_EB_S2R'], '')
            self.assertIn('generation_error', result)
