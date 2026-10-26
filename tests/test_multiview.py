"""Orchestration tests; mocked scores are never exported as benchmark evidence."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.multiview import run_multiview


class MultiViewRunnerTests(unittest.TestCase):
    def run_case(self, root, empty=False, fail=False):
        row = {'repo_name': 'test/repo', 'issue_id': 1, 'before_fix_sha': 'abc',
               'body': 'original report', 'updated_files': "['a.py']"}
        selection = {'selection_sha256': 'selection', 'samples': [{'position': 1, 'dataset_row': 0,
                      'repository': 'test/repo', 'issue_id': 1, 'before_fix_sha': 'abc'}]}
        semantic = {'file_aggregation': 'max', 'aggregation_temperature': 0.1}
        base = {'endpoint': 'http://localhost:9200', 'cluster_name': 'test', 'query_fields': ['body']}
        config = {'llm': {}, 'semantic_route': 'blagent_ast', 'evaluation_k': 10,
                  'weights': {'q0': 1, 'q1': 1, 'q2': 1}, 'rrf_constant': 60}
        docs = [{'file_path': 'a.py', 'sha256': 'hash'}]
        ranking = [{'rank': 1, 'file_path': 'a.py', 'score': 1.0}]
        generated = {'key': 'query', 'views': {'q0': row['body'], 'q1': '' if empty else 'token',
                                             'q2': '' if empty else 'behavior'}}
        with patch('bug_localization.multiview.load_settings', return_value=(semantic, base, [{'report': row}], selection)), \
             patch('bug_localization.multiview.QueryGenerator') as generator, \
             patch('elasticsearch.Elasticsearch') as es, \
             patch('bug_localization.multiview.LocalEmbedder') as encoder, \
             patch('bug_localization.multiview.DependencyRoutingCache') as cache, \
             patch('bug_localization.multiview.get_snapshot', return_value=(root, {})), \
             patch('bug_localization.multiview.collect_files', return_value=(docs, [])), \
             patch('bug_localization.multiview.retrieve', return_value={'ranking': ranking}) as retrieve, \
             patch('bug_localization.multiview.run_bm25', return_value={'candidate_files': docs, 'index': 'index'}) as bm25, \
             patch('bug_localization.multiview.full_lexical_ranking', return_value=(ranking, {})):
            generator.return_value.generate.return_value = generated
            client = es.return_value.__enter__.return_value
            client.info.return_value = {'cluster_name': 'test', 'version': {'number': '9.5.5'}}
            client.indices.analyze.return_value = {'tokens': [{'token': 'token'}]}
            encoder.return_value.identity = {}
            encoder.return_value.fingerprint = 'model'
            encoder.return_value.query.return_value = (None, {})
            cache.return_value.get.return_value = ([{'embedding_text': 'source'}], [])
            cache.return_value.index_identity.return_value = {}
            if fail:
                retrieve.side_effect = RuntimeError('test failure')
            folder, summary = run_multiview(root, config)
            self.assertEqual(encoder.call_count, 1)
            encoder.return_value.close.assert_called_once()
            generator.return_value.generate.assert_called_once_with({'body': row['body']}, cache_only=True)
            self.assertEqual(row['body'], 'original report')
            return folder, summary, bm25.call_args_list, retrieve.call_count

    def test_reuses_model_and_passes_different_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            _, summary, calls, searches = self.run_case(Path(folder))
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(searches, 2)
            self.assertEqual([c.kwargs['report']['body'] for c in calls], ['original report', 'token'])
            self.assertEqual(summary['routes']['multiview_rrf']['metrics']['MAP@10'], 1)

    def test_empty_views_preserve_q0(self):
        with tempfile.TemporaryDirectory() as folder:
            _, summary, calls, searches = self.run_case(Path(folder), empty=True)
            self.assertEqual(searches, 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(summary['routes']['q1']['metrics']['MAP@10'], 0)
            self.assertEqual(summary['routes']['multiview_rrf']['metrics']['MAP@10'], 1)

    def test_failure_keeps_paired_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            _, summary, _, _ = self.run_case(Path(folder), fail=True)
            self.assertEqual(summary['status'], 'failed')
            for route in summary['routes'].values():
                self.assertEqual(route['failed_samples'], 1)
                self.assertEqual(route['metrics']['MAP@10'], 0)


if __name__ == '__main__':
    unittest.main()
