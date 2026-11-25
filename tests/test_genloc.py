"""Behavioral checks for retrieval boundaries and real exploration orchestration."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / p) for p in ('.runtime/genloc-deps', '.runtime/rrl-deps', '.runtime/igsil-deps', 'src')]
from bug_localization.genloc import explore, shortlist
from bug_localization.batch import aggregate_results
from bug_localization.bm25 import ranking_metrics


class GenLocFlowTests(unittest.TestCase):
    def test_top_chunk_cutoff_precedes_file_deduplication(self):
        chunks = [{'file_path': 'a.py', 'chunk_id': str(i)} for i in range(300)]
        chunks.append({'file_path': 'b.py', 'chunk_id': 'b'})
        files, ranked = shortlist(chunks, list(range(301, 0, -1)))
        self.assertEqual(files, ['a.py'])
        self.assertEqual(len(ranked), 300)

    def test_nonfinite_scores_cannot_silently_become_a_ranking(self):
        with self.assertRaises(ValueError):
            shortlist([{'file_path': 'a.py', 'chunk_id': 'a'}], [float('nan')])

    def test_exploration_can_find_and_read_a_file_outside_initial_retrieval(self):
        from bug_localization.genloc_tools import AuthorCode
        author = AuthorCode()
        files = [{'filepath': 'a.py', 'filename': 'a.py', 'methods': []},
                 {'filepath': 'pkg/target.py', 'filename': 'target.py',
                  'methods': [{'signature': 'fix(self)', 'body': 'def fix(self):\n    return 7'}]}]
        outputs = [
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'c1', 'type': 'function', 'function': {'name': 'search_file', 'arguments': '{"filename":"target.py"}'}}]},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'c2', 'type': 'function', 'function': {'name': 'get_method_body',
                 'arguments': '{"filepath":"pkg/target.py","method_signature":"fix(self)"}'}}]},
            {'role': 'assistant', 'content': json.dumps({'analysis_of_the_bug_report': 'method evidence',
             'ranked_list': [{'file': 'pkg/target.py', 'justification': 'the inspected body returns seven'}]})}]
        seen = []
        def complete(messages, tools, **kwargs):
            seen.append(json.loads(json.dumps({'messages': messages, **kwargs})))
            return outputs.pop(0)
        generator = types.SimpleNamespace(complete=complete)
        with tempfile.TemporaryDirectory() as tmp:
            result = explore('The return value is wrong', files, ['a.py'], author,
                             generator, Path(tmp), {'max_iterations': 9})
            self.assertEqual(result['ranking'], ['pkg/target.py'])
            self.assertEqual(result['ranking_outside_initial_candidates'], ['pkg/target.py'])
            self.assertEqual(result['method_body_calls'], 1)
            self.assertEqual(seen[0]['tool_choice'], 'required')
            self.assertEqual(seen[1]['tool_choice'], 'auto')
            self.assertIn('return 7', seen[2]['messages'][-1]['content'])
            self.assertEqual(len(json.loads((Path(tmp) / 'tool_calls.json').read_text())), 2)

    def test_first_response_cannot_skip_required_tool_call(self):
        from bug_localization.genloc_tools import AuthorCode
        author = AuthorCode()
        generator = types.SimpleNamespace(complete=lambda *a, **k: {'role': 'assistant', 'content': '{}'})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'actual tool call'):
                explore('report', [{'filepath': 'a.py', 'filename': 'a.py', 'methods': []}],
                        ['a.py'], author, generator, Path(tmp), {'max_iterations': 9})

    def test_failed_pilot_report_stays_in_full_denominator(self):
        ok = {'status': 'ok', 'metrics': ranking_metrics(['a.py'], ['a.py', 'b.py'], 10), 'missing_gold': []}
        failed = {'status': 'failed', 'repository': 'x/y', 'issue_id': 1, 'error_type': 'ValueError', 'error': 'bad JSON'}
        summary = aggregate_results([ok, failed], 10, 2)
        self.assertEqual(summary['metrics']['MAP@10'], .25)
        self.assertEqual(summary['metrics']['Top-1'], .5)


if __name__ == '__main__':
    unittest.main()
