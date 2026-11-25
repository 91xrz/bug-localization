import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('c3_pilot_analysis', PROJECT / 'scripts/analysis/analyze_c3_improvement_pilot.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class PilotAuditTests(unittest.TestCase):
    def test_joint_head_cannot_remove_files_or_change_tail(self):
        source = [f'{i}.py' for i in range(20)]
        analysis.check_inventory(source[9::-1] + source[10:], source, 10)
        for bad in [source[:-1], source[:9] + [source[0]] + source[10:],
                    source[:10] + source[11:] + [source[10]],
                    source[:9] + [source[10], source[9]] + source[11:]]:
            with self.assertRaises(ValueError):
                analysis.check_inventory(bad, source, 10)

    def test_pending_cohort_never_exposes_subset_metrics(self):
        metrics = analysis.ranking_metrics(['a.py'], ['a.py'], 10)
        row = {'status': 'ok', 'metrics': metrics, 'missing_gold': []}
        result = analysis.summary([row], 2)
        self.assertEqual(result['pending_samples'], 1)
        self.assertIsNone(result['metrics'])
        self.assertNotIn('metrics_successful_only', result)

    def test_result_tampering_is_rejected_even_when_json_remains_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'result.json'
            path.write_text('{"ranking":["a.py"]}')
            record = {'result_file': 'result.json', 'result_sha256': analysis.sha256_file(path)}
            self.assertEqual(analysis.checked_result(root, record)['ranking'], ['a.py'])
            path.write_text('{"ranking":["b.py"]}')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                analysis.checked_result(root, record)

    def test_full_report_ranking_must_follow_actual_chunk_scores(self):
        identity = {'position': 1, 'dataset_row': 0, 'repository': 'r', 'issue_id': 1, 'before_fix_sha': 'abc'}
        evidence = [{'file_path': p, 'chunk_id': p, 'embedding_text': 'code ' + p}
                    for p in ['a.py', 'b.py']]
        saved = {'query': 'short', 'ground_truth': ['a.py'], 'scored_chunks': evidence,
                 'ranking': [{'file_path': p, 'semantic_rank': i + 1} for i, p in enumerate(['a.py', 'b.py'])]}
        query = 'short complete original report'
        for ranking, should_pass in [(['b.py', 'a.py'], True), (['a.py', 'b.py'], False)]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result = {**identity, 'status': 'ok', 'query': query, 'reranker_query_sha256': analysis.digest(query),
                          'baseline_query': 'short', 'selected_chunk_ids': ['a.py', 'b.py'],
                          'scored_chunks': [{**c, 'score': i} for i, c in enumerate(evidence)],
                          'ranking': ranking, 'ground_truth': ['a.py'],
                          'metrics': analysis.ranking_metrics(ranking, ['a.py'], 10)}
                path = root / 'case.json'; path.write_text(json.dumps(result))
                record = {**identity, 'status': 'ok', 'metrics': result['metrics'], 'result_file': 'case.json',
                          'result_sha256': analysis.sha256_file(path)}
                if should_pass:
                    analysis.audit_full_report(root, record, saved, {'query': query})
                else:
                    with self.assertRaisesRegex(ValueError, 'File max aggregation'):
                        analysis.audit_full_report(root, record, saved, {'query': query})


if __name__ == '__main__':
    unittest.main()
