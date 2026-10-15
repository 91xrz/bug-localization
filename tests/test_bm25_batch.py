import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bug_localization.batch import aggregate_results, select_samples, write_checkpoints
from bug_localization.bm25 import ranking_metrics


class BatchEvaluationTests(unittest.TestCase):
    def successes(self):
        return [{'status': 'ok', 'repository': 'test/repo', 'issue_id': 1,
                 'metrics': ranking_metrics(['a.py'], ['a.py'], 50), 'missing_gold': []},
                {'status': 'ok', 'repository': 'test/repo', 'issue_id': 2,
                 'metrics': ranking_metrics(['wrong.py', 'b.py'], ['b.py', 'c.py'], 50), 'missing_gold': ['c.py']}]

    def test_failed_sample_is_not_silently_removed_from_primary_mean(self):
        records = self.successes() + [{'status': 'failed', 'repository': 'test/repo', 'issue_id': 3,
                                      'error_type': 'TimeoutError', 'error': 'snapshot unavailable'}]
        summary = aggregate_results(records, 50)
        self.assertAlmostEqual(summary['metrics']['MAP@50'], (1 + 0.25) / 3)
        self.assertAlmostEqual(summary['metrics']['MRR@50'], 0.5)
        self.assertAlmostEqual(summary['metrics']['Top-1'], 1 / 3)
        self.assertEqual(summary['metrics_successful_only']['MAP@50'], 0.625)
        self.assertEqual(summary['failed_samples'], 1)
        self.assertAlmostEqual(summary['coverage']['candidate_coverage'], 2 / 3)

    def test_pending_samples_do_not_produce_final_metrics(self):
        summary = aggregate_results(self.successes()[:1], 50, selected_count=3)
        self.assertEqual(summary['status'], 'running')
        self.assertEqual(summary['pending_samples'], 2)
        self.assertIsNone(summary['metrics'])

    def test_all_failures_are_distinguished_from_a_successful_zero_score(self):
        record = {'status': 'failed', 'repository': 'test/repo', 'issue_id': 1,
                  'error_type': 'ValueError', 'error': 'no query'}
        summary = aggregate_results([record], 50)
        self.assertEqual(summary['metrics']['MAP@50'], 0)
        self.assertIsNone(summary['metrics_successful_only'])
        self.assertIsNone(summary['coverage']['candidate_coverage'])

    def test_mixed_cutoffs_cannot_be_averaged(self):
        records = self.successes()
        records[0]['metrics']['k'] = 10
        with self.assertRaises(ValueError):
            aggregate_results(records, 50)

    def test_sample_selection_preserves_dataset_order(self):
        rows = [{'repo_name': 'other/repo', 'issue_id': 5}, {'repo_name': 'test/repo', 'issue_id': 9},
                {'repo_name': 'test/repo', 'issue_id': 2}, {'repo_name': 'test/repo', 'issue_id': 1}]
        selected = select_samples(rows, ['test/repo'], 2)
        self.assertEqual([s['report']['issue_id'] for s in selected], [9, 2])
        self.assertEqual([s['dataset_row'] for s in selected], [1, 2])

    def test_duplicate_selected_issues_are_an_error(self):
        rows = [{'repo_name': 'test/repo', 'issue_id': 1}] * 2
        with self.assertRaises(ValueError):
            select_samples(rows, ['test/repo'], 2)

    def test_csv_and_json_export_failure_details(self):
        records = self.successes()[:1] + [{'status': 'failed', 'repository': 'test/repo', 'issue_id': 3,
                                         'error_type': 'ValueError', 'error': 'bad, label\nwith details'}]
        summary = aggregate_results(records, 50)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            write_checkpoints(folder, records, summary)
            with (folder / 'samples.csv').open(encoding='utf-8-sig', newline='') as handle:
                output = list(csv.DictReader(handle))
            self.assertEqual(output[1]['error'], 'bad, label\nwith details')
            self.assertEqual(output[1]['AP@K'], '')
            self.assertEqual(output[0]['K'], '50')
            self.assertEqual(json.loads((folder / 'summary.json').read_text())['metrics']['MAP@50'], 0.5)


if __name__ == '__main__':
    unittest.main()
