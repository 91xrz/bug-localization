"""Checks that baseline comparison cannot silently change candidates or denominators."""
import importlib.util
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
spec = importlib.util.spec_from_file_location('baselines500', ROOT / 'scripts/experiments/run_recall_baselines500.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BaselinesTest(unittest.TestCase):
    def test_candidate_content_mismatch_rejected(self):
        metrics = module.ranking_metrics(['a.py'], ['a.py'], 10)
        result = {'candidate_files': [{'file_path': 'a.py', 'sha256': 'changed'}],
                  'ground_truth': ['a.py'], 'results': [{'file_path': 'a.py'}], 'metrics': metrics}
        with self.assertRaisesRegex(ValueError, 'inventories differ'):
            module.verify_result(result, {'ground_truth': ['a.py']},
                                 {'files': [{'file_path': 'a.py', 'sha256': 'original'}]})

    def test_failure_keeps_full_denominator_and_pending_is_not_final(self):
        ok = {'repository': 'repo', 'issue_id': 1, 'status': 'ok', 'missing_gold': [],
              'metrics': module.ranking_metrics(['a.py'], ['a.py'], 10)}
        failed = {'repository': 'repo', 'issue_id': 2, 'status': 'failed',
                  'error_type': 'InputTooLong', 'error': 'input limit'}
        records = {'L0': [ok] * 499, 'S0': [ok] * 500, 'H0': [ok] * 499 + [failed]}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.assertEqual(module.summarize(folder, records), 'running')
            summary = module.read_json(folder / 'final_summary.json')['cohorts']
            self.assertIsNone(summary['all500']['L0']['metrics'])
            self.assertEqual(summary['all500']['H0']['metrics']['MAP@10'], 499/500)
            self.assertEqual(summary['new320']['H0']['metrics']['MAP@10'], 319/320)
            records['L0'].append(ok)
            self.assertEqual(module.summarize(folder, records), 'completed_with_failures')


if __name__ == '__main__':
    unittest.main()
