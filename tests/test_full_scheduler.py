import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts/experiments'))
from run_swe_full import build_manifests


class FullSchedulerTests(unittest.TestCase):
    def test_manifests_cover_all_samples_once_in_round_robin_order(self):
        rows = [dict(repo_name=repo, issue_id=i, before_fix_sha=str(i), body='query', updated_files=['a.py'])
                for repo, i in [('b/r', 1), ('a/r', 2), ('b/r', 3), ('a/r', 4), ('a/r', 5)]]
        batches = build_manifests(rows, 'hash', 2)
        self.assertEqual([b['samples'][0]['repository'] for b in batches], ['a/r', 'b/r', 'a/r'])
        entries = [entry for batch in batches for entry in batch['samples']]
        self.assertEqual(sorted(e['dataset_row'] for e in entries), list(range(len(rows))))
        self.assertTrue(all([x['position'] for x in b['samples']] == list(range(1, len(b['samples'])+1)) for b in batches))

    def test_duplicate_reports_rejected(self):
        row = dict(repo_name='a/r', issue_id=1, before_fix_sha='x', body='q', updated_files=[])
        with self.assertRaises(ValueError):
            build_manifests([row, row], 'hash')

if __name__ == '__main__':
    unittest.main()
