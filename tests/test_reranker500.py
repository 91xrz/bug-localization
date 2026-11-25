import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
from prepare_reranker500 import expand
from run_reranker500 import paired_stats
from bug_localization.batch import aggregate_results
from bug_localization.bm25 import ranking_metrics


class ExpandedRerankerTests(unittest.TestCase):
    def test_frozen_prefix_unique_unseen_bodies_and_deterministic_balancing(self):
        data = [{'repo_name': repo, 'issue_id': i, 'body': f'{repo}-{i}',
                 'before_fix_sha': 'a' * 40, 'updated_files': "['a.py']"}
                for repo in ('a/repo', 'b/repo') for i in range(10)]
        data[2]['body'] = data[0]['body']
        data[3]['body'] = data[4]['body']
        data[11]['body'] = data[1]['body']
        entries = [{'position': i, 'repository': data[index]['repo_name'], 'dataset_row': index}
                   for i, index in enumerate((0, 10), 1)]
        old = {'samples': entries, 'excluded_dataset_rows': [1], 'dataset_metadata_sha256': 'hash', 'selection_sha256': 'oldhash'}
        whole, new = expand(data, old, total=12)
        self.assertEqual(whole['samples'][:2], entries)
        self.assertEqual((whole, new), expand(data, old, total=12))
        bodies = [data[r['dataset_row']]['body'] for r in new['samples']]
        self.assertEqual(len(bodies), len(set(bodies)))
        self.assertFalse(set(bodies) & {data[i]['body'] for i in (0, 1, 10)})
        self.assertEqual(whole['project_counts'], {'a/repo': 6, 'b/repo': 6})
        self.assertEqual([r['position'] for r in whole['samples']], list(range(1, 13)))

    def test_failure_is_zero_and_not_removed_from_paired_denominator(self):
        ok = {'status': 'ok', 'metrics': ranking_metrics(['a.py'], ['a.py'], 10), 'missing_gold': []}
        fail = {'status': 'failed', 'repository': 'repo', 'issue_id': 1, 'error_type': 'Failure', 'error': 'failed'}
        rows = {'original': [ok, ok], 'hypothesis_reranked': [ok, fail]}
        result = paired_stats(rows)['original']
        self.assertEqual(result, {'wins': 0, 'ties': 1, 'losses': 1, 'delta_MAP@10': -0.5})
        self.assertEqual(aggregate_results(rows['hypothesis_reranked'], 10)['metrics']['MAP@10'], 0.5)


if __name__ == '__main__':
    unittest.main()
