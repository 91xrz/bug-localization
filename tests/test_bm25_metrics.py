import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bug_localization.bm25 import ranking_metrics


class RankingMetricsTests(unittest.TestCase):
    def test_multiple_gold_files_use_all_gold_as_ap_denominator(self):
        result = ranking_metrics(['irrelevant', 'a.py', 'other', 'b.py'], ['a.py', 'b.py', 'missing.py'], 4)
        self.assertAlmostEqual(result['average_precision_at_k'], (1 / 2 + 2 / 4) / 3)
        self.assertAlmostEqual(result['recall_at_k'], 2 / 3)
        self.assertEqual(result['reciprocal_rank_at_k'], 0.5)

    def test_cutoff_excludes_later_hits(self):
        result = ranking_metrics(['irrelevant', 'a.py'], ['a.py'], 1)
        self.assertEqual(result['reciprocal_rank_at_k'], 0)
        self.assertEqual(result['top_5'], 0)

    def test_duplicate_results_do_not_inflate_metrics(self):
        result = ranking_metrics(['a.py', 'a.py', 'b.py'], ['a.py', 'b.py'], 3)
        self.assertEqual(result['average_precision_at_k'], 1)
        self.assertEqual(result['recall_at_k'], 1)

    def test_empty_gold_is_an_error(self):
        with self.assertRaises(ValueError):
            ranking_metrics([], [], 10)


if __name__ == '__main__':
    unittest.main()
