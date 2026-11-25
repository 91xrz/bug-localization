from pathlib import Path
import sys
import unittest
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts/analysis'))
from analyze_c3_uncertainty import intervals


class UncertaintyTests(unittest.TestCase):
    def test_singleton_projects_retain_sampling_uncertainty(self):
        result = intervals([-1., 0., 0., 1.], ['a', 'b', 'c', 'd'], repetitions=2000)
        self.assertEqual(result['singleton_projects'], 4)
        for key in ('ci95_report_paired_bootstrap', 'ci95_project_cluster_bootstrap'):
            self.assertLess(result[key][0], 0)
            self.assertGreater(result[key][1], 0)

    def test_constant_delta_and_reproducibility(self):
        args = ([.25, .25, .25], ['a', 'a', 'b'])
        result = intervals(*args, repetitions=100)
        self.assertEqual(result['ci95_project_cluster_bootstrap'], [.25, .25])
        self.assertEqual(result, intervals(*args, repetitions=100))

    def test_invalid_alignment(self):
        with self.assertRaises(ValueError):
            intervals([1., 2.], ['a'])
        with self.assertRaises(ValueError):
            intervals([], [])

if __name__ == '__main__':
    unittest.main()
