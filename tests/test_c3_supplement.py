import copy
from pathlib import Path
import sys
import unittest
import numpy as np
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.paired_statistics import inference, align, values, analyze, METRICS


def rows():
    return [{'dataset_row': i, 'repository': 'a' if i < 2 else 'b', 'issue_id': i,
             'before_fix_sha': 'sha', 'query_sha256': str(i), 'labels_sha256': str(i),
             'status': 'ok', 'metrics': {m: 0. for m in METRICS}} for i in range(4)]


class PairedTests(unittest.TestCase):
    def test_alignment_rejects_reordered_and_duplicate(self):
        a = rows()
        with self.assertRaises(ValueError):
            align({'a': a, 'b': a[::-1]})
        with self.assertRaises(ValueError):
            align({'a': [a[0], a[0]]})

    def test_failure_is_zero_even_with_stale_metrics(self):
        a = rows(); a[0]['status'] = 'failed'
        a[0]['metrics']['top_1'] = 1
        self.assertEqual(values(a, 'top_1').sum(), 0)

    def test_zero_contrast_and_seed(self):
        a = inference([0., 0., 0.], ['a', 'b', 'b'], repetitions=100)
        self.assertEqual(a['p_two_sided_paired_randomization'], 1.)
        self.assertEqual(a['ci95_project_stratified_bootstrap'], [0., 0.])
        self.assertEqual(a, inference([0., 0., 0.], ['a', 'b', 'b'], repetitions=100))

    def test_project_contribution_and_sensitivity(self):
        a = inference([1., 1., -.2, -.2], ['a', 'a', 'b', 'b'], repetitions=100)
        self.assertAlmostEqual(sum(p['contribution_to_overall_delta'] for p in a['projects'].values()), a['delta'])
        self.assertAlmostEqual(a['leave_one_project_out']['a'], -.2)
        self.assertAlmostEqual(a['leave_one_project_out']['b'], 1.)

    def test_factorial_interaction_and_holm(self):
        methods = {m: rows() for m in ('A', 'A3', 'C1', 'C3')}
        for m, v in [('A', .1), ('A3', .2), ('C1', .3), ('C3', .4)]:
            for row in methods[m]:
                row['metrics'] = {metric: v for metric in METRICS}
        result = analyze(methods, {'interaction': {'C3': 1, 'C1': -1, 'A3': -1, 'A': 1},
                                  'gain': {'C3': 1, 'A': -1}}, repetitions=100)
        self.assertAlmostEqual(result['contrasts']['interaction']['top_1']['delta'], 0.)
        self.assertEqual(result['multiplicity_family_size'], 12)
        for c in result['contrasts'].values():
            for s in c.values():
                self.assertGreaterEqual(s['p_holm_all_contrasts_and_metrics'], s['p_two_sided_paired_randomization'])

if __name__ == '__main__':
    unittest.main()
