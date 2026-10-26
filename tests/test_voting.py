import itertools
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.bm25 import ranking_metrics
from bug_localization.voting import expected_tie_metrics, vote_files


def files(*paths):
    return [{'file_path': path} for path in paths]


class VotingTests(unittest.TestCase):
    def test_votes_ignore_positions_scores_and_duplicates_inside_a_route(self):
        a = {'bm25': files('b.py', 'a.py', 'a.py'), 'ast': files('c.py', 'a.py')}
        b = {'bm25': files('a.py', 'b.py'), 'ast': files('a.py', 'c.py')}
        first = vote_files(a)
        for row in b['bm25'] + b['ast']:
            row.update({'rank': -100, 'score': float('nan')})
        self.assertEqual(first, vote_files(b))
        self.assertEqual([(r['file_path'], r['votes']) for r in first], [('a.py', 2), ('b.py', 1), ('c.py', 1)])
        self.assertEqual((first[1]['tie_rank_start'], first[1]['tie_rank_end']), (2, 3))

    def test_topn_is_membership_selection_not_a_rank_bonus(self):
        a = {'bm25': files('a.py', 'b.py', 'c.py'), 'ast': files('b.py', 'a.py', 'd.py')}
        b = {'bm25': files('b.py', 'a.py', 'c.py'), 'ast': files('a.py', 'b.py', 'd.py')}
        self.assertEqual(vote_files(a, 2), vote_files(b, 2))
        self.assertEqual([r['votes'] for r in vote_files(a, 2)], [2, 2])
        for depth in [True, 0, -1, 1.5]:
            with self.assertRaises(ValueError):
                vote_files(a, depth)

    def test_exact_tie_expectation_matches_exhaustive_orderings_with_missing_gold(self):
        ranking = vote_files({'a': files('a', 'b', 'c', 'd', 'e'), 'b': files('a', 'b', 'c')})
        gold = ['b', 'e', 'missing']
        reference = [ranking_metrics(list(x) + list(y), gold, 10)
                     for x in itertools.permutations(['a', 'b', 'c']) for y in itertools.permutations(['d', 'e'])]
        actual = expected_tie_metrics(ranking, gold)
        for key in reference[0]:
            with self.subTest(metric=key):
                self.assertAlmostEqual(actual[key], sum(r[key] for r in reference) / len(reference))

    def test_expected_metrics_respect_top10_boundary_inside_large_tie(self):
        ranking = vote_files({'a': files(*(str(i) for i in range(12)))})
        result = expected_tie_metrics(ranking, ['3'])
        self.assertAlmostEqual(result['top_1'], 1 / 12)
        self.assertAlmostEqual(result['top_5'], 5 / 12)
        self.assertAlmostEqual(result['top_10'], 10 / 12)
        self.assertAlmostEqual(result['recall_at_k'], 10 / 12)
        self.assertAlmostEqual(result['reciprocal_rank_at_k'], sum(1 / i for i in range(1, 11)) / 12)
        self.assertEqual(expected_tie_metrics([], ['missing'])['top_10'], 0)


if __name__ == '__main__':
    unittest.main()
