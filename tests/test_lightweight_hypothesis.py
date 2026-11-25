import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from bug_localization.lightweight_hypothesis import anchors, balanced_pool, definitions, rank_variants


class GroundingChecks(unittest.TestCase):
    def test_metadata_noise_does_not_become_evidence(self):
        symbols = {'pkg/a.py': {'only_comment'}, 'pkg/b.py': {'actual_failure'}}
        score, _ = anchors('<!-- only_comment --> https://example.com/only_comment actual_failure', symbols)
        self.assertEqual(score['pkg/a.py'], 0.)
        self.assertGreater(score['pkg/b.py'], 0.)

    def test_source_declarations_not_calls(self):
        self.assertEqual(definitions('class Failure:\n def process_value(self):\n  unknown_call()\n'),
                         {'Failure', 'process_value'})

    def test_nonexistent_hypothesis_symbol_cannot_get_a_match(self):
        score, _ = anchors('invented_function', {'pkg/a.py': {'real_function'}})
        self.assertEqual(score['pkg/a.py'], 0.)

    def test_identifiers_keep_case_and_paths_require_module_boundary(self):
        score, _ = anchors('Classifier pkg.foo_extra', {'pkg/foo.py': {'classifier'}, 'pkg/bar.py': {'Classifier'}})
        self.assertEqual(score['pkg/foo.py'], 0.)
        self.assertGreater(score['pkg/bar.py'], 0.)

    def test_equal_scores_are_finite_and_no_anchor_keeps_B_order(self):
        a = [{'rank':i+1,'file_path':str(i),'score':1.} for i in range(60)]
        b = [{**r,'file_path':str(59-i)} for i,r in enumerate(a)]
        result = rank_variants(a,b,{str(i):0. for i in range(60)},[0.],[.25])
        self.assertEqual(result['score_a0_e0.25'], result['B'])
        self.assertEqual(len(set(balanced_pool(a,b))), 50)
        self.assertTrue({r['file_path'] for r in b[:10]} <= set(balanced_pool(a,b)))


if __name__ == '__main__':
    unittest.main()
