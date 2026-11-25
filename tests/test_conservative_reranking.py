import math
import unittest

from bug_localization.conservative_reranking import (
    constrain_proposal, evidence_score, residual_rank,
)


class ConservativeRerankingTests(unittest.TestCase):
    def test_normalization_and_extreme_logits(self):
        self.assertEqual(evidence_score([10000., 10000.], 2.), 10000.)
        self.assertEqual(evidence_score([3.]), 3.)
        self.assertAlmostEqual(evidence_score([3., 3., 3.], 4.), 3.)
        self.assertTrue(math.isfinite(evidence_score([-10000., 10000.], 1.)))
        with self.assertRaises(ValueError):
            evidence_score([float("nan")])

    def test_head_membership_and_tail_survive_large_residual(self):
        base = ["a", "b", "c"]
        out = residual_rank(base, {p: [0.] for p in base},
                            {"a": 0., "b": 1., "c": 100.}, prior_weight=10., head=2)
        self.assertEqual(out, ["b", "a", "c"])

    def test_evidence_constraint_overrides_proposal(self):
        base = ["a", "b", "c"]
        out, trace = constrain_proposal(base, ["c", "b", "a"],
                {"a": [8., 7.], "b": [2.], "c": [1.]},
                {"a": 1., "b": 0., "c": 0.}, margin=4.)
        self.assertEqual(out, ["a", "c", "b"])
        self.assertEqual(trace["blocked_inversions"], 2)

    def test_single_chunk_does_not_count_as_corroboration(self):
        out, trace = constrain_proposal(["a", "b"], ["b", "a"],
                {"a": [8.], "b": [1.]}, {"a": 1., "b": 0.},
                margin=4., corroboration=True)
        self.assertEqual(out, ["b", "a"])
        self.assertEqual(trace["protected_pairs"], 0)

    def test_failure_or_duplicate_is_explicit_abstention(self):
        for proposal in (None, ["a", "a"], ["a"], ["a", "outsider"]):
            out, trace = constrain_proposal(["a", "b"], proposal, {}, {})
            self.assertEqual(out, ["a", "b"])
            self.assertTrue(trace["abstained"])

    def test_all_permutations_preserve_protected_order(self):
        from itertools import permutations
        base = ["a", "b", "c", "d"]
        chunks = {"a": [8., 7.], "b": [5., 4.], "c": [2.], "d": [1.]}
        prior = {"a": 1., "b": .8, "c": .3, "d": .1}
        for proposal in permutations(base):
            out, _ = constrain_proposal(base, proposal, chunks, prior, margin=3.)
            self.assertEqual(set(out), set(base))
            self.assertLess(out.index("a"), out.index("c"))
            self.assertLess(out.index("a"), out.index("d"))
            self.assertLess(out.index("b"), out.index("c"))


if __name__ == "__main__":
    unittest.main()
