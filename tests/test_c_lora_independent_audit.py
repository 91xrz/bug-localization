import unittest

from scripts.analysis.audit_c_lora import accuracy, ap10, looks_like_test, pair_counts, score_map


class IndependentAuditTest(unittest.TestCase):
    def test_tie_stable_file_max(self):
        chunks = [{"file_path": p, "chunk_id": i} for p, i in [("b", 1), ("a", 2), ("a", 3)]]
        source = {"original_order": [{"file_path": p} for p in ("b", "a")], "chunks": chunks}
        scores = [{**c, "score": s} for c, s in zip(chunks, [1, 0, 1])]
        result = {"ranking": [{"file_path": p, "score": 1} for p in ("b", "a")], "scored_chunks": scores}
        maxima, order = score_map(source, result)
        self.assertEqual(order, ["b", "a"])
        self.assertEqual(accuracy(pair_counts(["a"], ["b"], maxima)), .5)
        result["ranking"].reverse()
        with self.assertRaises(ValueError):
            score_map(source, result)

    def test_missing_gold_counts_in_ap(self):
        self.assertEqual(ap10(["wrong", "hit"], {"hit", "absent"}), .25)
        self.assertIsNone(accuracy({}))

    def test_weak_test_proxy(self):
        self.assertTrue(looks_like_test("src/tests/fixture.py"))
        self.assertTrue(looks_like_test("src/x_test.py"))
        self.assertFalse(looks_like_test("src/testament.py"))


if __name__ == "__main__":
    unittest.main()
