import unittest
from bug_localization.evidence_packets import file_packets


class EvidencePacketsTests(unittest.TestCase):
    def test_order_invariance_and_no_metadata_leak(self):
        a = dict(file_path="a.py", chunk_id="a", embedding_text="def a(): pass", ground_truth="secret")
        b = dict(file_path="a.py", chunk_id="b", embedding_text="def b(): pass", score=900)
        self.assertEqual(file_packets([a, b]), file_packets([b, a]))
        text = file_packets([a, b])[0]["document"]
        self.assertNotIn("secret", text)
        self.assertNotIn("900", text)
        self.assertIn("def a(): pass", text)
        self.assertIn("def b(): pass", text)

    def test_duplicate_rejected(self):
        c = dict(file_path="a.py", chunk_id="a", embedding_text="pass")
        with self.assertRaises(ValueError):
            file_packets([c, c])


if __name__ == "__main__":
    unittest.main()
