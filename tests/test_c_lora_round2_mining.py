from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
from mine_c_lora_round2 import pick


def file(path):
    return {"path": path, "source_sha256": "h" + path, "chunks": ["code:" + path]}


class TestMining(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "repo#1", "gold": ["p", "q", "outside"],
                    "files": [file(p) for p in ("a", "b", "c", "d", "p", "q")]}
        self.pool = {**self.row, "files": self.row["files"] + [file("e"), file("f")]}

    def test_true_mistakes_and_fallback_not_conflated(self):
        values = dict(p=3, q=5, a=4, b=1, c=0, d=1, e=6, f=3)
        _, audit = pick(self.row, self.pool, values)
        self.assertEqual(audit["selected_negatives"], ["e", "a", "f", "b"])
        self.assertEqual(audit["new_above_any_gold"], 2)
        self.assertEqual(audit["new_above_all_gold"], 1)
        self.assertEqual(audit["n_replaced"], 2)

    def test_all_positive_evidence_preserved(self):
        values = {f["path"]: 1 for f in self.pool["files"]}
        new, audit = pick(self.row, self.pool, values)
        self.assertEqual(audit["new_above_any_gold"], 0)
        self.assertEqual(audit["selected_negatives"], ["a", "b", "c", "d"])
        self.assertEqual({f["path"] for f in new["files"]}, set("abcdpq"))
        self.assertEqual(new["gold"], self.row["gold"])

    def test_changed_evidence_rejected(self):
        values = {f["path"]: 1 for f in self.pool["files"]}
        self.pool["files"][0] = dict(file("a"), chunks=["different code"])
        with self.assertRaisesRegex(ValueError, "evidence"):
            pick(self.row, self.pool, values)


if __name__ == "__main__":
    unittest.main()
