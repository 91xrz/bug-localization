import copy
import unittest

from scripts.analysis.audit_c_lora_training_round2 import compare_data


def record(negatives):
    files = [{"path": path, "source_sha256": "abc", "chunks": ["source " + path]}
             for path in ["fixed", *negatives]]
    return {"id": "r#1", "repo": "r", "sha": "abc", "dataset_row": 0, "query": "bug",
            "body_sha256": "abc", "gold": ["fixed", "absent"], "available_old_python_gold": ["fixed"],
            "python_paths_in_snapshot": 8, "files": files}


class ComparisonTests(unittest.TestCase):
    def test_only_competing_file_differs(self):
        result = compare_data([record("abcd")], [record("bcde")])
        self.assertEqual(result["replaced_negative_slots"], 1)
        self.assertEqual(result["unchanged_positive_files"], 1)

    def test_report_and_shared_evidence_invariant(self):
        old, new = record("abcd"), record("bcde")
        for mutation in (lambda r: r.update(query="different"),
                         lambda r: r["files"][0].update(chunks=["wrong"]),
                         lambda r: r["files"][1].update(chunks=["wrong"]),
                         lambda r: r["files"].append(r["files"][0])):
            changed = copy.deepcopy(new)
            mutation(changed)
            with self.assertRaises(ValueError):
                compare_data([old], [changed])

    def test_requires_intervention(self):
        with self.assertRaises(ValueError):
            compare_data([record("abcd")], [record("abcd")])


if __name__ == "__main__":
    unittest.main()
