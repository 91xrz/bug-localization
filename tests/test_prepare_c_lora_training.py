import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.experiments.prepare_c_lora_training import digest, make_record, pools, source_chunks


class RepairRankingDataTests(unittest.TestCase):
    def test_source_windows_are_selected_by_query_not_label(self):
        query_terms = {"distinctive": 2.0}
        source = "ignored\n" * 400 + "distinctive failure\n" * 7
        chunks, score = source_chunks("a.py", source, query_terms)
        self.assertIn("distinctive", chunks[0])
        self.assertEqual(len(chunks), 2)
        self.assertGreater(score, 0)

    def test_candidates_exclude_all_changed_paths_and_verify_hashes(self):
        with TemporaryDirectory() as tmp:
            sha, repo = "a" * 40, "test/repo"
            snapshot = Path(tmp) / repo / sha
            (snapshot / "source").mkdir(parents=True)
            paths = [f"file_{i}.py" for i in range(10)]
            raw = b"raises an observable exception\n"
            for path in paths:
                (snapshot / "source" / path).write_bytes(raw)
            manifest = {path: hashlib.sha256(raw).hexdigest() for path in paths}
            (snapshot / "snapshot.json").write_text(json.dumps({
                "repository": repo, "commit": sha, "files": manifest}))
            row = {"id": "test-1", "repo": repo, "sha": sha, "query": "exception",
                   "gold": paths[:5] + ["newfile.py"]}
            record, skipped = make_record((row, tmp))
            self.assertIsNone(skipped)
            selected = {f["path"] for f in record["files"]}
            self.assertEqual(len(selected & set(row["gold"])), 3)
            self.assertEqual(len(selected - set(row["gold"])), 4)
            self.assertEqual(set(record["gold"]), set(row["gold"]))
            (snapshot / "source" / paths[-1]).write_bytes(b"post-fix")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                make_record((row, tmp))

    def test_deduplication_and_swe_overlap(self):
        row = {"repo_name": "apache/airflow", "language": "python", "issue_id": 1,
               "updated_files": "['a.py']", "before_fix_sha": "f"*40,
               "body": "original", "after_fix_sha": "fix1"}
        rows = [row, {**row, "issue_id": 2, "body": "  ORIGINAL\n", "after_fix_sha": "fix2"},
                {**row, "issue_id": 3, "body": "another body"},
                {**row, "issue_id": 4, "body": "SWE", "after_fix_sha": "fix3"}]
        out, _ = pools(rows, {"other/repo"}, {digest("swe")})
        self.assertEqual([r["id"] for r in out["apache/airflow"]], ["apache/airflow#1"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            pools(rows, {"apache/airflow"}, set())
