import sys
import tempfile
import unittest
from pathlib import Path
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments"))
from evaluate_c_lora_round3 import REPOS, joint_disjoint, normalized, select


class HeldoutProtocol(unittest.TestCase):
    def test_abstains_from_excluded_issues_and_bodies(self):
        rows = [dict(repository=REPOS[0],issue_id=i,before_fix_sha=str(i),dataset_row=i,
                     status="failed" if i==3 else "ok") for i in range(4)]
        old = [rows[0],dict(rows[1],issue_id=99,dataset_row=9)]
        rows[1]["body"] = rows[2]["body"] = "same"
        old[-1]["body"] = "same"
        key = lambda r:r.get("body",str(r["dataset_row"]))
        self.assertEqual(select(rows,old,bodykey=key),[rows[3]])
        self.assertEqual(select(rows[::-1],old,bodykey=key),[rows[3]])

    def test_caps_per_repo_without_dropping_short_project(self):
        rows = [dict(repository=repo,issue_id=i,dataset_row=i,before_fix_sha=str(i))
                for repo,n in zip(REPOS,(8,3,8,8,8)) for i in range(n)]
        self.assertEqual(len(select(rows,[],limit=5)),23)

    def test_overlap_in_original_body_rejected_even_when_query_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            for k in ("train","validation"):
                (data/(k+".jsonl")).write_text(json.dumps({"repo":"unseen/repo","query":"short",
                    "normalized_body_sha256":normalized(" case   text ")}))
            with self.assertRaises(ValueError):
                joint_disjoint(data,[{"repository":"other/project","normalized_sha256":normalized("Case Text")}])
