import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_ghrb", ROOT / "scripts/datasets/prepare_ghrb.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class GHRBPreparationTests(unittest.TestCase):
    def test_label_preimage_and_unreachable_additions_preserved(self):
        patch = """diff --git a/Old.java b/New.java
--- a/Old.java
+++ b/New.java
@@ -1 +1 @@
--- misleading content
+++ misleading content
diff --git a/Added.java b/Added.java
--- /dev/null
+++ b/Added.java
@@ -0,0 +1 @@
+class Added {}
"""
        labels, details = M.patch_labels(patch)
        self.assertEqual(labels, ["Old.java", "Added.java"])
        self.assertTrue(details[1]["added_file"])

    def test_refuse_unsafe_patch(self):
        with self.assertRaises(ValueError):
            M.patch_labels("diff --git a/../x b/../x\n--- a/../x\n+++ b/../x\n@@ -1 +1 @@\n-x\n+y\n")

    def test_issue_body_is_not_comments_or_pr(self):
        issue = {"number": 7, "repository": {"nameWithOwner": "org/project"},
                 "title": "Original title", "body": "Original **Markdown**", "comments": [{"body": "Patch answer"}]}
        value = {"payload": {"issueViewerRoute": {"data": {"repository": {"issue": issue}}}}}
        page = '<script type="application/json">' + json.dumps(value) + '</script>'
        report = M.extract_issue(page, "https://github.com/org/project/issues/7")
        self.assertEqual(report["body"], "Original **Markdown**")
        self.assertNotIn("comments", report)
        with self.assertRaises(ValueError):
            M.extract_issue(page, "https://github.com/org/project/issues/8")

    def test_repository_move_requires_matching_final_url(self):
        issue = {"number": 7, "repository": {"nameWithOwner": "new/project"}, "title": "Title", "body": "Body"}
        value = {"payload": {"issueViewerRoute": {"data": {"repository": {"issue": issue}}}}}
        page = '<script type="application/json">' + json.dumps(value) + '</script>'
        with self.assertRaises(ValueError):
            M.extract_issue(page, "https://github.com/old/project/issues/7")
        report = M.extract_issue(page, "https://github.com/old/project/issues/7", "https://github.com/new/project/issues/7")
        self.assertEqual(report["canonical_issue_repository"], "new/project")
        with self.assertRaises(ValueError):
            M.extract_issue(page, "https://github.com/old/project/issues/7", "https://github.com/new/project/issues/8")


if __name__ == "__main__":
    unittest.main()
