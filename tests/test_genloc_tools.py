"""Behavioral checks against the pinned, unmodified GenLoc CPU modules."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
if (PROJECT / ".runtime/genloc-deps").is_dir():
    sys.path.insert(0, str(PROJECT / ".runtime/genloc-deps"))

from bug_localization.genloc_tools import (
    AUTHOR_HASHES, DEFAULT_CHECKOUT, REVISION, AuthorCode, Toolbox,
    build_chunks, build_file_data, normalize_ranking,
)

CHECKOUT = Path(os.environ.get("GENLOC_AUTHOR_CHECKOUT", str(DEFAULT_CHECKOUT)))


@unittest.skipUnless(CHECKOUT.is_dir(), "Pinned GenLoc author checkout is not prepared")
class GenLocAuthorToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.author = AuthorCode(CHECKOUT)
        cls.documents = [
            {"file_path": "pkg/initial.py", "content": "def start():\n    return target()\n"},
            {"file_path": "pkg/other.py", "content": "def target(x=1):\n    return x + 27\n"},
            {"file_path": "pkg/constants.py", "content": "VALUE = 42\n"},
            {"file_path": "tests/test_target.py", "content": "def test_target():\n    assert True\n"},
        ]
        cls.data = build_file_data(cls.documents, author=cls.author)

    def final(self, *paths):
        return {"analysis_of_the_bug_report": "Analysis", "ranked_list": [
            {"file": path, "justification": "Evidence"} for path in paths
        ]}

    def test_original_prompts_schema_and_actual_round_policy_are_preserved(self):
        self.assertEqual(self.author.provenance["commit"], REVISION)
        self.assertEqual(self.author.provenance["source_sha256"], AUTHOR_HASHES)
        self.assertEqual(len(self.author.tools), 5)
        self.assertIn("10th iteration", self.author.system_prompt)
        self.assertEqual(self.author.iteration_policy["max_iterations"], 10)
        self.assertEqual(self.author.iteration_policy["forced_final_iteration_index"], 8)
        self.assertTrue(self.author.response_format["json_schema"]["strict"])
        self.assertEqual(self.author.response_format["json_schema"]["schema"]["required"],
                         ["analysis_of_the_bug_report", "ranked_list"])

    def test_author_loader_does_not_replace_generic_python_modules(self):
        sentinel = object()
        before = {name: sys.modules.get(name, sentinel) for name in (
            "file_parser", "file_data_processor", "post_processor", "bug_report_processor",
        )}
        self.author.module("file_parser")
        self.author.module("file_data_processor")
        self.author.module("post_processor")
        after = {name: sys.modules.get(name, sentinel) for name in before}
        self.assertEqual(before, after)
        with self.assertRaises(ValueError):
            self.author.module("bug_report_processor")

    def test_tools_search_outside_initial_candidates_and_return_actual_method_body(self):
        tools = Toolbox(self.data, ["pkg/initial.py"], author=self.author)
        self.assertEqual(tools.execute("get_candidate_files", {}), ["pkg/initial.py"])
        found = tools.execute("search_method", {"method_name": "target"})
        self.assertEqual(found[0]["filepath"], "pkg/other.py")
        signatures = tools.execute("get_method_signatures_of_a_file", {"filepath": "pkg/other.py"})
        self.assertEqual(signatures["method signatures"], ["target(x=1)"])
        body = tools.execute("get_method_body", {
            "filepath": "pkg/other.py", "method_signature": "target(x=1)",
        })
        self.assertEqual(body[0]["method body"], "def target(x=1):\n    return x + 27")
        self.assertIn(body[0]["method body"], self.documents[1]["content"])
        fuzzy = tools.execute("get_method_body", {
            "filepath": "other.py", "method_signature": "target(x=2)",
        })
        self.assertEqual(fuzzy[0]["method body"], body[0]["method body"])

    def test_common_pool_retains_tests_and_no_function_files(self):
        self.assertEqual([item["filepath"] for item in self.data],
                         [item["file_path"] for item in self.documents])
        tools = Toolbox(self.data, [], author=self.author)
        self.assertEqual(tools.execute("search_file", {"filename": "constants.py"})[0]["filepath"],
                         "pkg/constants.py")
        self.assertEqual(tools.execute("get_method_signatures_of_a_file", {
            "filepath": "pkg/constants.py",
        })["method signatures"], [])
        self.assertEqual(tools.execute("search_file", {"filename": "test_target.py"})[0]["filepath"],
                         "tests/test_target.py")
        chunks = build_chunks(self.data)
        self.assertNotIn("pkg/constants.py", [chunk["file_path"] for chunk in chunks])
        self.assertIn("tests/test_target.py", [chunk["file_path"] for chunk in chunks])

    def test_author_signature_collision_behavior_is_not_silently_changed(self):
        source = "class A:\n    def run(self):\n        return 1\n\nclass B:\n    def run(self):\n        return 2\n"
        data = build_file_data([{"file_path": "classes.py", "content": source}], author=self.author)
        self.assertEqual(len(data[0]["methods"]), 1)
        self.assertEqual(data[0]["methods"][0]["signature"], "run(self)")
        self.assertIn("return 2", data[0]["methods"][0]["body"])

    def test_unsafe_and_unknown_tool_paths_cannot_read_external_files(self):
        tools = Toolbox(self.data, [], author=self.author)
        for path in ("../other.py", "/etc/passwd", "pkg/../../other.py", "C:\\other.py"):
            self.assertIn("error", tools.execute("get_method_signatures_of_a_file", {"filepath": path}))
            self.assertIn("error", tools.execute("search_file", {"filename": path}))
        self.assertIn("error", tools.execute("get_method_signatures_of_a_file", {"filepath": "missing.py"}))
        self.assertIn("error", tools.execute("search_file", {"filename": "other.py", "extra": "x"}))
        self.assertIn("error", tools.execute("search_method", {"method_name": []}))
        self.assertIn("error", tools.execute("unknown", {}))
        with self.assertRaises(ValueError):
            Toolbox(self.data, ["outside.py"], author=self.author)
        with self.assertRaises(ValueError):
            build_file_data([{"file_path": "../outside.py", "content": ""}], author=self.author)

    def test_corrected_path_duplicates_do_not_inflate_final_metrics(self):
        paths, audit = normalize_ranking(
            self.final("other.py", "pkg/other.py", "pkg/constants.py", "missing.py", "../initial.py"),
            self.data, author=self.author,
        )
        self.assertEqual(paths, ["pkg/other.py", "pkg/constants.py"])
        self.assertEqual(audit["entries"][1]["action"], "removed_duplicate_corrected_path")
        self.assertEqual(audit["entries"][3]["action"], "removed_unknown_path")
        self.assertEqual(audit["entries"][4]["action"], "removed_unsafe_path")
        from bug_localization.bm25 import ranking_metrics
        self.assertEqual(ranking_metrics(paths, ["pkg/other.py"], 10)["average_precision_at_k"], 1.0)

    def test_jaccard_recovery_uses_only_the_shared_source_pool(self):
        data = [
            {"filepath": "pkg/a/other.py", "filename": "other.py", "methods": []},
            {"filepath": "pkg/b/other.py", "filename": "other.py", "methods": []},
        ]
        paths, _ = normalize_ranking(self.final("pkg/b/internal/other.py"), data, author=self.author)
        self.assertEqual(paths, ["pkg/b/other.py"])

    def test_final_output_rejects_invalid_structure_and_no_usable_paths(self):
        for final in ({}, [], self.final(), self.final("missing.py"), self.final("../other.py"),
                      {"analysis_of_the_bug_report": "", "ranked_list": [{"file": "pkg/other.py"}]},
                      {"analysis_of_the_bug_report": "", "ranked_list": "pkg/other.py"}):
            with self.assertRaises(ValueError):
                normalize_ranking(final, self.data, author=self.author)
        data = [{"filepath": f"p{i}.py", "filename": f"p{i}.py", "methods": []} for i in range(12)]
        paths, audit = normalize_ranking(self.final(*(f"p{i}.py" for i in range(12))), data, author=self.author)
        self.assertEqual(len(paths), 10)
        self.assertEqual(audit["truncated_paths"], ["p10.py", "p11.py"])

    def test_chunks_apply_token_budget_before_author_path_prefix(self):
        import tiktoken
        body = "def long_function():\n" + "\n".join(f"    value_{i} = {i}" for i in range(150))
        data = [{"filepath": "pkg/long.py", "filename": "long.py", "methods": [{"signature": "long_function()", "body": body}]}]
        chunks = build_chunks(data)
        self.assertGreater(len(chunks), 1)
        encoding = tiktoken.get_encoding("cl100k_base")
        for chunk in chunks:
            self.assertLessEqual(len(encoding.encode(chunk["code"])), 300)
            self.assertEqual(chunk["embedding_text"], "file: pkg/long.py\n" + chunk["code"])
            self.assertIn(chunk["code"], body)
        self.assertEqual(chunks, build_chunks(data))
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))

    def test_changed_author_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            checkout = Path(temp)
            source = checkout / "source-code/GenLoc-Python"
            source.mkdir(parents=True)
            for name in AUTHOR_HASHES:
                (source / name).write_bytes((CHECKOUT / "source-code/GenLoc-Python" / name).read_bytes())
            parser = source / "file_parser.py"
            parser.write_bytes(parser.read_bytes() + b"\n# tampered\n")
            with patch("bug_localization.genloc_tools.subprocess.check_output", return_value=REVISION + "\n"):
                with self.assertRaisesRegex(ValueError, "Modified GenLoc author source"):
                    AuthorCode(checkout)


if __name__ == "__main__":
    unittest.main()
