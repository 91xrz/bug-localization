import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bug_localization.structural_matching import anchors, extract_symbols, rank_candidates, score_sources


class StructuralMatchingTests(unittest.TestCase):
    def test_existing_ten_language_fixtures_and_kotlin(self):
        for p in (ROOT / "tests/fixtures/multilingual").iterdir():
            with self.subTest(path=p.name):
                symbols = extract_symbols("pkg/" + p.name, p.read_text())
                self.assertIn("Cache", symbols.names)
                self.assertTrue({"get", "Get"} & symbols.names)
                self.assertFalse(symbols.parse_has_error)
        symbols = extract_symbols("src/Cache.kt", "package org.demo\nclass Cache {\n fun readValue() {}\n}\n")
        self.assertIn("readValue", symbols.names)
        self.assertIn("org.demo.Cache.readValue", symbols.qualified_names)

    def test_namespaces_receivers_and_out_of_class_definitions(self):
        cases = [
            ("Demo.java", "package org.demo; public class Cache { public Cache() {} public void readValue() {} }", "org.demo.Cache.readValue"),
            ("Demo.cs", "namespace Org.Demo; public class Cache { public void ReadValue() {} }", "Org.Demo.Cache.ReadValue"),
            ("demo.cpp", "namespace demo { class Cache { public: void readValue(); }; void Cache::readValue() {} }", "demo::Cache::readValue"),
            ("demo.go", "package demo\ntype Cache struct {}\nfunc (c *Cache) ReadValue() {}", "demo.Cache.ReadValue"),
            ("demo.rs", "mod demo { struct Cache {} impl Cache { fn read_value(&self) {} } }", "demo::Cache::read_value"),
            ("demo.ts", "namespace Demo { export interface Cache { readValue(): void; } }", "Demo.Cache.readValue"),
            ("pkg/demo.py", "class Cache:\n def read_value(self): pass\n", "pkg.demo.Cache.read_value")]
        for path, source, expected in cases:
            with self.subTest(path=path):
                self.assertIn(expected, extract_symbols(path, source).qualified_names)

    def test_no_calls_comments_imports_or_string_declarations(self):
        cases = [
            ("demo.py", 'from hidden import Imported\ntext = """\nclass Bogus:\n def fake_method(): pass\n"""\ndef real_method():\n called_only()\n'),
            ("Demo.java", 'import hidden.Imported; /* class Bogus {} */ class Real { void real_method() { called_only(); String s="class Fake {}"; } }'),
            ("demo.js", 'import Imported from "hidden"; /* function Bogus() {} */ function real_method() { called_only(); }')]
        for path, source in cases:
            names = extract_symbols(path, source).names
            self.assertIn("real_method", names)
            self.assertFalse(names & {"Imported", "Bogus", "fake_method", "called_only", "Fake"})

    def test_named_js_callbacks_but_not_bare_callback_or_regular_variable(self):
        names = extract_symbols("demo.js", "const loadValue=()=>1; const processValue=function() {}; const ordinary=3; things.map(()=>0);").names
        self.assertEqual(names, {"loadValue", "processValue"})

    def test_c_prototype_is_evidence_but_struct_reference_is_not(self):
        names = extract_symbols("demo.c", "struct Existing *value; void readValue(void); int loadValue() { return 1; }").names
        self.assertEqual(names, {"readValue", "loadValue"})

    def test_fully_qualified_name_disambiguates_identical_short_names(self):
        sources = {"a/Cache.java": "package org.first; class Cache { void readValue() {} }",
                   "b/Cache.java": "package org.second; class Cache { void readValue() {} }"}
        for report in ("org.first.Cache.readValue", "org.first.Cache#readValue", "org.first.Cache"):
            scores, _ = score_sources(report, sources)
            self.assertGreater(scores["a/Cache.java"], 0)
            self.assertEqual(scores["b/Cache.java"], 0)
        self.assertEqual(score_sources("org.third.Cache.readValue", sources)[0], dict.fromkeys(sources, 0.))
        self.assertEqual(score_sources("org.first", sources)[0], dict.fromkeys(sources, 0.))

    def test_paths_have_boundaries_and_allow_windows_and_line_numbers(self):
        sources = {"pkg/Foo.java": "class Foo {}", "other/Bar.java": "class Bar {}"}
        for report in ("pkg/Foo.java:123", r"pkg\Foo.java", "`pkg/Foo.java`", "pkg/Foo.java."):
            self.assertEqual(score_sources(report, sources)[0]["pkg/Foo.java"], 1.)
        for report in ("otherpkg/Foo.java", "pkg/Foo.java_extra", "https://example.com/pkg/Foo.java", "<!-- pkg/Foo.java -->"):
            self.assertEqual(score_sources(report, sources)[0]["pkg/Foo.java"], 0.)

    def test_idf_formula_and_zero_denominator(self):
        sources = {"a.py": "def rare_name(): pass\ndef common_name(): pass",
                   "b.py": "def common_name(): pass", "c.py": "def other_name(): pass"}
        scores, _ = score_sources("rare_name common_name.", sources)
        rare, common = math.log(4/2), math.log(4/3)
        self.assertAlmostEqual(scores["a.py"], 1.)
        self.assertAlmostEqual(scores["b.py"], common/(rare+common))
        self.assertEqual(scores["c.py"], 0.)
        both = {"a.py": "def common_name(): pass", "b.py": "def common_name(): pass"}
        self.assertEqual(score_sources("common_name", both)[0], dict.fromkeys(both, 0.))

    def test_unicode_dollar_case_and_no_python_builtin_filter_on_java(self):
        sources = {"a/Thing.java": "class Thing { void list() {} void 读取数据() {} }",
                   "b/Other.js": "function $fetch() {}"}
        scores, _ = score_sources("list 读取数据 $fetch", sources)
        self.assertTrue(all(v > 0 for v in scores.values()))
        self.assertEqual(score_sources("LIST $Fetch", sources)[0], dict.fromkeys(sources, 0.))

    def test_unknown_language_errors_and_parse_recovery_is_visible(self):
        with self.assertRaises(ValueError):
            extract_symbols("unknown.xyz", "anything")
        parsed = extract_symbols("Bad.java", "class Bad { void broken( { ")
        self.assertTrue(parsed.parse_has_error)
        _, details = anchors("Bad", {"Bad.java": parsed})
        self.assertTrue(details["Bad.java"]["parse_has_error"])
        overridden = extract_symbols("api.h", "namespace demo { class Cache {}; }", language_overrides={".h": "cpp"})
        self.assertEqual(overridden.language, "cpp")

    def test_full_ab_entrypoint_computes_source_features_and_reranks(self):
        a = [{"file_path": p, "rank": i+1, "score": 2-i} for i, p in enumerate(("a/Foo.java", "b/Bar.java"))]
        sources = {"a/Foo.java": "class Foo {}", "b/Bar.java": "class Bar { void readValue() {} }"}
        result = rank_candidates("readValue", a, a, sources, anchor_weight=2.)
        self.assertEqual(result["ranking"][0]["file_path"], "b/Bar.java")
        self.assertEqual(result["anchor"]["b/Bar.java"], 1.)
        with self.assertRaises(ValueError):
            rank_candidates("readValue", a, a, {"b/Bar.java": sources["b/Bar.java"]})
        with self.assertRaises(ValueError):
            rank_candidates("readValue", a, a, sources, alpha=float("nan"))


if __name__ == "__main__":
    unittest.main()
