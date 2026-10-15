import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.chunking import PythonChunker, dynamic_ranges, fixed_ranges
from bug_localization.embeddings import LocalEmbedder
from bug_localization.vector import FUSIONS, rrf, validate_vector_config


class CharacterTokenizer:
    """A deliberately strict tokenizer for budget/Unicode/coverage tests."""
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text) + int(add_special_tokens)))

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {'input_ids': self.encode(text, add_special_tokens),
                'offset_mapping': [(i, i + 1) for i in range(len(text))]}


class ByteFallbackTokenizer(CharacterTokenizer):
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.encode('utf-8')) + int(add_special_tokens)))

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets = [(i, i + 1) for i, char in enumerate(text) for _ in char.encode('utf-8')]
        return {'input_ids': self.encode(text, add_special_tokens), 'offset_mapping': offsets}


CONFIG = {'prefix_max_tokens': 64, 'max_tokens': 512, 'dynamic_max_lines': 30,
          'dynamic_general_cost': 50, 'fixed_chunk_tokens': 400}
HAS_PARSER = importlib.util.find_spec('tree_sitter') is not None and importlib.util.find_spec('tree_sitter_python') is not None


class FusionTests(unittest.TestCase):
    def test_rrf_uses_file_ranks_and_stable_path_ties(self):
        result = rrf([[{'file_path': 'b.py'}, {'file_path': 'a.py'}],
                      [{'file_path': 'a.py'}, {'file_path': 'b.py'}]])
        self.assertEqual([item['file_path'] for item in result], ['a.py', 'b.py'])
        self.assertAlmostEqual(result[0]['score'], 1 / 61 + 1 / 62)

    def test_rrf_rejects_duplicate_files(self):
        with self.assertRaises(ValueError):
            rrf([[{'file_path': 'a.py'}, {'file_path': 'a.py'}]])

    def test_two_route_control_is_retained_and_third_route_has_equal_weight(self):
        self.assertEqual(FUSIONS['rrf'], ('dynamic', 'method'))
        self.assertEqual(FUSIONS['rrf_three'], ('dynamic', 'method', 'fixed'))
        rankings = [[{'file_path': p} for p in paths] for paths in
                    [('a.py', 'b.py'), ('a.py', 'b.py'), ('b.py', 'a.py')]]
        result = rrf(rankings, 60)
        self.assertEqual(result[0]['file_path'], 'a.py')
        self.assertAlmostEqual(result[0]['score'], 2 / 61 + 1 / 62)

    def test_dynamic_ranges_preserve_uploaded_algorithm(self):
        # Compile only the pure static method, without importing the supplied
        # module or its legacy llama-index dependencies.
        source = ast.parse((PROJECT / 'references/blaze/processor.py').read_text(encoding='utf-8'))
        class_node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'DynamicCodeSplitter')
        original = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == '_min_cost_split')
        original.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[]))
        namespace = {}
        exec(compile(module, '<original-dp>', 'exec'), namespace)
        for length in (1, 30, 31, 100):
            for points in ([], [(3, 3), (10, 5), (20, 3)]):
                expected, _ = namespace['_min_cost_split'](points, length, 30, 50)
                actual = dynamic_ranges(points, length, 30, 50)
                self.assertEqual(actual, [(start, end + 1) for start, end in expected])

    def test_invalid_prefix_budget_is_rejected(self):
        base = json.loads((PROJECT / 'configs/retrieval/bm25.json').read_text())
        config = json.loads((PROJECT / 'configs/retrieval/vector.json').read_text())
        config.update(max_tokens=128, prefix_max_tokens=120)
        with self.assertRaises(ValueError):
            validate_vector_config(config, base)

    def test_fixed_body_and_prefix_must_fit_total_budget(self):
        base = json.loads((PROJECT / 'configs/retrieval/bm25.json').read_text())
        config = json.loads((PROJECT / 'configs/retrieval/vector.json').read_text())
        config['fixed_chunk_tokens'] = 512
        with self.assertRaises(ValueError):
            validate_vector_config(config, base)

    def test_fixed_unicode_byte_tokens_never_split_a_character_or_drop_source(self):
        text = 'aa🙂bb中cc🙂'
        tokenizer = ByteFallbackTokenizer()
        ranges = fixed_ranges(text, tokenizer, 5)
        raw = text.encode('utf-8')
        pieces = [raw[r['start_byte']:r['end_byte']].decode('utf-8') for r in ranges]
        self.assertEqual(''.join(pieces), text)
        self.assertTrue(any(r['boundary_adjusted'] for r in ranges))
        self.assertTrue(all(len(tokenizer.encode(p, add_special_tokens=False)) <= 5 for p in pieces))
        with self.assertRaises(ValueError):
            fixed_ranges('🙂', tokenizer, 2)

    def test_fixed_substring_retokenization_cannot_exceed_body_budget(self):
        class ContextTokenizer(CharacterTokenizer):
            def encode(self, text, add_special_tokens=True):
                extra = int(bool(text) and not text.startswith('A'))
                return list(range(len(text) + extra + int(add_special_tokens)))

        text = 'Aabcdefghijklmnop'
        tokenizer = ContextTokenizer()
        ranges = fixed_ranges(text, tokenizer, 5)
        pieces = [text[r['start_byte']:r['end_byte']] for r in ranges]
        self.assertEqual(''.join(pieces), text)
        self.assertTrue(any(r['boundary_adjusted'] for r in ranges))
        self.assertTrue(all(len(tokenizer.encode(p, add_special_tokens=False)) <= 5 for p in pieces))

    def test_query_budget_includes_instruction_and_records_actual_input(self):
        embedder = LocalEmbedder.__new__(LocalEmbedder)
        embedder.config = {'max_tokens': 512, 'query_instruction': 'Find buggy source code.'}
        embedder.tokenizer = CharacterTokenizer()
        embedder.encode = lambda texts: [texts[0]]
        vector, metadata = embedder.query('界' * 900)
        self.assertTrue(metadata['truncated'])
        self.assertLessEqual(metadata['used_tokens'], 512)
        self.assertEqual(vector, metadata['embedding_text'])
        self.assertTrue(vector.startswith('Instruct: Find buggy source code.\nQuery: '))

    def test_short_query_is_not_modified_or_marked_truncated(self):
        embedder = LocalEmbedder.__new__(LocalEmbedder)
        embedder.config = {'max_tokens': 512, 'query_instruction': 'Find code.'}
        embedder.tokenizer = CharacterTokenizer()
        embedder.encode = lambda texts: [texts[0]]
        _, metadata = embedder.query('a short report')
        self.assertFalse(metadata['truncated'])
        self.assertTrue(metadata['embedding_text'].endswith('a short report'))


@unittest.skipUnless(HAS_PARSER, 'Semantic chunking requires tree-sitter and tree-sitter-python')
class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.chunker = PythonChunker(CharacterTokenizer(), CONFIG)

    def check_coverage(self, content, route):
        chunks, parents = self.chunker.chunks('pkg/module.py', content, route)
        self.assertEqual(''.join(chunk['code'] for chunk in chunks), content)
        self.assertEqual(chunks[0]['start_byte'], 0)
        self.assertEqual(chunks[-1]['end_byte'], len(content.encode()))
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous['end_byte'], current['start_byte'])
        for chunk in chunks:
            self.assertLessEqual(chunk['token_count'], 512)
            self.assertLessEqual(chunk['prefix_tokens'], 64)
        return chunks, parents

    def test_method_blocks_keep_decorators_nested_functions_and_residual_code(self):
        content = 'import os\nFLAG = 1\nclass C:\n    value = 2\n    @staticmethod\n    def f(x):\n        def inner():\n            return "你好"\n        return inner()\n\ndef g():\n    return FLAG\n'
        chunks, _ = self.check_coverage(content, 'method')
        self.assertEqual(sum(chunk['kind'] == 'method' for chunk in chunks), 2)
        function_chunk = next(chunk for chunk in chunks if '@staticmethod' in chunk['code'])
        self.assertIn('C', function_chunk['structure']['classes'])
        self.assertTrue(any('def f(x)' in s for s in function_chunk['structure']['signatures']))
        self.assertTrue(any(chunk['kind'] == 'outside_method' and 'import os' in chunk['code'] for chunk in chunks))

    def test_dynamic_blocks_and_oversized_unicode_statement_preserve_every_byte(self):
        content = 'class C:\n    def f(self):\n        value = "' + '数据' * 1000 + '"\n        return value\n'
        for route in ('dynamic', 'method'):
            chunks, _ = self.check_coverage(content, route)
            self.assertTrue(any(chunk['split_reason'] == 'oversized_statement_token_boundary' for chunk in chunks))

    def test_long_method_splits_at_statements_before_token_fallback(self):
        content = 'def f():\n' + ''.join('    value_' + str(i) + ' = ' + str(i) + '\n' for i in range(80))
        chunks, _ = self.check_coverage(content, 'method')
        self.assertTrue(any(chunk['split_reason'] == 'statement_boundary' for chunk in chunks))
        self.assertFalse(any(chunk['split_reason'] == 'oversized_statement_token_boundary' for chunk in chunks))

    def test_empty_and_invalid_source_are_retained(self):
        for content in ('', '# comment only\n', 'def broken(:\n   print "legacy"\n'):
            for route in ('dynamic', 'method', 'fixed'):
                self.check_coverage(content, route)

    def test_fixed_400_body_windows_are_nonoverlapping_and_keep_short_tail(self):
        chunks, _ = self.check_coverage('x' * 950, 'fixed')
        self.assertEqual([c['body_tokens'] for c in chunks], [400, 400, 150])
        self.assertEqual([c['start_byte'] for c in chunks], [0, 400, 800])
        self.assertTrue(all(c['split_reason'] is None for c in chunks))
        self.assertTrue(all(c['kind'] == 'fixed' for c in chunks))

    def test_long_prefix_keeps_field_labels_and_full_metadata(self):
        chunks, _ = self.chunker.chunks('long/' * 60 + 'module.py', 'class C:\n    def method(self):\n        pass\n', 'method')
        for chunk in chunks:
            self.assertIn('Class:', chunk['prefix'])
            self.assertIn('Signature:', chunk['prefix'])
            self.assertLessEqual(chunk['prefix_tokens'], 64)
            self.assertTrue(chunk['prefix_truncated'])
            self.assertIn('long/' * 60, chunk['structure']['full_prefix'])


if __name__ == '__main__':
    unittest.main()
