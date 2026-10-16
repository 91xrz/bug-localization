import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.ast_chunking import ASTChunkCache, ASTChunker, RoutingChunkCache, ast_ranges
from bug_localization.chunking import PythonChunker
from bug_localization.languages import GRAMMARS, language_for_path, parser_for_language
from bug_localization.vector import active_routes, validate_vector_config
from test_vector import CharacterTokenizer

HAS_GRAMMARS = all(importlib.util.find_spec(module) is not None for module, _ in GRAMMARS.values())
FIXTURES = {path.name: path.read_text(encoding='utf-8')
            for path in (PROJECT / 'tests/fixtures/multilingual').iterdir() if path.is_file()}


def config():
    return json.loads((PROJECT / 'configs/retrieval/vector.json').read_text(encoding='utf-8'))


@unittest.skipUnless(HAS_GRAMMARS, 'Install the complete grammar requirements')
class ASTChunkingTests(unittest.TestCase):
    def assert_coverage(self, chunks, content):
        self.assertEqual(''.join(c['code'] for c in chunks), content)
        position = 0
        for chunk in chunks:
            self.assertEqual(chunk['start_byte'], position)
            self.assertEqual(chunk['end_byte'] - position, len(chunk['code'].encode('utf-8')))
            self.assertEqual(chunk['embedding_text'], chunk['prefix'] + '\n\n' + chunk['code'])
            self.assertLessEqual(chunk['prefix_tokens'], 64)
            self.assertLessEqual(chunk['token_count'], 512)
            position = chunk['end_byte']
        self.assertEqual(position, len(content.encode('utf-8')))

    def test_real_grammars_prefixes_and_all_four_routes_in_ten_syntaxes(self):
        self.assertEqual(len(FIXTURES), 10)
        chunker = ASTChunker(CharacterTokenizer(), config())
        for path, content in FIXTURES.items():
            for route in active_routes(config()):
                with self.subTest(path=path, route=route):
                    chunks, _ = chunker.chunks(path, content, route)
                    self.assert_coverage(chunks, content)
                    self.assertTrue(all(not c['parse_has_error'] for c in chunks))
                    self.assertTrue(any('Cache' in c['structure']['classes'] for c in chunks))
                    self.assertTrue(any(c['structure']['signatures'] for c in chunks))
                    self.assertTrue(all(c['language'] == language_for_path(path) for c in chunks))

    def test_ast_packing_keeps_fitting_function_nodes_intact(self):
        content = ''.join(f'def f{i}():\n    return {i}\n\n' for i in range(6))
        raw = content.encode()
        tree = parser_for_language('python').parse(raw)
        spans = ast_ranges(tree.root_node, raw, 55)
        self.assertGreater(len(spans), 1)
        for node in tree.root_node.named_children:
            self.assertTrue(any(start <= node.start_byte and end >= node.end_byte for start, end, _ in spans))
        self.assertEqual(b''.join(raw[start:end] for start, end, _ in spans), raw)

    def test_oversized_unicode_leaf_comments_whitespace_and_empty_files_are_preserved(self):
        chunker = ASTChunker(CharacterTokenizer(), {**config(), 'ast_max_chars': 1000})
        for content in ('', '\r\n  \t\r\n', '# ' + '你好🙂' * 1000 + '\r\n', 'value = "' + '你好🙂' * 2000 + '"\n'):
            with self.subTest(length=len(content)):
                chunks, parents = chunker.chunks('demo.py', content, 'blagent_ast')
                self.assert_coverage(chunks, content)
                self.assertTrue(parents)
                if len(content) > 1000:
                    self.assertGreater(len(chunks), 1)
                    self.assertTrue(any('oversized' in p['kind'] for p in parents))

    def test_parse_errors_are_recorded_without_dropping_a_file(self):
        content = 'class Broken { public void get( { 你好🙂\n'
        chunks, _ = ASTChunker(CharacterTokenizer(), config()).chunks('Demo.java', content, 'blagent_ast')
        self.assert_coverage(chunks, content)
        self.assertTrue(all(c['parse_has_error'] and c['parse_status'] == 'recovered' for c in chunks))

    def test_python_baseline_chunks_and_index_identity_are_preserved(self):
        content = FIXTURES['demo.py'] * 5
        document = {'file_path': 'demo.py', 'content': content, 'sha256': hashlib.sha256(content.encode()).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            router = RoutingChunkCache(Path(directory), CharacterTokenizer(), config(), 'model')
            original = PythonChunker(CharacterTokenizer(), config())
            for route in ('dynamic', 'method', 'fixed'):
                self.assertEqual(router.get(document, route), original.chunks('demo.py', content, route))
                self.assertEqual(set(router.index_identity([document], route)), {'chunker_sha256'})
            chunks, _ = router.get(document, 'blagent_ast')
            self.assert_coverage(chunks, content)
            self.assertIn('ast_pipeline_fingerprint', router.index_identity([document], 'blagent_ast'))
            self.assertTrue(all(c['route'] == 'blagent_ast' for c in chunks))

    def test_ast_parameters_and_language_overrides_separate_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folders = [ASTChunkCache(root, ASTChunker(CharacterTokenizer(), cfg), 'model').folder
                       for cfg in (config(), {**config(), 'ast_max_chars': 900},
                                   {**config(), 'language_overrides': {'.h': 'cpp'}})]
            self.assertEqual(len(set(folders)), 3)
            content = FIXTURES['Demo.java']
            document = {'file_path': 'Demo.java', 'content': content, 'sha256': hashlib.sha256(content.encode()).hexdigest()}
            cache = ASTChunkCache(root, ASTChunker(CharacterTokenizer(), config()), 'model')
            first = cache.get(document, 'blagent_ast')
            self.assertEqual(first, cache.get(document, 'blagent_ast'))
            self.assertEqual(cache.hits, 1)
            path = next(cache.folder.glob('*.json'))
            stored = json.loads(path.read_text(encoding='utf-8'))
            stored['payload']['chunks'][0]['code'] = 'tampered'
            path.write_text(json.dumps(stored), encoding='utf-8')
            with self.assertRaises(ValueError):
                cache.get(document, 'blagent_ast')


class LanguageConfigurationTests(unittest.TestCase):
    def test_supported_extensions_and_explicit_header_override(self):
        self.assertEqual(language_for_path('x.h'), 'c')
        self.assertEqual(language_for_path('x.h', {'.h': 'cpp'}), 'cpp')
        self.assertEqual(language_for_path('DEMO.JAVA'), 'java')
        with self.assertRaises(ValueError):
            language_for_path('demo.unknown')
        base = json.loads((PROJECT / 'configs/retrieval/bm25.json').read_text())
        validate_vector_config(config(), {**base, 'extensions': ['.py', '.java', '.cpp', '.ts']})
        for changes in ({'ast_max_chars': 0}, {'chunking_routes': ['blagent_ast', 'blagent_ast']},
                        {'language_overrides': {'.h': 'unknown'}}):
            with self.assertRaises(ValueError):
                validate_vector_config({**config(), **changes}, base)
        with self.assertRaises(ValueError):
            validate_vector_config(config(), {**base, 'extensions': ['.unknown']})


if __name__ == '__main__':
    unittest.main()
