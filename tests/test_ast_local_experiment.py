import json
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.ast_local_experiment import LOCAL_ROUTES, LocalChunker
from bug_localization.ast_chunking import ASTChunker
from test_vector import CharacterTokenizer


class LocalExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((PROJECT / 'configs/retrieval/vector.json').read_text())
        self.config['ast_max_chars'] = 180
        self.chunker = LocalChunker(CharacterTokenizer(), self.config)

    def test_routes_preserve_coverage_and_token_limits(self):
        text = ('def first(x):\n    if x > 0:\n        value = x + 1\n' +
                '        value = value + 1\n' * 25 + '        return value\n\n' +
                'def second():\n    return 2\n')
        for route in LOCAL_ROUTES:
            with self.subTest(route=route):
                chunks, _ = self.chunker.chunks('example.py', text, route)
                self.assertEqual(''.join(c['code'] for c in chunks), text)
                self.assertTrue(all(c['token_count'] <= 512 and c['route'] == route for c in chunks))
                if route != 'ast_scope':
                    self.assertTrue(any(c.get('local_context_sources') for c in chunks))
                for c in chunks:
                    for src in c.get('local_context_sources', []):
                        self.assertLess(src['start_byte'], c['start_byte'])
                        self.assertLessEqual(src['end_byte'], c['start_byte'])

    def test_scope_splits_distinct_small_functions(self):
        text = 'def first():\n    return 1\n\ndef second():\n    return 2\n'
        base, _ = ASTChunker(CharacterTokenizer(), self.config).chunks('x.py', text, 'blagent_ast')
        chunks, _ = self.chunker.chunks('x.py', text, 'ast_scope')
        self.assertEqual(len(base), 1)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(''.join(c['code'] for c in chunks), text)

    def test_sources_are_exact_preceding_text(self):
        text = 'def f(x):\n    if x:\n        a = x + 1\n' + '        a = a + 1\n' * 30 + '        return a\n'
        raw = text.encode()
        for route in ('ast_control', 'ast_variables', 'ast_local_both', 'ast_overlap'):
            chunks, _ = self.chunker.chunks('example.py', text, route)
            for chunk in chunks:
                for source in chunk.get('local_context_sources', []):
                    fragment = raw[source['start_byte']:source['end_byte']].decode().strip()
                    self.assertIn(source['kind'] + ': ' + fragment, chunk['embedding_text'])
                    self.assertLessEqual(source['end_byte'], chunk['start_byte'])

    def test_empty_and_other_language(self):
        for text, name in [('', 'x.py'), ('class A { int f() { return 1; } }', 'A.java')]:
            for route in LOCAL_ROUTES:
                chunks, _ = self.chunker.chunks(name, text, route)
                self.assertEqual(''.join(c['code'] for c in chunks), text)

if __name__ == '__main__':
    unittest.main()
