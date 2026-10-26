import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.ast_chunking import RoutingChunkCache
from bug_localization.token_ast import EXPERIMENTAL_ROUTES, ExperimentalChunkCache, TokenASTChunker, choose_partition
from bug_localization.vector import active_routes
from test_vector import CharacterTokenizer


def config():
    return json.loads((PROJECT / 'configs/retrieval/vector.json').read_text(encoding='utf-8'))


class TokenASTTests(unittest.TestCase):
    def test_dp_prefers_shallow_boundary_over_greedy_and_has_lower_cost(self):
        atoms = [(0, 1, 0), (1, 2, 8), (2, 3, 0)]
        def measure(a, b):
            return {'tokens': sum([6, 4, 4][a:b]), 'structure': {'signatures': []}}
        greedy = choose_partition(atoms, measure, 10, 'ast_token_greedy')
        dp = choose_partition(atoms, measure, 10, 'ast_token_dp')
        self.assertEqual([(a, b) for a, b, _ in greedy], [(0, 2), (2, 3)])
        self.assertEqual([(a, b) for a, b, _ in dp], [(0, 1), (1, 3)])
        self.assertLess(sum(v[2] for v in dp), sum(v[2] for v in greedy))

    def test_nonmonotone_token_counts_do_not_hide_later_fitting_edge(self):
        atoms = [(i, i + 1, 0) for i in range(3)]
        def measure(a, b):
            return {'tokens': 11 if b - a == 2 else 5, 'structure': {'signatures': []}}
        for route in EXPERIMENTAL_ROUTES:
            self.assertEqual([(a, b) for a, b, _ in choose_partition(atoms, measure, 10, route)], [(0, 3)])

    def test_multilingual_unicode_empty_and_recovered_source_preserved(self):
        fixture = PROJECT / 'tests/fixtures/multilingual'
        cases = [(p.name, p.read_text(encoding='utf-8') * 3) for p in fixture.iterdir() if p.is_file()]
        cases += [('x.py', ''), ('x.py', '\r\n \t\n'), ('x.py', '# ' + '你好🙂' * 400 + '\n'),
                  ('x.java', 'class Broken { void broken( { 你好\n')]
        chunker = TokenASTChunker(CharacterTokenizer(), config())
        for path, text in cases:
            costs = []
            for route in EXPERIMENTAL_ROUTES:
                with self.subTest(path=path, route=route, length=len(text)):
                    chunks, parents = chunker.chunks(path, text, route)
                    self.assertEqual(''.join(c['code'] for c in chunks), text)
                    position = 0
                    for c in chunks:
                        self.assertEqual(c['start_byte'], position)
                        position = c['end_byte']
                        self.assertEqual(c['embedding_text'], c['prefix'] + '\n\n' + c['code'])
                        self.assertLessEqual(c['token_count'], 512)
                        self.assertLessEqual(c['prefix_tokens'], 64)
                    self.assertEqual(position, len(text.encode('utf-8')))
                    costs.append(sum(p['packing_cost'] for p in parents))
            self.assertLessEqual(costs[1], costs[0] + 1e-8)

    def test_cache_baselines_and_implicit_defaults_remain_unchanged(self):
        cfg = config()
        cfg.pop('chunking_routes')
        self.assertEqual(active_routes(cfg), ('dynamic', 'method', 'fixed', 'blagent_ast'))
        text = 'def f(x):\n    return x + 1\n'
        doc = {'file_path': 'x.py', 'content': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            old = RoutingChunkCache(p, CharacterTokenizer(), cfg, 'model')
            new = ExperimentalChunkCache(p, CharacterTokenizer(), cfg, 'model')
            for route in active_routes(cfg):
                self.assertEqual(old.get(doc, route), new.get(doc, route))
                self.assertEqual(old.index_identity([doc], route), new.index_identity([doc], route))
            for route in EXPERIMENTAL_ROUTES:
                first = new.get(doc, route)
                self.assertEqual(first, new.get(doc, route))
                self.assertNotEqual(old.index_identity([doc], 'blagent_ast'), new.index_identity([doc], route))


if __name__ == '__main__':
    unittest.main()
