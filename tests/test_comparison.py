import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bug_localization.chunk_cache import ChunkCache
from bug_localization.comparison import paired_records
from bug_localization.vector import RETRIEVAL_ROUTES, active_fusions, checkpoint


class ComparisonTests(unittest.TestCase):
    def test_fusion_disabled_has_no_rrf_outputs_or_pending_failures(self):
        self.assertEqual(active_fusions({'enable_fusion': False}), {})
        with self.assertRaises(ValueError):
            active_fusions({'enable_fusion': 'false'})
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            records = {route: [] for route in ('dynamic', 'method', 'fixed')}
            summary = checkpoint(folder, {}, records, 10, 1, SimpleNamespace(hits=0, misses=0))
            self.assertEqual(set(summary['routes']), set(records))
            self.assertFalse((folder / 'rrf').exists())
            self.assertFalse((folder / 'rrf_three').exists())

    def test_comparison_rejects_mismatched_sample_order_or_snapshot(self):
        row = {'position': 1, 'dataset_row': 3, 'repository': 'owner/repo', 'issue_id': 4, 'before_fix_sha': 'a' * 40}
        semantic = {route: [dict(row)] for route in RETRIEVAL_ROUTES}
        paired_records([row], semantic)
        for route in RETRIEVAL_ROUTES:
            semantic[route][0]['before_fix_sha'] = 'b' * 40
            with self.subTest(route=route), self.assertRaises(ValueError):
                paired_records([row], semantic)
            semantic[route][0]['before_fix_sha'] = row['before_fix_sha']


class ChunkCacheTests(unittest.TestCase):
    def make_chunker(self):
        class FakeChunker:
            config = {'max_tokens': 512, 'prefix_max_tokens': 64, 'dynamic_max_lines': 30,
                      'dynamic_general_cost': 50, 'fixed_chunk_tokens': 400}
            calls = 0

            def chunks(self, path, content, route):
                self.calls += 1
                return ([{'file_path': path, 'route': route, 'start_byte': 0, 'end_byte': len(content.encode()),
                          'code': content, 'token_count': 10, 'prefix_tokens': 3, 'body_tokens': 2}], [])
        return FakeChunker()

    def document(self, text):
        return {'file_path': 'pkg/a.py', 'content': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}

    def test_unchanged_file_reuses_chunks_and_changed_source_is_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            chunker = self.make_chunker()
            cache = ChunkCache(Path(directory), chunker, 'model-id')
            first = cache.get(self.document('你好\n'), 'fixed')
            self.assertEqual(cache.get(self.document('你好\n'), 'fixed'), first)
            self.assertEqual(chunker.calls, 1)
            self.assertEqual(cache.hits, 1)
            cache.get(self.document('changed\n'), 'fixed')
            self.assertEqual(chunker.calls, 2)

    def test_tampered_cache_fails_before_reusing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ChunkCache(Path(directory), self.make_chunker(), 'model-id')
            document = self.document('source\n')
            cache.get(document, 'fixed')
            path = next(cache.folder.glob('*.json'))
            stored = json.loads(path.read_text(encoding='utf-8'))
            stored['payload']['chunks'][0]['code'] = 'tampered\n'
            path.write_text(json.dumps(stored), encoding='utf-8')
            with self.assertRaises(ValueError):
                cache.get(document, 'fixed')


if __name__ == '__main__':
    unittest.main()
