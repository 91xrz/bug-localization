"""Snapshot sharing, failure isolation and prepared-query integrity."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
spec = importlib.util.spec_from_file_location('recall_paired', ROOT / 'scripts/experiments/run_recall_paired.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PairedRecallTest(unittest.TestCase):
    def test_group_requires_same_repository_and_commit(self):
        rows = [{'repository': 'a', 'before_fix_sha': '1'}, {'repository': 'b', 'before_fix_sha': '1'},
                {'repository': 'a', 'before_fix_sha': '2'}, {'repository': 'a', 'before_fix_sha': '1'}]
        self.assertEqual(m.grouped(rows), [[rows[0], rows[3]], [rows[1]], [rows[2]]])

    def test_two_queries_load_once_share_index_and_cleanup_after_both(self):
        self.snapshot_case(failed=False)

    def test_generation_failure_does_not_suppress_original_query(self):
        self.snapshot_case(failed=True)

    def snapshot_case(self, failed):
        meta = {'position': 1, 'dataset_row': 0, 'repository': 'x/y', 'before_fix_sha': 'a'*40, 'issue_id': 1}
        docs = [{'file_path': 'a.py', 'sha256': 'filehash'}]
        chunks = [{'file_path': 'a.py', 'chunk_id': 'chunk1', 'embedding_text': 'code'}]
        vector = np.array([1., 0.], dtype='<f4')
        cache = Mock(identity={'code': 'identity'}, fingerprint='model', tokenizer=Mock())
        cache.encode.return_value = vector[None, :]
        cache.query.return_value = (vector, {'embedding_text': 'original'})
        prepared = {1: {'status': 'failed', 'error': 'input limit'} if failed else
                    {'status': 'ok', 'query': 'hypothesis', 'vector': vector.tolist(),
                     'query_info': {}, 'embedding_identity': {'single': 'identity'}}}
        client = Mock()
        state = {'index': None}
        events = []
        client.indices.exists.side_effect = lambda **kw: state['index'] is not None

        def retrieve(client, documents, actual_chunks, vectors, query, identity, *args):
            reused = state['index'] is not None
            fingerprint = m.digest({'identity': identity, 'chunks': ['chunk1'],
                'vectors_sha256': hashlib.sha256(vectors.astype('<f4').tobytes()).hexdigest()})
            index = 'bl-vector-blagent_ast-' + 'a'*12 + '-' + fingerprint[:16]
            state['index'] = index
            client.indices.get_mapping.return_value = {index: {'mappings': {'_meta': {
                'fingerprint': fingerprint, 'full_experiment': identity['full_experiment']}}}}
            events.append('query')
            return {'index': index, 'index_reused': reused, 'ranking': [{'file_path': 'a.py'}]}

        client.indices.delete.side_effect = lambda **kw: events.append('delete')
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(m.old, 'get_snapshot', return_value=(Path(temp), {})) as source, \
             patch.object(m.old, 'collect_files', return_value=(docs, [])) as collect, \
             patch.object(m.old, 'inventory_reference', return_value=docs), \
             patch.object(m.old, 'prepare_report', return_value=({}, ['a.py'], 'original')), \
             patch.object(m.old, 'retrieve', side_effect=retrieve) as score, \
             patch('bug_localization.parallel_chunking.chunk_documents', return_value=[(chunks,)]) as chunk, \
             patch('bug_localization.ast_chunking.RoutingChunkCache') as chunk_cache:
            chunk_cache.return_value.index_identity.return_value = {}
            records = {'S0': {}, 'H0': {}}
            timing = m.execute_snapshot(Path(temp), {'base': {}}, {'aggregation_temperature': .1},
                [{'report': {}}], [meta], prepared, records, cache, Mock(), client, 1)
            source.assert_called_once()
            collect.assert_called_once()
            chunk.assert_called_once()
            cache.encode.assert_called_once()
            self.assertEqual(score.call_count, 1 if failed else 2)
            self.assertEqual(events, ['query', 'delete'] if failed else ['query', 'query', 'delete'])
            self.assertEqual(records['S0'][1]['status'], 'ok')
            self.assertEqual(records['H0'][1]['status'], 'failed' if failed else 'ok')
            if not failed:
                self.assertEqual([q['index_reused'] for q in timing['queries']], [False, True])

    def test_changed_generation_or_vector_rejected(self):
        meta = {'position': 1}
        vector = np.zeros(1024, dtype='<f4'); vector[0] = 1
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            m.old.save(root / 'generation/0001.json', {'query': 'hypothesis'})
            m.old.save(root / 'paired_protocol.json', {'hypothesis_query_embedding_identity': {'gpu_count': 1}})
            entry = {'sample': meta, 'query': 'hypothesis', 'generation_sha256': m.old.sha256_file(root/'generation/0001.json')}
            m.save_query(root, entry, vector, {}, {'gpu_count': 1}, True)
            self.assertEqual(m.load_prepared(root, meta)['status'], 'ok')
            value = m.old.read(m.query_path(root, meta));value['vector'][0] = .5
            m.old.save(m.query_path(root, meta), value)
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                m.load_prepared(root, meta)
            m.old.save(root / 'generation/0001.json', {'query': 'changed'})
            with self.assertRaisesRegex(ValueError, 'Generation changed'):
                m.load_prepared(root, meta)


if __name__ == '__main__':
    unittest.main()
