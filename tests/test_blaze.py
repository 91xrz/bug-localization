import ast
import json
import logging
from pathlib import Path
import re
import sys
import tempfile
import types
import typing
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization import blaze, blaze_experiment
from bug_localization.bm25 import ranking_metrics
from bug_localization.languages import parser_for_language


class BlazeTests(unittest.TestCase):
    def test_author_preprocessing_and_dp_parity(self):
        # Execute only the original pure classes, never its external imports.
        original = ast.parse((PROJECT / 'references/blaze/processor.py').read_text())
        nodes = [n for n in original.body if isinstance(n, (ast.Assign, ast.ClassDef, ast.FunctionDef))]
        # Original module-level type_cost.update is needed by the extractor.
        scope = {'Any': typing.Any, 'List': typing.List, 're': re, 'logger': logging.getLogger(__name__)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<author-reference>', 'exec'), scope)
        scope['type_cost'].update({name: 5 for name in scope['types_related_to_function']})
        cases = [
            ('sample.py', ''),
            ('sample.py', '# comment\n\nclass A:\n    def f(self):\n        return 1\n'),
            ('sample.py', 'text = "中文"\n# 中文注释\ndef f():\n    return text\n'),
            ('sample.py', 'def f():\n    x = lambda y: y\n' + '    value = 1\n' * 90),
            ('sample.py', 'class !!!\n' + 'invalid!' * 600),
            ('Sample.java', 'class Sample {\n // comment\n void f() { int x = 1; }\n}\n'),
            ('sample.cpp', 'class A {\n// comment\nint f() { return 1; }\n};\n'),
        ]
        stub = types.SimpleNamespace(get_parser=parser_for_language)
        with patch.dict(sys.modules, {'tree_sitter_languages': stub}):
            for path, code in cases:
                with self.subTest(path=path, code=code[:30]):
                    language = blaze.LANGUAGES[Path(path).suffix]
                    cleaner = scope['CommentCleaner'](language)
                    splitter = scope['DynamicCodeSplitter'](language, general_cost=50, max_window_size=30)
                    expected = splitter.split_text(cleaner.remove_comments_from_code(code))
                    self.assertEqual([c['text'] for c in blaze.split_file(path, code)], expected)

    def test_repeated_chunks_contribute_before_file_deduplication(self):
        dense = [{'file_path': p} for p in ['a.py', 'a.py', 'b.py']]
        lexical = [{'file_path': p} for p in ['b.py', 'a.py']]
        ranked = blaze.chunk_rrf(dense, lexical)
        self.assertEqual([x['file_path'] for x in ranked], ['a.py', 'b.py'])
        self.assertAlmostEqual(ranked[0]['score'], 1 / 61 + 1 / 62 + 1 / 62)
        self.assertEqual(ranked[0]['dense_chunk_ranks'], [1, 2])
        tied = blaze.chunk_rrf([{'file_path': 'z'}], [{'file_path': 'a'}])
        self.assertEqual([x['file_path'] for x in tied], ['z', 'a'])

    def test_exact_cosine_depth_and_file_control(self):
        chunks = [{'file_path': p, 'chunk_id': str(i), 'chunk_position': i}
                  for i, p in enumerate(['a', 'a', 'b', 'c'])]
        vectors = np.array([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype='float32')
        ranked = blaze.dense_rank(chunks, vectors, np.array([1, 0], dtype='float32'), 3)
        self.assertEqual([x['chunk_id'] for x in ranked], ['0', '1', '2'])
        files = blaze.unique_dense_files(ranked)
        self.assertEqual([x['file_path'] for x in files], ['a', 'b'])
        self.assertEqual(files[1]['first_chunk_rank'], 3)
        # Missing candidate labels stay in AP and recall denominators.
        self.assertEqual(ranking_metrics(['a'], ['a', 'absent'], 10)['average_precision_at_k'], .5)

    def test_cache_reuse_change_and_corruption(self):
        encoder = types.SimpleNamespace(identity={'model': 'test'}, device='cpu', batch_size=2,
                                        tokenizer=types.SimpleNamespace(encode=lambda text, **_: [1] if text else []),
                                        model=types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=2)))
        encoder.encode = MagicMock(side_effect=lambda texts, _: (
            np.tile(np.array([[1, 0]], dtype='float32'), (len(texts), 1)),
            [{'tokens_before_truncation': 4, 'truncated': False} for _ in texts]))
        docs = [{'file_path': 'a.py', 'sha256': 'first', 'content': 'x = 1'}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, first = blaze.cached_vectors(root, docs, encoder, 512)
            _, _, second = blaze.cached_vectors(root, docs, encoder, 512)
            self.assertFalse(first['cache_hit'])
            self.assertTrue(second['cache_hit'])
            encoder.encode.assert_called_once()
            changed = [{**docs[0], 'sha256': 'second', 'content': 'x = 2'}]
            _, _, third = blaze.cached_vectors(root, changed, encoder, 512)
            self.assertNotEqual(first['fingerprint'], third['fingerprint'])
            target = root / 'cache/blaze' / first['fingerprint'] / 'vectors.npy'
            target.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                blaze.cached_vectors(root, docs, encoder, 512)

    def test_empty_dense_chunks_are_audited_without_removing_files(self):
        # Match the released tokenizer's no-BOS behavior for the empty string.
        tokenizer = types.SimpleNamespace(encode=lambda text, **_: list(text.encode('utf-8')))
        encoder = types.SimpleNamespace(identity={'model': 'test'}, device='cpu', batch_size=2,
                                        tokenizer=tokenizer,
                                        model=types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=2)))
        encoder.encode = MagicMock(side_effect=lambda texts, _: (
            np.tile(np.array([[1, 0]], dtype='float32'), (len(texts), 1)),
            [{'tokens_before_truncation': len(text), 'truncated': False} for text in texts]))
        docs = [{'file_path': path, 'sha256': blaze.digest(content), 'content': content}
                for path, content in [('empty.py', ''), ('comment.py', '# only a comment\n'),
                                      ('mixed.py', '# comment\ndef f():\n    return 1\n')]]
        unchanged = json.dumps(docs, sort_keys=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunks, vectors, audit = blaze.cached_vectors(root, docs, encoder, 512)
            self.assertEqual([chunk['file_path'] for chunk in chunks], ['mixed.py'])
            self.assertEqual(vectors.shape, (1, 2))
            self.assertEqual(json.dumps(docs, sort_keys=True), unchanged)
            self.assertEqual({entry['file_path'] for entry in audit['excluded_empty_chunks']},
                             {'empty.py', 'comment.py'})
            self.assertTrue(all(entry['chunk_id'] for entry in audit['excluded_empty_chunks']))
            self.assertTrue(all(text for text in encoder.encode.call_args.args[0]))
            metadata = json.loads((root / 'cache/blaze' / audit['fingerprint'] / 'metadata.json').read_text())
            self.assertEqual(metadata['identity']['excluded_empty_chunks'], audit['excluded_empty_chunks'])
            self.assertIn('retain file candidates and gold', metadata['identity']['empty_chunk_policy'])
            _, _, cached = blaze.cached_vectors(root, docs, encoder, 512)
            self.assertTrue(cached['cache_hit'])
            self.assertEqual(cached['excluded_empty_chunks'], audit['excluded_empty_chunks'])
            encoder.encode.assert_called_once()
            with self.assertRaisesRegex(ValueError, 'excluding 2 zero-token chunks'):
                blaze.cached_vectors(root, docs[:2], encoder, 512)

    @unittest.skipUnless((PROJECT / 'models/BLAZE-release/tokenizer.json').is_file(),
                         'Pinned tokenizer has not been prepared')
    def test_released_tokenizer_empty_chunk_policy(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(PROJECT / 'models/BLAZE-release'),
                                                  local_files_only=True, trust_remote_code=False)
        self.assertEqual(tokenizer.encode('', add_special_tokens=True), [])
        encoder = types.SimpleNamespace(identity={'model': 'real-tokenizer-fixture'}, device='cpu', batch_size=2,
                                        tokenizer=tokenizer,
                                        model=types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=2)))
        encoder.encode = MagicMock(side_effect=lambda texts, _: (
            np.tile(np.array([[1, 0]], dtype='float32'), (len(texts), 1)),
            [{'tokens_before_truncation': len(tokenizer.encode(text)), 'truncated': False} for text in texts]))
        docs = [{'file_path': path, 'sha256': blaze.digest(content), 'content': content}
                for path, content in [('empty.py', ''), ('comment.py', '# only a comment\n'),
                                      ('mixed.py', '# comment\nvalue = 1\n')]]
        with tempfile.TemporaryDirectory() as temporary:
            chunks, _, audit = blaze.cached_vectors(Path(temporary), docs, encoder, 512)
        self.assertEqual([chunk['file_path'] for chunk in chunks], ['mixed.py'])
        self.assertEqual(len(audit['excluded_empty_chunks']), 2)
        self.assertTrue(all(tokenizer.encode(text) for text in encoder.encode.call_args.args[0]))

    def test_original_lexical_query_and_foreign_index_guard(self):
        client = MagicMock()
        client.indices.exists.return_value = False
        client.count.return_value = {'count': 1}
        client.search.return_value = {'timed_out': False, '_shards': {'failed': 0}, 'hits': {'hits': [
            {'_source': {'file_path': 'a.py'}, '_score': 3}]}}
        docs = [{'file_path': 'a.py', 'sha256': 'hash', 'content': 'code', 'repo': 'o/r', 'commit': 'a' * 40}]
        query = 'CamelCase_name ' * 1000
        ranked, audit = blaze.lexical_rank(client, docs, query, 1000)
        self.assertEqual(client.search.call_args.kwargs['query'], {'match': {'code': query}})
        self.assertEqual(client.indices.create.call_args.kwargs['mappings']['properties']['code']['analyzer'], 'standard')
        self.assertFalse(audit['query_truncated'])
        self.assertEqual(ranked[0]['file_path'], 'a.py')
        client.indices.exists.return_value = True
        client.indices.get_mapping.return_value = {audit['index']: {'mappings': {'_meta': {'owner': 'someone-else'}}}}
        with self.assertRaisesRegex(ValueError, 'another configuration'):
            blaze.lexical_rank(client, docs, query, 1000)
        client.indices.delete.assert_not_called()

    def test_config_rejects_solution_titles_and_unsupported_languages(self):
        config = json.loads((PROJECT / 'configs/retrieval/blaze.json').read_text())
        base = json.loads((PROJECT / config['base_config']).read_text())
        blaze_experiment.validate_config(config, base)
        with self.assertRaisesRegex(ValueError, 'solution PR'):
            blaze_experiment.validate_config(config, {**base, 'query_fields': ['title']})
        with self.assertRaisesRegex(ValueError, 'extension map'):
            blaze_experiment.validate_config(config, {**base, 'extensions': ['.kt']})
        with self.assertRaisesRegex(ValueError, 'rrf_constant'):
            blaze_experiment.validate_config({**config, 'rrf_constant': 61}, base)

    def test_failed_samples_remain_in_each_routes_denominator(self):
        config = json.loads((PROJECT / 'configs/retrieval/blaze.json').read_text())
        base = json.loads((PROJECT / config['base_config']).read_text())
        samples = [{'dataset_row': i, 'report': {'repo_name': 'o/r', 'issue_id': i, 'before_fix_sha': 'a' * 40}}
                   for i in range(2)]
        selection = {'selection_sha256': 'selection', 'dataset_metadata_sha256': 'dataset', 'samples': []}
        encoder = types.SimpleNamespace(identity={'variant': 'test'})
        result = {'routes': {r: {'metrics': ranking_metrics(['a'], ['a'], 10)} for r in blaze_experiment.ROUTES},
                  'missing_gold': [], 'indexed_files': 1, 'query_token_audit': {'truncated': False}}
        client = MagicMock()
        client.info.return_value = {'cluster_name': base['cluster_name'], 'version': {'number': '9.5.5'}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'base.json').write_text(json.dumps(base))
            config['base_config'] = 'base.json'
            with patch('elasticsearch.Elasticsearch') as es, \
                    patch.object(blaze_experiment, 'select_reports', return_value=(samples, selection)), \
                    patch.object(blaze_experiment, 'BlazeEncoder', return_value=encoder), \
                    patch.object(blaze_experiment, 'evaluate_case', side_effect=[result, ValueError('fixture failure')]):
                es.return_value.__enter__.return_value = client
                folder, summary = blaze_experiment.run_batch(root, config)
            for route, stats in summary['routes'].items():
                self.assertEqual(stats['failed_samples'], 1)
                self.assertEqual(stats['metrics']['MAP@10'], .5)
                self.assertEqual(stats['metrics_successful_only']['MAP@10'], 1)
                records = json.loads((folder / route / 'samples.json').read_text())
                self.assertEqual([r['status'] for r in records], ['ok', 'failed'])
                self.assertTrue((folder / route / records[0]['result_file']).is_file())

    def test_dense_failure_preserves_real_lexical_result_and_per_route_denominators(self):
        config = json.loads((PROJECT / 'configs/retrieval/blaze.json').read_text())
        base = json.loads((PROJECT / config['base_config']).read_text())
        row = {'repo_name': 'o/r', 'issue_id': 7, 'before_fix_sha': 'a' * 40,
               'body': 'A bug report', 'updated_files': "['a.py']"}
        docs = [{'file_path': 'a.py', 'sha256': 'source-hash', 'content': 'value = 1'}]
        selection = {'selection_sha256': 'selection', 'dataset_metadata_sha256': 'dataset', 'samples': []}
        encoder = types.SimpleNamespace(identity={'variant': 'test'}, encode=MagicMock())
        lexical = [{'rank': 1, 'file_path': 'a.py', 'score': 1.0}]
        client = MagicMock()
        client.info.return_value = {'cluster_name': base['cluster_name'], 'version': {'number': '9.5.5'}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'base.json').write_text(json.dumps(base))
            config['base_config'] = 'base.json'
            with patch('elasticsearch.Elasticsearch') as es, \
                    patch.object(blaze_experiment, 'select_reports', return_value=(
                        [{'dataset_row': 0, 'report': row}], selection)), \
                    patch.object(blaze_experiment, 'BlazeEncoder', return_value=encoder), \
                    patch.object(blaze_experiment, 'get_snapshot', return_value=(root, {'files': {}})), \
                    patch.object(blaze_experiment, 'collect_files', return_value=(docs, [])), \
                    patch.object(blaze_experiment, 'lexical_rank', return_value=(lexical, {'query_truncated': False})), \
                    patch.object(blaze_experiment, 'cached_vectors', side_effect=ValueError('dense fixture failure')):
                es.return_value.__enter__.return_value = client
                folder, summary = blaze_experiment.run_batch(root, config)
            self.assertEqual(summary['routes']['blaze_bm25']['successful_samples'], 1)
            self.assertEqual(summary['routes']['blaze_bm25']['metrics']['MAP@10'], 1)
            for route in ('blaze_dense', 'blaze_rrf'):
                self.assertEqual(summary['routes'][route]['failed_samples'], 1)
                self.assertEqual(summary['routes'][route]['metrics']['MAP@10'], 0)
                record = json.loads((folder / route / 'samples.json').read_text())[0]
                self.assertEqual(record['error_stage'], 'chunking_and_source_embeddings')
                errors = json.loads((folder / route / record['error_file']).read_text())
                self.assertIn('dense fixture failure', errors[route]['traceback'])
            result = json.loads((folder / 'cases/0001/result.json').read_text())
            self.assertEqual(result['routes']['blaze_bm25']['ranking'], lexical)
            self.assertEqual(result['route_errors']['blaze_rrf']['blocked_by'], 'blaze_dense')
            self.assertIsNone(result['query_token_audit'])
            encoder.encode.assert_not_called()

    def test_fusion_failure_preserves_both_completed_retrieval_routes(self):
        config = json.loads((PROJECT / 'configs/retrieval/blaze.json').read_text())
        base = json.loads((PROJECT / config['base_config']).read_text())
        row = {'repo_name': 'o/r', 'issue_id': 7, 'before_fix_sha': 'a' * 40,
               'body': 'A bug report', 'updated_files': "['a.py']"}
        docs = [{'file_path': 'a.py', 'sha256': 'source-hash', 'content': 'value = 1'}]
        chunks = [{'file_path': 'a.py', 'chunk_id': 'a', 'chunk_position': 0, 'text': 'value = 1'}]
        vectors = np.array([[1, 0]], dtype='float32')
        encoder = types.SimpleNamespace(encode=MagicMock(return_value=(vectors, [{'truncated': False}])))
        with patch.object(blaze_experiment, 'get_snapshot', return_value=(PROJECT, {'files': {}})), \
                patch.object(blaze_experiment, 'collect_files', return_value=(docs, [])), \
                patch.object(blaze_experiment, 'lexical_rank', return_value=([{'file_path': 'a.py'}], {})), \
                patch.object(blaze_experiment, 'cached_vectors', return_value=(chunks, vectors, {})), \
                patch.object(blaze_experiment, 'chunk_rrf', side_effect=ValueError('fusion fixture failure')):
            result = blaze_experiment.evaluate_case(PROJECT, base, config, row, encoder, MagicMock())
        self.assertEqual(set(result['routes']), {'blaze_bm25', 'blaze_dense'})
        self.assertEqual(result['routes']['blaze_dense']['metrics']['average_precision_at_k'], 1)
        self.assertEqual(set(result['route_errors']), {'blaze_rrf'})
        self.assertEqual(result['route_errors']['blaze_rrf']['stage'], 'fusion')
        self.assertIn('fusion fixture failure', result['route_errors']['blaze_rrf']['traceback'])


if __name__ == '__main__':
    unittest.main()
