import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch
from urllib.error import HTTPError

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.model_files import (required_model_files, encoder_implementation_sha256,
                                         LEGACY_ENCODER_SHA256, SHARDED_LOADER_ENCODER_SHA256)
from bug_localization.snapshot import sha256_file

spec = importlib.util.spec_from_file_location('warmup', PROJECT / 'scripts/retrieval/warm_embedding_cache.py')
warmup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warmup)


class ModelFilesTests(unittest.TestCase):
    def root(self, folder):
        root = Path(folder)
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
            (root / name).write_text('{}')
        return root

    def index(self, root, mapping):
        (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))

    def test_single_file_preserves_existing_identity_file_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.root(tmp)
            (root / 'model.safetensors').write_bytes(b'weights')
            self.assertEqual(required_model_files(root), ['model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'])

    def test_all_shards_and_index_are_fingerprinted_and_missing_shard_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.root(tmp)
            self.index(root, {'layer.a': 'one.safetensors', 'layer.b': 'two.safetensors', 'layer.c': 'one.safetensors'})
            (root / 'one.safetensors').write_bytes(b'a')
            with self.assertRaises(FileNotFoundError):
                required_model_files(root)
            (root / 'two.safetensors').write_bytes(b'b')
            files = required_model_files(root)
            self.assertEqual(files[:3], ['model.safetensors.index.json', 'one.safetensors', 'two.safetensors'])
            before = {n: sha256_file(root / n) for n in files}
            (root / 'two.safetensors').write_bytes(b'changed')
            self.assertNotEqual(before, {n: sha256_file(root / n) for n in files})

    def test_invalid_indices_and_escaping_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.root(tmp)
            for mapping in ({}, [], {'a': '../bad.safetensors'}, {'a': 'a\\b.safetensors'}, {'a': 3}):
                with self.subTest(mapping=mapping):
                    self.index(root, mapping)
                    with self.assertRaises(ValueError):
                        required_model_files(root)

    def test_compatibility_is_limited_to_exact_reviewed_encoder_source(self):
        current = PROJECT / 'src/bug_localization/embeddings.py'
        self.assertEqual(sha256_file(current), SHARDED_LOADER_ENCODER_SHA256)
        self.assertEqual(encoder_implementation_sha256(current), LEGACY_ENCODER_SHA256)
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / 'embeddings.py'
            changed.write_text(current.read_text().replace('hidden[:, -1]', 'hidden[:, 0]'))
            self.assertNotEqual(encoder_implementation_sha256(changed), LEGACY_ENCODER_SHA256)
            self.assertEqual(encoder_implementation_sha256(changed), sha256_file(changed))


class WarmupTests(unittest.TestCase):
    def test_unrelated_experiment_additions_do_not_invalidate_cache_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            for name in warmup.IMPLEMENTATION_FILES:
                path = project / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('original')
            frozen = warmup.implementation_identity(project)
            unrelated = project / 'src/bug_localization/blaze.py'
            unrelated.write_text('new unrelated baseline')
            self.assertEqual(warmup.implementation_identity(project), frozen)
            unrelated.write_text('another baseline change')
            with patch.object(warmup, 'PROJECT', project):
                warmup.validate_implementation(frozen)
            encoder = project / 'src/bug_localization/embeddings.py'
            encoder.write_text('changed pooling logic')
            with patch.object(warmup, 'PROJECT', project):
                with self.assertRaisesRegex(ValueError, 'embeddings.py'):
                    warmup.validate_implementation(frozen)
            encoder.write_text('original')
            (project / 'src/bug_localization/chunking.py').unlink()
            with self.assertRaises(FileNotFoundError):
                warmup.implementation_identity(project)

    def test_report_fingerprint_accepts_arrow_timestamp_metadata(self):
        row = {'repo_name': 'a/b', 'issue_id': 1, 'before_fix_sha': 'a' * 40,
               'body': 'report', 'title': None, 'updated_files': "['a.py']", 'created_at': datetime.now()}
        original = warmup.digest(warmup.report_identity(row))
        row['body'] = 'changed report'
        self.assertNotEqual(warmup.digest(warmup.report_identity(row)), original)

    def test_network_retries_only_transient_errors(self):
        transient = HTTPError('url', 503, 'unavailable', {}, None)
        with patch.object(warmup, 'get_snapshot', side_effect=[transient, ('source', {})]) as get, patch.object(warmup.time, 'sleep'):
            self.assertEqual(warmup.fetch_source({}, 'a/b', 'c'), ('source', {}))
            self.assertEqual(get.call_count, 2)
        with patch.object(warmup, 'get_snapshot', side_effect=HTTPError('url', 404, 'missing', {}, None)) as get:
            with self.assertRaises(HTTPError):
                warmup.fetch_source({}, 'a/b', 'c')
            self.assertEqual(get.call_count, 1)

    def test_snapshot_dedup_preserves_every_report_and_order(self):
        rows = [{'repo_name': 'a/b', 'before_fix_sha': '1'}, {'repo_name': 'a/c', 'before_fix_sha': '1'},
                {'repo_name': 'a/b', 'before_fix_sha': '1'}]
        self.assertEqual(list(warmup.grouped_rows(rows).items()), [(('a/b', '1'), [0, 2]), (('a/c', '1'), [1])])

    def test_checkpoint_resume_and_explicit_failure_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.sqlite3'
            progress = warmup.Progress(path)
            self.assertTrue(progress.should_run('snapshot', 'a', False))
            progress.put('snapshot', 'a', 'complete', {'reports': 2})
            progress.put('snapshot', 'b', 'failed', {'error': 'HTTP 404'})
            progress.db.close()
            resumed = warmup.Progress(path)
            self.assertFalse(resumed.should_run('snapshot', 'a', True))
            self.assertFalse(resumed.should_run('snapshot', 'b', False))
            self.assertTrue(resumed.should_run('snapshot', 'b', True))
            self.assertEqual(resumed.counts(), {'snapshot_complete': 1, 'snapshot_failed': 1})
            self.assertEqual(resumed.failures()[0]['error'], 'HTTP 404')
            resumed.db.close()

    def test_frozen_settings_refuse_silent_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pinned.json'
            warmup.pin(path, {'batch': 64})
            warmup.pin(path, {'batch': 64})
            with self.assertRaises(ValueError):
                warmup.pin(path, {'batch': 32})

    def test_frozen_profiles_validate_and_exclude_unrequested_datasets(self):
        plan = json.loads((PROJECT / 'configs/experiments/embedding_cache_20261113.json').read_text())
        self.assertEqual(len(plan['jobs']), 8)
        for job in plan['jobs']:
            config, base = warmup.load_job(PROJECT / job['config'])
            self.assertIn(base['dataset'], ('swe_bench_formatted', 'bench4bl_formatted', 'BeetleBox'))
            self.assertEqual(config['gpu_count'], 2)
            if base['dataset'] == 'swe_bench_formatted':
                self.assertEqual(base['query_fields'], ['body'])
            if config['model_path'].endswith('0.6B'):
                self.assertEqual(config['embedding_batch_size'], 64)


if __name__ == '__main__':
    unittest.main()
