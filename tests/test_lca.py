"""LCA stays offline, uses base_sha rather than HEAD, and preserves evaluation labels."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.lca import extract_archive, normalize_row, prepare, git
from bug_localization.snapshot import get_snapshot, sha256_file
from bug_localization.bm25 import validate_config
from bug_localization.languages import language_for_path, parser_for_language


class LCATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.repo = self.project / 'fixture'
        self.repo.mkdir()
        def run(*args):
            return subprocess.run(['git', '-C', str(self.repo), *args], check=True, capture_output=True).stdout
        self.command = run
        run('init', '-q'); run('config', 'user.name', 'LCA Test'); run('config', 'user.email', 'test@example.invalid')
        (self.repo / 'Example.java').write_text('class Example { int value = 1; }\n')
        run('add', '.'); run('commit', '-qm', 'base')
        self.base = run('rev-parse', 'HEAD').decode().strip()
        (self.repo / 'Example.java').write_text('class Example { int value = 999; }\n')
        run('commit', '-qam', 'later HEAD')
        self.head = run('rev-parse', 'HEAD')
        self.row = {'id': 42, 'repo_owner': 'owner', 'repo_name': 'sample', 'base_sha': self.base,
                    'changed_files': "['Example.java', 'Added.java']", 'issue_title': 'original title',
                    'issue_body': 'original report', 'issue_url': 'https://example.invalid/issues/7',
                    'pull_url': 'https://example.invalid/pull/8', 'text_id': 'owner/sample/8/7',
                    'diff': 'MUST NOT APPEAR IN QUERY'}
        self.raw = self.project / 'datasets/LCA_bug_localization'
        (self.raw / 'repos/java').mkdir(parents=True)
        self.archive = self.raw / 'repos/java/owner__sample.zip'
        with zipfile.ZipFile(self.archive, 'w') as z:
            for p in self.repo.rglob('*'):
                if p.is_file():z.write(p, p.relative_to(self.repo))
        (self.raw / 'metadata').mkdir()
        (self.raw / 'metadata/download-status.json').write_text(json.dumps({'status': 'complete', 'ready': {
            'repos/java/owner__sample.zip': {'sha256': sha256_file(self.archive)}}}))
        (self.raw / 'java').mkdir()
        self.write_rows()

    def write_rows(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist([self.row]), self.raw / 'java/test-00000-of-00001.parquet')

    def test_offline_base_commit_gold_and_cache_integrity(self):
        from datasets import load_from_disk
        with patch('urllib.request.urlopen', side_effect=AssertionError('must remain offline')):
            result = prepare(self.project, self.raw, ['java'])
            self.assertEqual(result['status'], 'complete')
            ds = load_from_disk(str(self.project / 'datasets/lca_java_formatted'))
            self.assertEqual(set(ds), {'test'})
            row = ds['test'][0]
            self.assertEqual(row['issue_id'], 42)
            self.assertEqual(ast.literal_eval(row['updated_files']), ['Example.java', 'Added.java'])
            self.assertEqual(row['body'], 'original report')
            source, manifest = get_snapshot(self.project, 'owner/sample', self.base,
                    source_provider='lca_git', source_dataset='lca_java_formatted')
            self.assertIn('value = 1;', (source / 'Example.java').read_text())
            self.assertNotIn('Added.java', manifest['files'])
            self.assertTrue(manifest['source_url'].startswith('https://huggingface.co/'))
            self.assertEqual(git(self.repo, 'rev-parse', 'HEAD'), self.head)
            (source / 'Example.java').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'contents changed'):
                get_snapshot(self.project, 'owner/sample', self.base,
                             source_provider='lca_git', source_dataset='lca_java_formatted')

    def test_missing_commit_cannot_fall_back_to_head(self):
        self.row['base_sha'] = 'f' * 40; self.write_rows()
        with self.assertRaisesRegex(ValueError, 'Missing/non-commit'):
            prepare(self.project, self.raw, ['java'])
        self.assertFalse((self.project / 'datasets/lca_java_formatted').exists())

    def test_zip_path_traversal_rejected(self):
        p = self.project / 'bad.zip'
        with zipfile.ZipFile(p, 'w') as z:z.writestr('../escape', 'bad')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            extract_archive(p, self.project / 'extracted', sha256_file(p))
        self.assertFalse((self.project / 'escape').exists())

    def test_labels_are_data_not_code(self):
        with self.assertRaises((ValueError, SyntaxError)):
            normalize_row({**self.row, 'changed_files': "__import__('os').system('false')"}, 'java')

    def test_lca_requires_explicit_local_provider(self):
        config = json.loads((PROJECT / 'configs/retrieval/lca/java/bm25.json').read_text())
        validate_config(config)
        config.pop('source_provider')
        with self.assertRaisesRegex(ValueError, 'offline'):
            validate_config(config)

    def test_kotlin_grammar(self):
        self.assertEqual(language_for_path('src/Test.kt'), 'kotlin')
        tree = parser_for_language('kotlin').parse(b'fun add(a: Int, b: Int): Int {\n return a + b\n}\n')
        self.assertFalse(tree.root_node.has_error)
        self.assertEqual(tree.root_node.named_children[0].type, 'function_declaration')


if __name__ == '__main__':
    unittest.main()
