import importlib.util
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('source_prefetch', PROJECT / 'scripts/datasets/prefetch_embedding_sources.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
COMMIT = 'a' * 40


class SourcePrefetchTests(unittest.TestCase):
    def test_git_reuses_local_objects_and_exports_exact_commit_not_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            source = project / 'original'
            source.mkdir()

            def git(*args):
                return subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-c', 'user.name=Test',
                                       '-c', 'user.email=test@example.invalid', '-C', str(source), *args],
                                      check=True, capture_output=True, text=True).stdout.strip()

            git('init')
            (source / 'code.py').write_text('old content')
            (source / '.gitattributes').write_text('excluded.txt export-ignore\n')
            (source / 'excluded.txt').write_text('not exported')
            git('add', '.')
            git('commit', '-m', 'first')
            commit = git('rev-parse', 'HEAD')
            (source / 'code.py').write_text('new content')
            git('commit', '-am', 'second')
            mirror = project / 'repos/.embedding-source-git/owner/repo.git'
            mirror.parent.mkdir(parents=True)
            subprocess.run(['git', 'clone', '--bare', str(source), str(mirror)], check=True, capture_output=True)
            destination = project / 'snapshot.zip'
            with patch.object(module, 'PROJECT', project):
                provenance = module.git_archive('owner/repo', commit, destination)
            self.assertEqual(provenance[0]['fetch_attempts'], 0)
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(archive.read('repo-' + commit + '/code.py'), b'old content')
                self.assertNotIn('repo-' + commit + '/excluded.txt', archive.namelist())

    def archive(self, path, name=None):
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(name or ('repo-' + COMMIT + '/code.py'), 'print(1)')

    def test_archive_must_match_requested_commit_and_safe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.zip'
            self.archive(path)
            module.validate_archive(path, 'owner/repo', COMMIT)
            for name in ['repo-wrong/code.py', 'repo-' + COMMIT + '/../escape.py']:
                self.archive(path, name)
                with self.assertRaises(ValueError):
                    module.validate_archive(path, 'owner/repo', COMMIT)

    def test_interrupted_transfer_is_retried_and_only_valid_archive_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'snapshot.zip'
            calls = []

            def curl(command, **kwargs):
                partial = Path(command[command.index('--output') + 1])
                calls.append(command)
                self.assertFalse(destination.exists())
                if len(calls) == 1:
                    partial.write_bytes(b'incomplete')
                    return SimpleNamespace(returncode=18, stdout='200', stderr='partial transfer')
                self.archive(partial)
                return SimpleNamespace(returncode=0, stdout='200', stderr='')

            with patch.object(module.subprocess, 'run', side_effect=curl), patch.object(module.time, 'sleep'):
                attempts = module.download_archive('owner/repo', COMMIT, destination)
            self.assertEqual(len(attempts), 2)
            module.validate_archive(destination, 'owner/repo', COMMIT)
            self.assertFalse(destination.with_name('snapshot.curl.part').exists())

    def test_404_is_not_retried_or_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'snapshot.zip'
            with patch.object(module.subprocess, 'run', return_value=SimpleNamespace(returncode=22, stdout='404', stderr='missing')) as curl:
                with self.assertRaises(RuntimeError):
                    module.download_archive('owner/repo', COMMIT, destination)
                self.assertEqual(curl.call_count, 1)
                self.assertFalse(destination.exists())

    def test_bad_archive_cannot_be_published_on_http_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'snapshot.zip'

            def curl(command, **kwargs):
                Path(command[command.index('--output') + 1]).write_bytes(b'not a zip')
                return SimpleNamespace(returncode=0, stdout='200', stderr='')

            with patch.object(module.subprocess, 'run', side_effect=curl) as calls, patch.object(module.time, 'sleep'):
                with self.assertRaises(RuntimeError):
                    module.download_archive('owner/repo', COMMIT, destination)
                self.assertEqual(calls.call_count, 3)
                self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
