import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repair_sources', PROJECT / 'scripts/datasets/repair_source_archives.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SHA = 'a' * 40


class RepairArchiveTests(unittest.TestCase):
    def make(self, project, members=None, comment=SHA):
        base = project / 'repos/owner/repo' / SHA
        base.mkdir(parents=True)
        archive = base / 'snapshot.git.part'
        with zipfile.ZipFile(archive, 'w') as z:
            z.comment = comment.encode()
            for name, body in (members or [('code.py', b'print(123)'), ('test:fixture.py', b'pass')]):
                z.writestr('repo-' + SHA + '/' + name, body)
        return base, archive

    def test_verified_source_is_compatible_with_existing_snapshot_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            base, archive = self.make(project)
            result = module.materialize(project, 'owner/repo', SHA, archive, 10000, 10000, 0)
            self.assertEqual(result['files']['code.py'], hashlib.sha256(b'print(123)').hexdigest())
            self.assertEqual((base / 'source/test:fixture.py').read_bytes(), b'pass')
            _, reused = module.warmup.get_snapshot(project, 'owner/repo', SHA)
            self.assertEqual(result, reused)
            (base / 'source/code.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                module.materialize(project, 'owner/repo', SHA, archive, 10000, 10000, 0)

    def test_capacity_limits_refuse_before_source_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            base, archive = self.make(project)
            for compressed, expanded in [(1, 10000), (10000, 1)]:
                with self.assertRaisesRegex(ValueError, 'limit'):
                    module.materialize(project, 'owner/repo', SHA, archive, compressed, expanded, 0)
                self.assertFalse((base / 'source').exists())
                self.assertFalse((base / 'snapshot.json').exists())

    def test_traversal_and_wrong_revision_are_rejected(self):
        for members, comment in [([('../escape.py', b'bad')], SHA), (None, 'b' * 40)]:
            with tempfile.TemporaryDirectory() as tmp:
                project = Path(tmp)
                base, archive = self.make(project, members, comment)
                with self.assertRaises(ValueError):
                    module.materialize(project, 'owner/repo', SHA, archive, 10000, 10000, 0)
                self.assertFalse((base / 'snapshot.json').exists())

    def test_crc_failure_cannot_publish_partial_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            base, archive = self.make(project)
            archive.write_bytes(archive.read_bytes().replace(b'print(123)', b'print(456)'))
            with self.assertRaises(zipfile.BadZipFile):
                module.materialize(project, 'owner/repo', SHA, archive, 10000, 10000, 0)
            self.assertFalse((base / 'source').exists())
            self.assertFalse((base / 'snapshot.json').exists())
            self.assertEqual(list(base.glob('source.repair-*')), [])

    def test_existing_incomplete_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            base, archive = self.make(project)
            (base / 'source').mkdir()
            (base / 'source/keep.txt').write_text('keep')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                module.materialize(project, 'owner/repo', SHA, archive, 10000, 10000, 0)
            self.assertEqual((base / 'source/keep.txt').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
