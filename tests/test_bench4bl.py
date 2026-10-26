"""Release pinning, offline source integrity, literal parsing and gold preservation."""

import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.bench4bl import literal, map_labels, prepare, release_commit, git
from bug_localization.snapshot import get_snapshot


class LabelTests(unittest.TestCase):
    def test_unique_qualified_name_and_exact_path(self):
        entries = [{'path': 'src/main/java/org/demo/A.java'}, {'path': 'src/test/java/org/demo/ATest.java'}]
        gold, audit = map_labels(['org.demo.A.java', 'src/test/java/org/demo/ATest.java'], entries)
        self.assertEqual(gold, [e['path'] for e in entries])
        self.assertTrue(all(m['status'] == 'resolved' for m in audit))

    def test_ambiguous_and_absent_labels_preserve_denominator(self):
        entries = [{'path': 'one/org/demo/A.java'}, {'path': 'two/org/demo/A.java'}]
        gold, audit = map_labels(['org.demo.A.java', 'org.demo.Missing.java'], entries)
        self.assertEqual([a['status'] for a in audit], ['ambiguous', 'not_in_release'])
        self.assertEqual(len(gold), 2)
        self.assertTrue(all(p.startswith('__bench4bl_unresolved_gold__/') for p in gold))

    def test_metadata_is_not_executed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'versions.txt'
            path.write_text("__import__('os').system('must_not_execute')", encoding='utf-8')
            with self.assertRaises(ValueError):
                literal(path, 'CSV')


@unittest.skipUnless(shutil.which('git'), 'Local Git is required for source-adapter tests')
class Bench4BLTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.raw = self.project / 'datasets/Bench4BL'
        for group in ('Apache', 'Commons', 'JBoss', 'Spring', 'Wildfly'):
            (self.raw / 'data' / group).mkdir(parents=True)
        self.subject = self.raw / 'data/Commons/CSV'
        self.repo = self.subject / 'gitrepo'
        self.repo.mkdir(parents=True)
        def command(*args):
            return subprocess.run(['git', '-C', str(self.repo), *args], check=True, capture_output=True)
        command('init', '-q')
        command('config', 'user.name', 'Adapter Test')
        command('config', 'user.email', 'adapter-test@example.invalid')
        self.path = self.repo / 'src/main/java/org/demo/A.java'
        self.path.parent.mkdir(parents=True)
        self.path.write_text('package org.demo; class A { int value() { return 0; } }\n', encoding='utf-8')
        command('add', '.')
        command('commit', '-qm', 'Release fixture')
        command('tag', 'v1.0')
        self.release = release_commit(self.repo, 'v1.0')
        self.path.write_text('package org.demo; class A { int value() { return 99; } }\n', encoding='utf-8')
        command('commit', '-qam', 'Later HEAD must not be used')
        self.head = git(self.repo, 'rev-parse', 'HEAD').decode().strip()
        (self.subject / 'bugrepo').mkdir()
        (self.subject / 'versions.txt').write_text("{'CSV': {'1.0': 'v1.0'}}", encoding='utf-8')
        (self.subject / 'bugs.txt').write_text("{'CSV': {'all': [1, 2], 'CSV_1_0': [1, 2], 'miss': []}}", encoding='utf-8')
        (self.subject / 'answers.txt').write_text("{'CSV': {'all': {1: 1, 2: 1}}}", encoding='utf-8')
        xml = '<bugrepository>'
        for number, label in ((1, 'org.demo.A.java'), (2, 'org.demo.Absent.java'), (3, 'org.demo.A.java')):
            xml += f'<bug id="{number}"><buginformation><summary>原始报告{number}</summary><description>original body</description><version>1.0</version></buginformation><fixedFiles><file>{label}</file></fixedFiles></bug>'
        (self.subject / 'bugrepo/repository.xml').write_text(xml + '</bugrepository>', encoding='utf-8')
        self.output = self.project / 'datasets/bench4bl_formatted'

    def convert(self):
        return prepare(self.project, self.raw, self.output, ['Commons/CSV'])

    def test_all_list_and_unresolved_gold_preserved(self):
        report = self.convert()
        from datasets import load_from_disk
        rows = load_from_disk(str(self.output))['test']
        self.assertEqual(report['reports'], 2)  # XML-only id 3 is not in the author selection.
        self.assertEqual(rows['issue_id'], [1, 2])
        self.assertEqual(rows[0]['before_fix_sha'], self.release)
        self.assertEqual(rows[0]['source_revision_role'], 'author_release_tag_not_per_bug_pre_fix')
        self.assertEqual(rows[0]['title'], '原始报告1')
        self.assertEqual(rows[1]['unresolved_gold_count'], 1)
        self.assertEqual(len(ast.literal_eval(rows[1]['updated_files'])), 1)
        self.assertEqual(git(self.repo, 'rev-parse', 'HEAD').decode().strip(), self.head)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.convert()

    def test_offline_release_source_cache_and_tamper_detection(self):
        self.convert()
        with patch('bug_localization.snapshot.urlopen') as network:
            source, manifest = get_snapshot(self.project, 'Bench4BL/Commons_CSV', self.release,
                source_provider='bench4bl_release', source_dataset='bench4bl_formatted')
            raw = (source / 'src/main/java/org/demo/A.java').read_text(encoding='utf-8')
            self.assertIn('return 0;', raw)
            self.assertNotIn('return 99;', raw)
            self.assertEqual(manifest['bench4bl_source']['release_tag'], 'v1.0')
            self.assertEqual(len(manifest['files']), 1)
            self.assertEqual(git(self.repo, 'rev-parse', 'HEAD').decode().strip(), self.head)
            get_snapshot(self.project, 'Bench4BL/Commons_CSV', self.release,
                source_provider='bench4bl_release', source_dataset='bench4bl_formatted')
            (source / 'src/main/java/org/demo/A.java').write_text('tampered', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'contents changed'):
                get_snapshot(self.project, 'Bench4BL/Commons_CSV', self.release,
                    source_provider='bench4bl_release', source_dataset='bench4bl_formatted')
            network.assert_not_called()

    def test_no_guessed_tag_or_remote_fallback(self):
        with self.assertRaises(ValueError):
            release_commit(self.repo, 'unknown-tag')
        with patch('bug_localization.snapshot.urlopen') as network:
            with self.assertRaisesRegex(ValueError, 'explicit'):
                get_snapshot(self.project, 'Bench4BL/Commons_CSV', self.release)
            network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
