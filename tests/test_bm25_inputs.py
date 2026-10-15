import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

from bug_localization.bm25 import collect_files, run, validate_config


class RetrievalInputTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((PROJECT / 'configs/retrieval/bm25.json').read_text(encoding='utf-8'))

    def test_invalid_limits_fail_before_loading_data(self):
        for field, value in [('top_k', 0), ('top_k', 10001), ('max_query_terms', -1),
                             ('max_query_terms', 513), ('max_file_bytes', True)]:
            with self.subTest(field=field, value=value):
                config = copy.deepcopy(self.config)
                config[field] = value
                with patch('bug_localization.bm25.select_report') as select:
                    with self.assertRaises(ValueError):
                        run(PROJECT, config)
                    select.assert_not_called()

    def test_formatted_swe_solution_title_cannot_be_used_as_query(self):
        self.config['query_fields'] = ['title', 'body']
        with self.assertRaisesRegex(ValueError, 'solution PR'):
            validate_config(self.config)

    def test_bm25_parameters_reject_invalid_scores(self):
        for parameters in [{'k1': float('nan'), 'b': 0.75}, {'k1': 1.2, 'b': 1.01}]:
            with self.subTest(parameters=parameters):
                self.config['bm25'] = parameters
                with self.assertRaises(ValueError):
                    validate_config(self.config)

    def test_filtered_files_have_explicit_reasons(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            fixtures = {'ok.py': b'print(1)', 'readme.md': b'documentation',
                        'binary.py': b'a\x00b', 'large.py': b'x' * 20,
                        '__pycache__/skip.py': b'print(2)'}
            for relative, content in fixtures.items():
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            self.config['max_file_bytes'] = 10
            manifest = {'repository': 'test/repo', 'commit': 'a' * 40,
                        'files': {relative: 'test-hash' for relative in fixtures}}
            documents, skipped = collect_files(source, manifest, self.config)
            self.assertEqual([d['file_path'] for d in documents], ['ok.py'])
            self.assertEqual({r['file_path']: r['reason'] for r in skipped}, {
                'readme.md': 'extension_not_selected', 'binary.py': 'binary',
                'large.py': 'too_large', '__pycache__/skip.py': 'excluded_directory'})


if __name__ == '__main__':
    unittest.main()
