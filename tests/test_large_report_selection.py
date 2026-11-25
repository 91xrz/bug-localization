"""Large batches require explicit bounds and a content-verified pinned manifest."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bug_localization.embeddings import digest
from bug_localization.snapshot import sha256_file
from bug_localization.vector import select_reports


class LargeSelectionTests(unittest.TestCase):
    def setup_data(self, root):
        metadata = root / 'datasets/test/test/dataset_info.json'
        metadata.parent.mkdir(parents=True)
        metadata.write_text('{}')
        rows = [{'repo_name': 'test/repo', 'issue_id': i, 'before_fix_sha': str(i),
                 'body': f'Report {i}', 'updated_files': "['a.py']"} for i in range(60)]
        entries = [{'position': i+1, 'dataset_row': i, 'repository': row['repo_name'],
                    'issue_id': row['issue_id'], 'before_fix_sha': row['before_fix_sha'],
                    'query_sha256': hashlib.sha256(json.dumps({'body': row['body']},sort_keys=True).encode()).hexdigest(),
                    'labels_sha256': hashlib.sha256(str(row['updated_files']).encode()).hexdigest()}
                   for i,row in enumerate(rows)]
        (root / 'selection.json').write_text(json.dumps({'dataset_metadata_sha256': sha256_file(metadata),
            'selection_sha256': digest(entries), 'samples': entries}))
        return rows, {'selection_manifest': 'selection.json'}, {'dataset':'test','split':'test','query_fields':['body']}

    def test_explicit_large_bound_and_content_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);rows,config,base=self.setup_data(root)
            with patch('datasets.load_from_disk',return_value={'test':rows}):
                with self.assertRaisesRegex(ValueError,'1-50'):
                    select_reports(root,config,base)
                selected,manifest=select_reports(root,config,base,max_samples=60)
                self.assertEqual(len(selected),60)
                rows[0]['body']='Changed after freezing'
                with self.assertRaisesRegex(ValueError,'changed'):
                    select_reports(root,config,base,max_samples=60)

    def test_large_selection_without_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'frozen selection'):
            select_reports(Path('.'),{}, {},max_samples=180)

    def test_invalid_bound_is_rejected(self):
        for value in [0,True,5001]:
            with self.assertRaisesRegex(ValueError,'max_samples'):
                select_reports(Path('.'),{}, {},max_samples=value)


if __name__=='__main__':
    unittest.main()
