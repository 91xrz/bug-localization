"""Verify the project environment using existing local datasets only."""

import ast
import importlib.metadata
import json
from pathlib import Path
import sys

from datasets import load_from_disk


PROJECT = Path(__file__).resolve().parents[2]
EXPECTED_ENV = PROJECT / '.venv'
RESULTS = PROJECT / 'results' / 'environment'


def main():
    assert Path(sys.prefix).resolve() == EXPECTED_ENV.resolve(), 'Use the project .venv Python'
    assert sys.prefix != sys.base_prefix, 'Python is not running in an isolated environment'
    report = {
        'python': sys.version,
        'executable': sys.executable,
        'environment': sys.prefix,
        'base_python': sys.base_prefix,
        'packages': {name: importlib.metadata.version(name) for name in ('datasets', 'pyarrow', 'pip')},
        'datasets': {},
    }
    for name, expected_splits in [('BeetleBox', {'train': 10191, 'test': 8955}),
                                  ('swe_bench_formatted', {'test': 2294}),
                                  ('ye', {'train': 22709})]:
        data = load_from_disk(str(PROJECT / 'datasets' / name))
        assert set(data) == set(expected_splits), 'Unexpected splits for ' + name
        report['datasets'][name] = {}
        for split, expected_rows in expected_splits.items():
            assert len(data[split]) == expected_rows, 'Unexpected row count for ' + name + '/' + split
            row = data[split][0]
            labels = ast.literal_eval(row['updated_files'])
            assert isinstance(labels, list) and all(isinstance(path, str) for path in labels)
            report['datasets'][name][split] = {
                'rows': len(data[split]),
                'fields': data[split].column_names,
                'sample_repository': row['repo_name'],
                'sample_issue_id': row['issue_id'],
                'sample_label_count': len(labels),
            }
        if name == 'BeetleBox':
            report['datasets'][name]['test']['null_body_count'] = data['test'].data.column('body').null_count
            assert report['datasets'][name]['test']['null_body_count'] == 0
    report['status'] = 'passed'
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
