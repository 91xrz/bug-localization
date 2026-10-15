"""Verify dependencies; optionally check the prepared local datasets."""

import argparse
import ast
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
EXPECTED_ENV = PROJECT / '.venv'
RESULTS = PROJECT / 'results' / 'environment'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-data', action='store_true', help='Also validate the three prepared local datasets')
    parser.add_argument('--semantic', action='store_true', help='Also verify embedding dependencies and CUDA computation when available')
    args = parser.parse_args()
    packages = ('datasets', 'pyarrow', 'elasticsearch')
    if args.semantic:
        packages += ('numpy', 'torch', 'transformers', 'tree-sitter', 'tree-sitter-python')
    for package in packages:
        module = {'tree-sitter': 'tree_sitter', 'tree-sitter-python': 'tree_sitter_python'}.get(package, package)
        importlib.import_module(module)
    report = {
        'python': sys.version,
        'executable': sys.executable,
        'environment': sys.prefix,
        'base_python': sys.base_prefix,
        'is_project_environment': Path(sys.prefix).resolve() == EXPECTED_ENV.resolve(),
        'is_virtual_environment': sys.prefix != sys.base_prefix,
        'packages': {name: importlib.metadata.version(name) for name in packages + ('pip',)},
        'datasets': {},
    }
    if args.semantic:
        import torch
        available = torch.cuda.is_available()
        semantic = {'cuda_available': available, 'cuda_runtime': torch.version.cuda,
                    'model_directory_present': (PROJECT / 'models/Qwen3-Embedding-0.6B').is_dir(),
                    'cuda_compute_passed': None}
        if available:
            semantic['gpu'] = torch.cuda.get_device_name(0)
            semantic['bf16_supported'] = torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if semantic['bf16_supported'] else torch.float32
            matrix = torch.randn((32, 32), device='cuda', dtype=dtype)
            semantic['cuda_compute_passed'] = bool(torch.isfinite(matrix @ matrix.T).all())
            if not semantic['cuda_compute_passed']:
                raise ValueError('CUDA matrix computation failed')
        report['semantic'] = semantic
    report['semantic_checked'] = args.semantic
    data_checks = [('BeetleBox', {'train': 10191, 'test': 8955}),
                                  ('swe_bench_formatted', {'test': 2294}),
                                  ('ye', {'train': 22709})] if args.check_data else []
    if args.check_data:
        from datasets import load_from_disk
    for name, expected_splits in data_checks:
        if not (PROJECT / 'datasets' / name / 'dataset_dict.json').is_file():
            raise FileNotFoundError(f'Prepare datasets/{name} first; see docs/datasets.md')
        data = load_from_disk(str(PROJECT / 'datasets' / name))
        if set(data) != set(expected_splits):
            raise ValueError('Unexpected splits for ' + name)
        report['datasets'][name] = {}
        for split, expected_rows in expected_splits.items():
            if len(data[split]) != expected_rows:
                raise ValueError('Unexpected row count for ' + name + '/' + split)
            row = data[split][0]
            labels = ast.literal_eval(row['updated_files'])
            if not isinstance(labels, list) or not all(isinstance(path, str) for path in labels):
                raise ValueError('Invalid updated_files for ' + name + '/' + split)
            report['datasets'][name][split] = {
                'rows': len(data[split]),
                'fields': data[split].column_names,
                'sample_repository': row['repo_name'],
                'sample_issue_id': row['issue_id'],
                'sample_label_count': len(labels),
            }
        if name == 'BeetleBox':
            report['datasets'][name]['test']['null_body_count'] = data['test'].data.column('body').null_count
            if report['datasets'][name]['test']['null_body_count'] != 0:
                raise ValueError('BeetleBox test still contains null bodies')
    report['data_checked'] = args.check_data
    report['status'] = 'passed'
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
