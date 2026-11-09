"""Validate prepared LCA test commits, snapshots and candidate/gold coverage offline."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from datasets import load_from_disk
from bug_localization.bm25 import collect_files
from bug_localization.lca import save
from bug_localization.snapshot import get_snapshot


def audit(languages):
    import pyarrow.parquet as pq
    coverage = {}
    report = {'status': 'running', 'languages': {}, 'samples': [], 'storage': {}}
    for key in ('datasets', 'repos', 'cache', 'results'):
        report['storage'][key] = str((PROJECT / key).resolve())
    for lang in languages:
        raw = PROJECT / 'datasets/LCA_bug_localization'
        available = {p.stem.lower() for p in (raw / 'repos' / lang).glob('*.zip')}
        coverage[lang] = {}
        for split in ('train', 'dev', 'test'):
            raw_rows = pq.read_table(next((raw / lang).glob(split + '-*.parquet'))).to_pylist()
            missing = [{'id': r['id'], 'repository': r['repo_owner'] + '/' + r['repo_name'], 'base_sha': r['base_sha']}
                       for r in raw_rows if (r['repo_owner'] + '__' + r['repo_name']).lower() not in available]
            coverage[lang][split] = {'rows': len(raw_rows), 'rows_without_author_archive': len(missing),
                                     'missing': missing, 'adapted': split == 'test'}
        name = 'lca_' + lang + '_formatted'
        config = json.loads((PROJECT / 'configs/retrieval/lca' / lang / 'bm25.json').read_text())
        rows = load_from_disk(str(PROJECT / 'datasets' / name))['test']
        totals = Counter()
        for row in rows:
            source, manifest = get_snapshot(PROJECT, row['repo_name'], row['before_fix_sha'],
                                           source_provider='lca_git', source_dataset=name)
            documents, skipped = collect_files(source, manifest, config)
            candidates = {d['file_path'] for d in documents}
            gold = list(dict.fromkeys(ast.literal_eval(row['updated_files'])))
            missing = sorted(set(gold) - candidates)
            report['samples'].append({'language': lang, 'repository': row['repo_name'],
                'issue_id': row['issue_id'], 'base_sha': row['before_fix_sha'],
                'candidate_files': len(candidates), 'gold': gold, 'missing_gold': missing,
                'missing_from_exported_source': sorted(set(gold) - manifest['files'].keys()),
                'skipped_by_reason': dict(Counter(s['reason'] for s in skipped))})
            totals.update({'reports': 1, 'gold_files': len(gold), 'missing_gold': len(missing),
                           'reports_with_missing_gold': int(bool(missing)),
                           'empty_candidate_reports': int(not candidates)})
        report['languages'][lang] = dict(totals)
    save(PROJECT / 'datasets/LCA_bug_localization/metadata/split-source-coverage.json', coverage)
    report['status'] = 'complete'
    save(PROJECT / 'datasets/LCA_bug_localization/metadata/test-audit.json', report)
    print(json.dumps(report['languages'], indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--languages', nargs='+', choices=['java', 'py', 'kt'], default=['java', 'py', 'kt'])
    audit(parser.parse_args().languages)
