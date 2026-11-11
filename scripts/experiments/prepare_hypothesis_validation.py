"""Freeze a balanced cross-project selection without consulting retrieval outcomes."""
import hashlib
import json
from pathlib import Path
import random
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import write_json
from bug_localization.embeddings import digest
from bug_localization.snapshot import sha256_file

REPOSITORIES = ('astropy/astropy', 'django/django', 'matplotlib/matplotlib', 'pydata/xarray',
                'pylint-dev/pylint', 'pytest-dev/pytest', 'scikit-learn/scikit-learn',
                'sphinx-doc/sphinx', 'sympy/sympy')
SEED = 20261110


def main():
    from datasets import load_from_disk
    target = PROJECT / 'results/report_hypothesis/selections/crossproject_180_seed20261110'
    target.mkdir(parents=True, exist_ok=True)
    data = load_from_disk(str(PROJECT / 'datasets/swe_bench_formatted'))['test']
    # Previous module-2 cross-project protocol included the first 11 pytest reports,
    # all Seaborn/Flask reports, and all Requests reports. None may enter validation.
    previous_pytest = [i for i, row in enumerate(data) if row['repo_name'] == 'pytest-dev/pytest'][:11]
    excluded = set(previous_pytest) | {i for i, row in enumerate(data) if row['repo_name'] in
                                      ('psf/requests', 'mwaskom/seaborn', 'pallets/flask')}
    excluded_bodies = {data[i]['body'] for i in excluded}
    chosen = []
    for repository in REPOSITORIES:
        available = [i for i, row in enumerate(data) if row['repo_name'] == repository
                     and i not in excluded and row['body'] not in excluded_bodies]
        rng = random.Random(f'{SEED}:{repository}')
        chosen.extend(sorted(rng.sample(available, 20)))
    entries = []
    for position, i in enumerate(chosen, 1):
        row = data[i]
        entries.append({'position': position, 'dataset_row': i, 'repository': row['repo_name'],
                        'issue_id': row['issue_id'], 'before_fix_sha': row['before_fix_sha'],
                        'query_sha256': hashlib.sha256(json.dumps({'body': row['body']}, sort_keys=True).encode()).hexdigest(),
                        'labels_sha256': hashlib.sha256(str(row.get('updated_files')).encode()).hexdigest()})
    manifest = {'dataset_metadata_sha256': sha256_file(PROJECT / 'datasets/swe_bench_formatted/test/dataset_info.json'),
                'selection_sha256': digest(entries), 'samples': entries,
                'selection_rule': '20 reports per repository, random.Random(seed:repository), no outcome filtering; excludes prior module-2 reports and identical bodies',
                'seed': SEED, 'excluded_dataset_rows': sorted(excluded)}
    path = PROJECT / 'configs/experiments/selections/crossproject_180_seed20261110.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Existing frozen selection differs; refusing overwrite')
    else:
        write_json(path, manifest)
    print(path)
    print('samples', len(entries), 'selection_sha256', manifest['selection_sha256'])


if __name__ == '__main__':
    main()
