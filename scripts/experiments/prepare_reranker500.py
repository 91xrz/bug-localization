"""Freeze a balanced 500-case expansion, keeping 320 new report bodies disjoint."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import write_json
from bug_localization.embeddings import digest


def expand(data, old, total=500, seed=20261111):
    repos = sorted({r['repository'] for r in old['samples']})
    excluded = set(old['excluded_dataset_rows'])
    used = {data[r['dataset_row']]['body'] for r in old['samples']} | {data[i]['body'] for i in excluded}
    counts = Counter(r['repository'] for r in old['samples'])
    pools = {}
    for repo in repos:
        candidates = [i for i, row in enumerate(data) if row['repo_name'] == repo and i not in excluded]
        random.Random(f'{seed}:{repo}').shuffle(candidates)
        pools[repo] = candidates
    chosen = []
    while len(chosen) + len(old['samples']) < total:
        available = []
        for repo in repos:
            while pools[repo] and data[pools[repo][-1]]['body'] in used:
                pools[repo].pop()
            if pools[repo]:
                available.append(repo)
        if not available:
            raise ValueError('Not enough distinct unseen reports')
        repo = min(available, key=lambda name: (counts[name], name))
        row_index = pools[repo].pop()
        chosen.append(row_index)
        used.add(data[row_index]['body'])
        counts[repo] += 1
    chosen.sort(key=lambda i: (data[i]['repo_name'], i))
    new = []
    for position, index in enumerate(chosen, 1):
        row = data[index]
        new.append({'position': position, 'dataset_row': index, 'repository': row['repo_name'],
            'issue_id': row['issue_id'], 'before_fix_sha': row['before_fix_sha'],
            'query_sha256': hashlib.sha256(json.dumps({'body': row['body']}, sort_keys=True).encode()).hexdigest(),
            'labels_sha256': hashlib.sha256(str(row['updated_files']).encode()).hexdigest()})
    combined = old['samples'] + [{**r, 'position': len(old['samples']) + i} for i, r in enumerate(new, 1)]
    common = {'dataset_metadata_sha256': old['dataset_metadata_sha256'], 'seed': seed,
        'selection_rule': 'Keep frozen original 180; add randomly ordered unseen unique bodies, balancing project counts without using outcomes',
        'excluded_dataset_rows': sorted(excluded), 'old_selection_sha256': old['selection_sha256']}
    return ({**common, 'samples': combined, 'selection_sha256': digest(combined), 'project_counts': dict(counts)},
            {**common, 'samples': new, 'selection_sha256': digest(new), 'cohort': 'new320'})


def main():
    from datasets import load_from_disk
    old = json.loads((PROJECT / 'configs/experiments/selections/crossproject_180_seed20261110.json').read_text())
    data = load_from_disk(str(PROJECT / 'datasets/swe_bench_formatted'))['test']
    combined, new = expand(data, old)
    for name, value in [('crossproject_500_seed20261111.json', combined), ('crossproject_new320_seed20261111.json', new)]:
        path = PROJECT / 'configs/experiments/selections' / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Refusing to replace a different frozen selection: ' + str(path))
        write_json(path, value)
    print(json.dumps({'total': len(combined['samples']), 'new': len(new['samples']), 'projects': combined['project_counts']}))


if __name__ == '__main__':
    main()
