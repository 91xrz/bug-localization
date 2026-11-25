"""Compare lexical- and model-mined negatives on a fresh, matched historical C3 cohort."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics

from evaluate_c_lora import (METRICS, ROOT, check_disjoint, hard_pairs, prepare, ranking_metrics,
                             read, read_checked, save, score, sha256_file)

ROUND2 = ROOT.parent / 'round2'
SEED = 'c-lora-round2-training-negative-effect-frozen-v1'
MODES = ('base', 'lexical', 'hard')
PRIMARY = 'Paired report AP@10: hard minus lexical (mining effect); hard minus base (replacement). All selected issues.'


def freeze(root):
    prior = ROOT / 'evaluation/selection.json'
    prepare(root, 10, seed=SEED, additional_exclusions=(prior,), primary=PRIMARY)


def verify_training(run):
    data, training = run / 'data', run / 'training'
    proto = read(training / 'protocol.json')
    config = read(training / 'adapter/adapter_config.json')
    if (proto['selection'] != 'fixed final epoch' or proto['epochs'] != 2 or
            config['protocol_sha256'] != sha256_file(training/'protocol.json') or
            any(proto[k + '_sha256'] != sha256_file(data/(k + '.jsonl'))
                for k in ('train', 'validation'))):
        raise ValueError('Adapter protocol/data is not the frozen two-epoch experiment')
    return data, training / 'adapter'


def evaluate(root, mode, device):
    if mode == 'base':
        score(root, mode, device, None, None)
        return
    data, adapter = verify_training(ROOT if mode == 'lexical' else ROUND2)
    check_disjoint(root, read(root/'manifest.json'), data)
    score(root, 'lora', device, adapter, data, output_name=mode)


def contrast(rows, treatment, comparator, repetitions=10000):
    groups = defaultdict(list)
    gained, lost = 0, 0
    for r in rows:
        a, b = r[treatment], r[comparator]
        groups[r['repository']].append(a['average_precision_at_k'] - b['average_precision_at_k'])
        gained += a['top_1'] > b['top_1']
        lost += a['top_1'] < b['top_1']
    ds = [d for g in groups.values() for d in g]
    rng = random.Random(20261120)
    bootstrap = sorted(sum(sum(rng.choices(g, k=len(g))) for g in groups.values()) / len(ds)
                       for _ in range(repetitions))
    return dict(delta_MAP=statistics.mean(ds), wins=sum(d>0 for d in ds), ties=ds.count(0),
                losses=sum(d<0 for d in ds), top1_gained=gained, top1_lost=lost,
                project_delta_MAP={r:statistics.mean(g) for r,g in groups.items()},
                project_stratified_bootstrap_CI95=[bootstrap[int(.025*repetitions)-1],
                                                    bootstrap[int(.975*repetitions)-1]])


def summarize(root):
    manifest = read(root/'manifest.json')
    labels = read_checked(root/'labels.json', manifest['labels_sha256'])
    protocols, runs = {}, {}
    for m in MODES:
        runs[m] = read(root/m/'run.json')
        protocols[m] = read_checked(root/m/'protocol.json', runs[m]['protocol_sha256'])
        if protocols[m]['manifest_sha256'] != sha256_file(root/'manifest.json'):
            raise ValueError('Not scored on this manifest: ' + m)
    rows = []
    for i, lab in enumerate(labels, 1):
        h = manifest['input_sha256'][i-1]
        name = f'{i:03d}.json'
        item = read_checked(root/'inputs'/name, h)
        row = {k:lab[k] for k in ('id','repository','position','blagent_metrics','history_metrics')}
        for m in MODES:
            res = read_checked(root/m/'cases'/name, runs[m]['case_sha256'][i-1])
            if res['id'] != lab['id'] or res['input_sha256'] != h:
                raise ValueError('Case mismatch')
            row[m] = ranking_metrics([x['file_path'] for x in res['ranking']], lab['gold'], 10)
            row[m+'_hard_pair_accuracy'] = hard_pairs(item, res['ranking'], lab['gold'])
            row[m+'_status'] = res['status']
        rows.append(row)
    save(root/'paired_cases.json', rows)
    methods = (*MODES, 'history_metrics', 'blagent_metrics')
    summary = {m:{key:statistics.mean(r[m][k] for r in rows) for key,k in METRICS.items()} for m in methods}
    hard_rows = [r for r in rows if all(r[m+'_hard_pair_accuracy'] is not None for m in MODES)]
    summary.update(n=len(rows), primary=PRIMARY, historical_reproduced=sum(r['base']==r['history_metrics'] for r in rows),
                   hard_pair_reports=len(hard_rows),
                   hard_pair_accuracy={m:statistics.mean(r[m+'_hard_pair_accuracy'] for r in hard_rows)
                                       if hard_rows else None for m in MODES},
                   status_counts={m:Counter(r[m+'_status'] for r in rows) for m in MODES},
                   paired_cases_sha256=sha256_file(root/'paired_cases.json'),
                   contrasts={a+'_minus_'+b:contrast(rows,a,b) for a,b in (
                       ('hard','lexical'),('hard','base'),('hard','blagent_metrics'),('lexical','base'))})
    save(root/'paired_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', *MODES, 'summarize'))
    p.add_argument('--output', type=Path, default=ROUND2/'evaluation')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.mode == 'prepare':
        freeze(args.output)
    elif args.mode == 'summarize':
        summarize(args.output)
    else:
        evaluate(args.output, args.mode, args.device)
