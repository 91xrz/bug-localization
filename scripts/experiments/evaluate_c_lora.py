"""Frozen, project-stratified C3 comparison. Only the adapter changes."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import statistics
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / p) for p in ('src', 'scripts/experiments')]
from bug_localization.bm25 import ranking_metrics
from bug_localization.bm25 import prepare_report
from bug_localization.embeddings import digest
from bug_localization.reranking import LocalReranker, rerank
from bug_localization.snapshot import sha256_file

C3 = PROJECT / 'results/chunk_ablation_reranking/20261112_C2_C3_500/C3'
BL = PROJECT / 'results/blagent/recovery-20261118/dual-v4/full2294'
PILOT = PROJECT / 'results/c3_improvement_pilot/20261119'
ROOT = PROJECT / 'results/c_lora/round1'
KEY = ('dataset_row', 'repository', 'issue_id', 'before_fix_sha')
SEED = 'c-lora-disjoint-history-per-project-v1'
METRICS = dict(zip(('MAP@10', 'MRR@10', 'Top-1', 'Top-5', 'Top-10', 'MeanRecall@10'),
                   ('average_precision_at_k', 'reciprocal_rank_at_k', 'top_1', 'top_5', 'top_10', 'recall_at_k')))


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    tmp = path.with_suffix('.partial')
    tmp.write_text(text, encoding='utf8')
    tmp.replace(path)


def identity(row):
    return tuple(row[k] for k in KEY)


def select(rows, excluded, per_project=10, seed=SEED):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = identity(row)
        if key in seen:
            raise ValueError('duplicate source identity')
        seen.add(key)
        if key not in excluded:
            groups[row['repository']].append(row)
    result = []
    for repo in sorted(groups):
        if len(groups[repo]) < per_project:
            raise ValueError('inadequate fixed sample size: ' + repo)
        result += sorted(groups[repo], key=lambda row: digest([seed, identity(row)]))[:per_project]
    return result


def checked_result(root, row):
    path = root / row['result_file']
    if sha256_file(path) != row['result_sha256']:
        raise ValueError(f'source checksum mismatch: {path}')
    data = read(path)
    if identity(data) != identity(row):
        raise ValueError('source identity mismatch')
    return data


def frozen_input(case):
    chunks = [{k: c[k] for k in ('file_path', 'chunk_id', 'embedding_text')} for c in case['scored_chunks']]
    by_file = Counter(c['file_path'] for c in chunks)
    original = [{'rank': r['semantic_rank'], 'score': r['semantic_score'], 'file_path': r['file_path']}
                for r in sorted(case['ranking'], key=lambda r: r['semantic_rank'])]
    paths = [r['file_path'] for r in original]
    if (not paths or len(paths) > 50 or len(paths) != len(set(paths)) or set(by_file) != set(paths)
            or max(by_file.values()) > 3 or len({c['chunk_id'] for c in chunks}) != len(chunks)
            or set(paths) != {r['file_path'] for r in case['candidates']}):
        raise ValueError('invalid frozen C3 pool/evidence')
    return {'query': case['query'], 'original_order': original, 'chunks': chunks}


def prepare(root, per_project, *, seed=SEED, additional_exclusions=(), primary=None):
    if (root / 'manifest.json').exists():
        raise FileExistsError('frozen evaluation already exists')
    rows = read(C3 / 'samples.json')
    exclude_files = [PILOT / f'{n}.json' for n in ('development45', 'confirmation90')]
    excluded = {identity(r) for p in exclude_files for r in read(p)['samples']}
    excluded.update(identity(r) for p in additional_exclusions for r in read(p))
    exclude_files += list(additional_exclusions)
    selected = select(rows, excluded, per_project, seed=seed)
    selection = [dict(position=i, c3_position=r['position'], **{k:r[k] for k in KEY})
                 for i, r in enumerate(selected, 1)]
    save(root / 'selection.json', selection)
    bl = {identity(r): r for r in read(BL / 'samples.json')}
    from datasets import load_from_disk
    dataset = load_from_disk(str(PROJECT/'datasets/swe_bench_formatted/test'))
    labels, inputs = [], []
    for index, row in enumerate(selected, 1):
        repo = row['repository']
        key = identity(row)
        uid = f"{repo}#{row['issue_id']}@{row['before_fix_sha']}"
        item = dict(position=index, id=uid, repository=repo, source_status=row['status'])
        b = bl[key]
        bl_inputs = BL / f"cases/{b['position']:04d}/inputs.json"
        body = read(bl_inputs)['query']
        bcase = checked_result(BL, b) if b['status'] == 'ok' else None
        ccase = checked_result(C3, row) if row['status'] == 'ok' else None
        if ccase:
            item.update(frozen_input(ccase))
            if not body.startswith(item['query']):
                raise ValueError(f'C3 query not prefix of original body: {uid}')
        srow, gold, normalized = prepare_report(dataset[row['dataset_row']],
            {'repository':repo, 'issue_id':row['issue_id'], 'query_fields':['body']})
        if body.strip() != normalized or srow['before_fix_sha'] != row['before_fix_sha']:
            raise ValueError('dataset revision or body mismatch')
        if ccase and set(ccase['ground_truth']) != set(gold):
            raise ValueError('original C3 gold differs from dataset')
        if bcase and set(bcase['gold_files']) != set(gold):
            raise ValueError('baseline gold disagreement')
        hist = ranking_metrics([r['file_path'] for r in ccase['ranking']] if ccase else [], gold, 10)
        bm = ranking_metrics(bcase['ranking'] if bcase else [], gold, 10)
        if (ccase and hist != row['metrics']) or (bcase and bm != b['metrics']):
            raise ValueError('source metrics did not reproduce')
        path = root / 'inputs' / f'{index:03d}.json'
        save(path, item)
        inputs.append(sha256_file(path))
        labels.append(dict(position=index, id=uid, repository=repo, gold=gold, history_metrics=hist,
                           blagent_metrics=bm, blagent_status=b['status'],
                           body_sha256=sha256(body.encode()).hexdigest(), source_inputs_sha256=sha256_file(bl_inputs),
                           c3_source_sha256=row.get('result_sha256'), blagent_source_sha256=b.get('result_sha256')))
    save(root / 'labels.json', labels)
    config = read(C3 / 'effective_config.json')
    save(root / 'manifest.json', dict(seed=seed, per_project=per_project, n=len(labels),
         primary=primary or 'paired mean AP@10 (lora minus base); all selected issues retained',
         diagnostic='Within each report, compare every in-pool gold to the top 5 non-gold by pre-reranking order; ties count 1/2.',
         historical_development=True, previous_pilot_exclusions={p.name:sha256_file(p) for p in exclude_files},
         source_samples_sha256={str(p):sha256_file(p) for p in (C3/'samples.json', BL/'samples.json')},
         base_model_files=config['model']['files_sha256'], config=config['reranker'],
         input_sha256=inputs, labels_sha256=sha256_file(root/'labels.json'),
         selection_sha256=sha256_file(root/'selection.json')))


def read_checked(path, checksum):
    if sha256_file(path) != checksum:
        raise ValueError(f'frozen file changed: {path}')
    return read(path)


def infer(item, model):
    if item['source_status'] != 'ok':
        return dict(id=item['id'], status='source_failed', ranking=[], scored_chunks=[])
    chunks = item['chunks']
    scores = model.score(item['query'], [c['embedding_text'] for c in chunks])
    ranked = rerank(item['original_order'], chunks, scores)
    return dict(id=item['id'], status='ok', ranking=ranked,
                scored_chunks=[{**{k:c[k] for k in ('file_path','chunk_id')}, **s} for c,s in zip(chunks,scores)])


def check_disjoint(root, man, data):
    labels = read_checked(root/'labels.json', man['labels_sha256'])
    repos = {x['repository'].lower() for x in labels}
    hashes = {x['body_sha256'] for x in labels}
    result = {}
    for part in ('train', 'validation'):
        p = data / (part + '.jsonl')
        records = [json.loads(line) for line in p.read_text().splitlines()]
        if not records:
            raise ValueError('empty ' + part)
        if any(r['repo'].lower() in repos or sha256(r['query'].encode()).hexdigest() in hashes
               or r.get('body_sha256') in hashes for r in records):
            raise ValueError('train or validation overlaps test')
        result[part] = dict(n=len(records), sha256=sha256_file(p))
    return result


def score(root, mode, device, adapter, data, *, output_name=None):
    man = read(root/'manifest.json')
    out = root / (output_name or mode)
    result_path = out/'run.json'
    if result_path.exists():
        raise FileExistsError(f'{mode} run complete')
    audit = None
    if mode == 'lora':
        audit = check_disjoint(root, man, data)
    model = LocalReranker(PROJECT, {**man['config'], 'device': device})
    if model.identity['files_sha256'] != man['base_model_files']:
        raise ValueError('model differs from C3')
    adapter_hash = None
    if mode == 'lora':
        from train_c_lora import load_adapter
        load_adapter(model, adapter)
        adapter_hash = {str(p.relative_to(adapter)):sha256_file(p) for p in sorted(adapter.rglob('*')) if p.is_file()}
    protocol = dict(mode=mode, manifest_sha256=sha256_file(root/'manifest.json'), config=model.config,
                    model=model.identity, adapter_sha256=adapter_hash, disjoint=audit)
    protocol_path = out/'protocol.json'
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError('cannot resume: model or inputs changed')
    save(protocol_path, protocol)
    hashes = []
    for i, h in enumerate(man['input_sha256'], 1):
        item = read_checked(root/'inputs'/f'{i:03d}.json', h)
        case_path = out/'cases'/f'{i:03d}.json'
        stamp = out/'checksums'/f'{i:03d}.json'
        if stamp.exists():
            saved = read_checked(case_path, read(stamp)['sha256'])
            if saved['id'] != item['id'] or saved['input_sha256'] != h:
                raise ValueError('resume input changed')
        else:
            start = time.monotonic()
            saved = infer(item, model)
            save(case_path, {**saved, 'input_sha256':h, 'seconds':time.monotonic()-start})
            save(stamp, {'sha256':sha256_file(case_path)})
        hashes.append(sha256_file(case_path))
        if i % 10 == 0:
            print(f'{mode}: completed {i}/{man["n"]}', flush=True)
    save(result_path, dict(protocol_sha256=sha256_file(protocol_path), case_sha256=hashes))


def hard_pairs(item, ranked, gold):
    positives = set(gold)
    scores = {r['file_path']:r['score'] for r in ranked}
    pos = [scores[p] for p in positives if p in scores]
    neg = [scores[r['file_path']] for r in item.get('original_order', []) if r['file_path'] not in positives][:5]
    if not pos or not neg:
        return None
    return statistics.mean((p > n) + 0.5*(p == n) for p in pos for n in neg)


def summarize(root):
    man = read(root/'manifest.json')
    labels = read_checked(root/'labels.json', man['labels_sha256'])
    runs = {name:read(root/name/'run.json') for name in ('base','lora')}
    rows = []
    for i, lab in enumerate(labels, 1):
        item = read_checked(root/'inputs'/f'{i:03d}.json', man['input_sha256'][i-1])
        row = {k:lab[k] for k in ('position','repository','id','history_metrics','blagent_metrics')}
        for name, run in runs.items():
            r = read_checked(root/name/'cases'/f'{i:03d}.json', run['case_sha256'][i-1])
            if r['id'] != lab['id'] or r['input_sha256'] != man['input_sha256'][i-1]:
                raise ValueError('paired identity mismatch')
            row[name] = ranking_metrics([x['file_path'] for x in r['ranking']], lab['gold'], 10)
            row[name+'_hard_pair_accuracy'] = hard_pairs(item, r['ranking'], lab['gold'])
            row[name+'_status'] = r['status']
        rows.append(row)
    save(root/'paired_cases.json', rows)
    variants = ('base','lora','history_metrics','blagent_metrics')
    summary = {n:{display:statistics.mean(r[n][k] for r in rows) for display,k in METRICS.items()} for n in variants}
    rng = random.Random(20261120)
    groups = defaultdict(list)
    for r in rows:
        groups[r['repository']].append(r['lora']['average_precision_at_k'] - r['base']['average_precision_at_k'])
    ds = [d for group in groups.values() for d in group]
    boots = sorted(sum(sum(rng.choices(v,k=len(v))) for v in groups.values())/len(ds) for _ in range(10000))
    hard = [r for r in rows if r['base_hard_pair_accuracy'] is not None and r['lora_hard_pair_accuracy'] is not None]
    summary.update(n=len(rows), delta_MAP=statistics.mean(ds), project_stratified_bootstrap_CI95=[boots[249],boots[9749]],
                   wins=sum(d>0 for d in ds), ties=ds.count(0), losses=sum(d<0 for d in ds),
                   hard_pair_eligible=len(hard), cases_sha256=sha256_file(root/'paired_cases.json'),
                   hard_pair_accuracy={n:statistics.mean(r[n+'_hard_pair_accuracy'] for r in hard) if hard else None for n in runs},
                   status_counts={n:Counter(r[n+'_status'] for r in rows) for n in runs})
    save(root/'paired_summary.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('prepare','base','lora','summarize'))
    ap.add_argument('--output', type=Path, default=ROOT/'evaluation')
    ap.add_argument('--per-project', type=int, default=10)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--adapter', type=Path, default=ROOT/'training/adapter')
    ap.add_argument('--data', type=Path, default=ROOT/'data')
    args = ap.parse_args()
    if args.mode == 'prepare':
        prepare(args.output, args.per_project)
    elif args.mode == 'summarize':
        summarize(args.output)
    else:
        score(args.output, args.mode, args.device, args.adapter, args.data)


if __name__ == '__main__':
    main()
