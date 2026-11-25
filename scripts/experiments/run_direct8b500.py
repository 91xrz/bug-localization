"""Direct 8B reranking of pre-reranker RRF Top10 on the frozen 500 reports.

Reconstruct candidates/evidence from semantic retrieval, never from 0.6B scores.
Historical reranker outputs are read only to audit provenance and evaluate controls.
"""
import argparse
import copy
import csv
import fcntl
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pilot_c3_joint import PROMPT, ServerGenerator, infer
from run_mixed_recall500 import Sources, same_sample, now
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.chunk_ablation import select_evidence
from bug_localization.mixed_recall import mixed_candidates
from bug_localization.paired_statistics import analyze
from bug_localization.snapshot import sha256_file


def read(path):
    return json.loads(path.read_text())


def direct_inputs(case, head=10):
    """Accept only retrieval/chunks; no reranker ranking, score, or gold needed."""
    original, hypothesis = case['original']['ranking'], case['hypothesis']['ranking']
    candidates = mixed_candidates(original, hypothesis, 'hypothesis', 40, 50)
    ar = {r['file_path']: r['rank'] for r in original}
    br = {r['file_path']: r['rank'] for r in hypothesis}
    if ar.keys() != br.keys():
        raise ValueError('Unpaired retrieval inventories')
    pool = {r['file_path'] for r in candidates}
    all_paths = sorted(ar, key=lambda p: (-(1 / (60 + ar[p]) + 1 / (60 + br[p])), p))
    order = [p for p in all_paths if p in pool]
    full_order = order + [p for p in all_paths if p not in pool]
    ranking = [{'file_path': p, 'rank': i, 'score': 1 / (60 + ar[p]) + 1 / (60 + br[p])}
               for i, p in enumerate(full_order, 1)]
    _, chunks = select_evidence(candidates, ranking, case['chunks'],
                               case['original']['chunk_scores'], case['hypothesis']['chunk_scores'], 'mixed')
    paths = order[:head]
    evidence = {p: [{'chunk_id': c['chunk_id'], 'text': c['embedding_text']}
                    for c in chunks if c['file_path'] == p] for p in paths}
    if any(not rows for rows in evidence.values()):
        raise ValueError('Missing candidate code evidence')
    return {'query': case['original']['query'], 'candidates': paths,
            'evidence': evidence, 'tail': order[head:]}


def prepare(output, config_path):
    config = read(config_path)
    if config['head'] != 10 or config['rrf_k'] != 60:
        raise ValueError('This frozen experiment requires RRF60 / Top10')
    source = Sources(read(ROOT / config['source_config']))
    selection = read(ROOT / config['selection'])
    if selection != source.selection:
        raise ValueError('Selection differs from retrieval sources')
    baseline_root = ROOT / config['baseline']
    baselines = read(baseline_root / 'samples.json')
    if read(baseline_root / 'verification.json')['status'] != 'passed':
        raise ValueError('Historical controls must have passed audit')
    for rows in baselines.values():
        if len(rows) != 500:
            raise ValueError('Incomplete baseline')
        for sample, row in zip(source.samples, rows):
            same_sample(sample, row)
    files = [config_path, ROOT / config['selection'], ROOT / config['source_config'],
             baseline_root / 'samples.json', baseline_root / 'verification.json', Path(__file__),
             Path(__file__).with_name('pilot_c3_joint.py'), Path(__file__).with_name('run_mixed_recall500.py')]
    files += [ROOT / 'src/bug_localization' / name for name in
              ('blagent_runtime.py', 'chunk_ablation.py', 'mixed_recall.py', 'bm25.py', 'batch.py', 'paired_statistics.py')]
    output.mkdir(parents=True, exist_ok=True)
    (output / 'prepared').mkdir(exist_ok=True)
    entries = []
    for sample in source.samples:
        pos = sample['position']
        target = output / 'prepared' / f'{pos:04d}.json'
        if source.failures(sample):
            if pos != 308:
                raise ValueError('Unexpected upstream failure')
            entry = {'sample': sample, 'status': 'source_failed',
                     'error': 'Existing hypothesis generation failure; no fallback'}
        else:
            case = source.load(sample)
            inputs = direct_inputs(case, config['head'])
            baseline = baselines['A+B'][pos - 1]
            path = baseline_root / baseline['result_file']
            if sha256_file(path) != baseline['result_sha256']:
                raise ValueError('Historical result hash differs')
            saved = read(path)
            same_sample(saved, sample)
            hashes = {str(p.resolve()): sha256_file(p) for p in case['paths']}
            if hashes != saved['source_hashes']:
                raise ValueError('Retrieval sources differ from audited control')
            full = inputs['candidates'] + inputs['tail']
            if full != saved['ab_ranking']:
                raise ValueError('Reconstructed pre-reranker RRF order differs')
            for p, evidence in inputs['evidence'].items():
                expected = [{'chunk_id': c['chunk_id'], 'text': c['embedding_text']}
                            for c in saved['scored_chunks'] if c['file_path'] == p]
                if evidence != expected:
                    raise ValueError('Reconstructed mixed evidence differs')
            for name, ranking in [('A+B', full), ('A+B+C', [r['file_path'] for r in saved['ranking']])]:
                if ranking_metrics(ranking, case['gold'], 10) != baselines[name][pos - 1]['metrics']:
                    raise ValueError('Baseline metric recomputation differs')
            entry = {'sample': sample, 'status': 'ready', 'inference': inputs,
                     'ground_truth': case['gold'], 'missing_gold': sorted(set(case['gold']) - case['inventory']),
                     'source_sha256': {**hashes, str(path.resolve()): sha256_file(path)}}
        if target.exists() and read(target) != entry:
            raise ValueError('Prepared inputs changed; use a new output directory')
        write_json(target, entry)
        entries.append({'position': pos, 'file': str(target.relative_to(output)), 'sha256': sha256_file(target)})
        if pos % 10 == 0:
            print(f'Prepared and source-audited {pos}/500', flush=True)
    protocol = {'config': config, 'prompt': PROMPT, 'files_sha256': {str(p.resolve()): sha256_file(p) for p in files},
                'prepared': entries, 'selected': 500, 'expected_upstream_failures': [308],
                'no_06b_dependency': 'All 8B inputs rebuilt solely from semantic retrieval and source chunks',
                'caveat': 'Previously observed development500; Top10-only reranking preserves recall ceiling'}
    if (output / 'protocol.json').exists() and read(output / 'protocol.json') != protocol:
        raise ValueError('Frozen experiment protocol changed')
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'selection.json', selection)
    write_json(output / 'status.json', {'status': 'prepared', 'updated_at': now(), 'selected': 500})


def verify(output):
    protocol = read(output / 'protocol.json')
    for p, checksum in protocol['files_sha256'].items():
        if sha256_file(Path(p)) != checksum:
            raise ValueError('Frozen file changed: ' + p)
    for entry in protocol['prepared']:
        if sha256_file(output / entry['file']) != entry['sha256']:
            raise ValueError('Prepared input changed')
    return protocol


def run(output):
    protocol = verify(output)
    generator = ServerGenerator(ROOT, protocol['config']['generator'])
    identity_path = output / 'generator_identity.json'
    if identity_path.exists() and read(identity_path) != generator.identity:
        raise ValueError('Generator identity changed')
    write_json(identity_path, generator.identity)
    def work(item):
        entry = read(output / item['file']); sample = entry['sample']
        folder = output / 'cases' / f'{sample["position"]:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        record_path = folder / 'record.json'
        if record_path.exists():
            record = read(record_path); same_sample(sample, record)
            if record['prepared_sha256'] != item['sha256']:
                raise ValueError('Checkpoint inputs changed')
            if record['status'] == 'ok' and sha256_file(output / record['result_file']) != record['result_sha256']:
                raise ValueError('Checkpoint result changed')
            return record
        started = time.monotonic()
        record = {**sample, 'prepared_sha256': item['sha256']}
        if entry['status'] == 'source_failed':
            record.update(status='failed', error_type='SourceDependencyFailure', error=entry['error'])
        else:
            g = copy.copy(generator); g.start_case(folder)
            inputs = entry['inference']
            try:
                ranked, reasoning = infer(g, inputs['query'], inputs['candidates'], inputs['evidence'], folder)
                ranking = ranked + inputs['tail']
                result = {'ranking': ranking, 'reasoning': reasoning, 'ground_truth': entry['ground_truth'],
                          'metrics': ranking_metrics(ranking, entry['ground_truth'], 10), 'model_calls': g.calls}
                write_json(folder / 'result.json', result)
                record.update(status='ok', metrics=result['metrics'], missing_gold=entry['missing_gold'],
                              result_file=str((folder / 'result.json').relative_to(output)),
                              result_sha256=sha256_file(folder / 'result.json'))
            except (OSError, ConnectionError, TimeoutError, RuntimeError):
                write_json(folder / 'infrastructure_error.json', {'traceback': traceback.format_exc()})
                raise
            except ValueError as error:
                record.update(status='failed', error_type=type(error).__name__, error=str(error))
        record['elapsed_seconds'] = time.monotonic() - started
        write_json(record_path, record)
        return record
    records = []; started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(work, item) for item in protocol['prepared']]
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
            records.sort(key=lambda r: r['position'])
            summary = aggregate_results(records, 10, 500)
            summary['wall_seconds_this_invocation'] = time.monotonic() - started
            write_json(output / 'samples.json', records); write_json(output / 'summary.json', summary)
            write_json(output / 'status.json', {'status': 'running', 'processed': len(records),
                       'successful': summary['successful_samples'], 'failed': summary['failed_samples'], 'updated_at': now()})
            print(f'{len(records)}/500 successful={summary["successful_samples"]} failed={summary["failed_samples"]}', flush=True)
    finalize(output)


def finalize(output):
    protocol = verify(output)
    records = read(output / 'samples.json')
    if len(records) != 500 or [r['position'] for r in records] != list(range(1, 501)):
        raise ValueError('Incomplete final sample set')
    calls = []; verified = 0
    for record, item in zip(records, protocol['prepared']):
        entry = read(output / item['file']); same_sample(record, entry['sample'])
        folder = output / 'cases' / f'{record["position"]:04d}'
        for path in sorted(folder.glob('call_*.json')):
            call = read(path); calls.append(call)
            from bug_localization.embeddings import digest
            if digest(call['request']) != call['request_sha256']:
                raise ValueError('Model call hash mismatch')
            if call['request']['messages'][0]['content'] != PROMPT:
                raise ValueError('Prompt changed')
            inputs = entry['inference']
            contents = '\n\n'.join('FILE ' + p + '\n' + '\n\n'.join(c['text'] for c in inputs['evidence'][p])
                                   for p in sorted(inputs['candidates']))
            wanted = 'BUG REPORT:\n' + inputs['query'] + '\n\nCANDIDATE EVIDENCE:\n' + contents
            if path.stem != 'call_001':
                wanted += '\nPrevious output repeated or omitted a file. Return each path exactly once: ' + json.dumps(sorted(inputs['candidates']))
            if call['request']['messages'][1]['content'] != wanted:
                raise ValueError('Actual model request differs from label-free input')
        if record['status'] == 'ok':
            path = output / record['result_file']
            if sha256_file(path) != record['result_sha256']:
                raise ValueError('Result checksum mismatch')
            result = read(path); inputs = entry['inference']; head = len(inputs['candidates'])
            ranking = result['ranking']
            if len(ranking[:head]) != len(set(ranking[:head])) or set(ranking[:head]) != set(inputs['candidates']) or ranking[head:] != inputs['tail']:
                raise ValueError('Candidate permutation or unchanged tail audit failed')
            last_call = read(sorted(folder.glob('call_*.json'))[-1])
            from bug_localization.blagent_recovery import json_object
            if json_object(last_call['response'])['ranked_files'] != ranking[:head]:
                raise ValueError('Ranking differs from actual model response')
            if ranking_metrics(ranking, entry['ground_truth'], 10) != record['metrics'] or result['metrics'] != record['metrics']:
                raise ValueError('Independent metric recomputation failed')
            verified += 1
    baselines = read(ROOT / protocol['config']['baseline'] / 'samples.json')
    methods = {'no_rerank_RRF': baselines['A+B'], 'reranker_06B': baselines['A+B+C'], 'direct_8B': records}
    groups = {'all500': set(range(1, 501)), 'previous180': set(range(1, 181)), 'new320': set(range(181, 501))}
    summary = {group: {name: aggregate_results([r for r in rows if r['position'] in positions], 10)
                       for name, rows in methods.items()} for group, positions in groups.items()}
    write_json(output / 'comparison.json', summary)
    write_json(output / 'project_summary.json', {repo: {name: aggregate_results([r for r in rows if r['repository'] == repo], 10)
               for name, rows in methods.items()} for repo in sorted({r['repository'] for r in records})})
    stats = analyze(methods, {'8B_minus_no_rerank': {'direct_8B': 1, 'no_rerank_RRF': -1},
                            '8B_minus_06B': {'direct_8B': 1, 'reranker_06B': -1}}, repetitions=5000)
    write_json(output / 'paired_statistics.json', stats)
    costs = {'calls': len(calls), 'input_tokens': sum(c.get('input_tokens', 0) for c in calls),
             'output_tokens': sum(c.get('output_tokens', 0) for c in calls),
             'incremental_case_seconds': sum(r['elapsed_seconds'] for r in records),
             'note': 'Incremental 8B rerank only; cached retrieval/hypothesis excluded; not end-to-end latency'}
    write_json(output / 'cost.json', costs)
    with (output / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['cohort', 'method', 'successful', 'MAP@10', 'MRR@10', 'Top1', 'Top5', 'Top10'])
        for group, rows in summary.items():
            for name, row in rows.items():
                m = row['metrics']; writer.writerow([group, name, row['successful_samples'], m['MAP@10'], m['MRR@10'], m['Top-1'], m['Top-5'], m['Top-10']])
    write_json(output / 'verification.json', {'status': 'passed', 'selected': 500, 'verified_successes': verified,
               'checks': ['retrieval-only candidates and evidence', 'source hashes', 'actual prompt content without labels or 0.6B scores',
                          'model response permutation', 'unchanged tail', 'independent metrics', 'full denominator']})
    write_json(output / 'status.json', {'status': 'completed', 'processed': 500, 'successful': verified,
               'failed': 500 - verified, 'updated_at': now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/direct8b_reranker500.json')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare:
            prepare(output, args.config.resolve())
        elif args.finalize:
            finalize(output)
        else:
            run(output)
