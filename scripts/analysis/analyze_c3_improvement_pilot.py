"""Audit frozen C3 pilots without altering inference artifacts, then pair complete cohorts.

Joint pilots are reconstructed from terminal case records, including runs whose
summary writer failed. Missing cases stay pending; no successful-only estimates
are substituted for complete-cohort results. This is analysis, never new inference.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.embeddings import digest
from bug_localization.paired_statistics import analyze as paired_analyze
from bug_localization.snapshot import sha256_file

C3 = PROJECT / 'results/chunk_ablation_reranking/20261112_C2_C3_500/C3'
BL = PROJECT / 'results/blagent/recovery-20261118/dual-v4/full2294'
IDENTITY = ('dataset_row', 'repository', 'issue_id', 'before_fix_sha')
HASH_FIELDS = ('query_sha256', 'labels_sha256')
METRIC_NAMES = ('MAP@10', 'MRR@10', 'Top-1', 'Top-5', 'Top-10', 'MeanRecall@10')


def read(path):
    return json.loads(path.read_text())


def key(row):
    return tuple(row[field] for field in IDENTITY)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_identity(row, sample, *, position=True):
    fields = IDENTITY + (('position',) if position else ())
    for field in fields:
        require(row.get(field) == sample[field], 'Sample identity differs: ' + field)
    for field in HASH_FIELDS:
        if field in row:
            require(row[field] == sample[field], 'Sample query/label hash differs: ' + field)


def ranking_paths(result):
    return [r['file_path'] if isinstance(r, dict) else r for r in result['ranking']]


def check_inventory(ranking, original, head=None):
    require(len(ranking) == len(set(ranking)), 'Duplicate ranking file')
    require(len(ranking) == len(original) and set(ranking) == set(original), 'Candidate inventory changed')
    if head is not None:
        require(ranking[head:] == original[head:], 'Joint reranking changed the untouched tail')
        require(set(ranking[:head]) == set(original[:head]), 'Joint head is not a permutation')
        if head == 10:
            require(set(ranking[:10]) == set(original[:10]), 'Top-10 membership changed')


def checked_result(folder, record):
    path = folder / record['result_file']
    require(sha256_file(path) == record['result_sha256'], 'Result checksum mismatch: ' + str(path))
    return read(path)


def check_metrics(result, record, gold):
    require(result['ground_truth'] == gold, 'Gold differs from frozen C3 labels')
    calculated = ranking_metrics(ranking_paths(result), gold, 10)
    require(calculated == result['metrics'] == record['metrics'], 'Ranking metrics differ')


def original_script(run, protocol, output, source_checks):
    """Snapshot the exact pre-fix script when the running file has since changed."""
    expected = protocol.get('script_sha256')
    if expected is not None:
        candidates = [PROJECT / 'scripts/experiments/pilot_c3_joint.py']
        candidates.extend(sorted(run.parent.glob('pilot_c3_joint*.py')))
    else:
        candidates = [PROJECT / 'scripts/experiments/pilot_c3_full_report.py']
        expected = protocol['files_sha256'].get(str(candidates[0].resolve()))
    require(bool(expected), 'Protocol does not identify its experiment script')
    match = next((p for p in candidates if p.exists() and sha256_file(p) == expected), None)
    require(match is not None, 'No exact experiment-script snapshot matching frozen protocol')
    target = output / 'source_snapshots' / (expected + '.py')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(sha256_file(target) == expected, 'Stored script snapshot differs')
    else:
        shutil.copyfile(match, target)
    source_checks.append({'source': str(match), 'saved': str(target), 'sha256': expected})


def audit_joint(run, sample, record, saved, inputs, protocol):
    case = run / 'cases' / f'{sample["position"]:04d}'
    original = ranking_paths(saved)
    head = protocol['head']
    paths = original[:head]
    evidence = {p: [{'chunk_id': c['chunk_id'], 'text': c['embedding_text']}
                    for c in saved['scored_chunks'] if c['file_path'] == p] for p in paths}
    query = inputs['query']
    input_path = case / 'inference_input.json'
    if input_path.exists():
        observed = read(input_path)
        require(observed['query'] == query and observed['candidates'] == paths
                and observed['evidence'] == evidence, 'Joint model input differs from original report/C3 evidence')
    contents = '\n\n'.join('FILE ' + p + '\n' + '\n\n'.join(c['text'] for c in evidence[p])
                           for p in sorted(paths))
    user = 'BUG REPORT:\n' + query + '\n\nCANDIDATE EVIDENCE:\n' + contents
    schema = {'type': 'object', 'properties': {'reasoning': {'type': 'string'},
              'ranked_files': {'type': 'array', 'items': {'enum': sorted(paths)},
                               'minItems': len(paths), 'maxItems': len(paths)}},
              'required': ['reasoning', 'ranked_files'], 'additionalProperties': False}
    calls = sorted(case.glob('call_*.json'))
    require(len(calls) <= 2, 'Joint inference call budget exceeded')
    for number, path in enumerate(calls):
        call = read(path)
        request = call['request']
        require(call['request_sha256'] == digest(request), 'Model request checksum mismatch')
        require(set(request) == {'messages', 'temperature', 'stop', 'stage', 'max_new_tokens',
                                 'identity', 'response_schema'}, 'Unexpected fields in model request')
        expected_user = user + ('\nPrevious output repeated or omitted a file. Return each path exactly once: '
                               + json.dumps(sorted(paths)) if number else '')
        require(request['messages'] == [{'role': 'system', 'content': protocol['prompt']},
                                       {'role': 'user', 'content': expected_user}], 'Model prompt differs from allowed inputs')
        require(request['response_schema'] == schema, 'Candidate constraint differs')
        require(request['stage'] == 'C3_joint' and request['temperature'] == 0
                and request['identity'] == digest(protocol['generator_identity']), 'Generation identity/settings differ')
    if record['status'] == 'ok':
        require(calls and input_path.exists(), 'Successful joint case missing model audit')
        result = checked_result(run, record)
        require(result['model_calls'] == len(calls), 'Model call count differs')
        for path in calls:
            require(read(path)['status'] == 'ok', 'Successful case contains a failed model call')
        last = read(calls[-1])
        from bug_localization.blagent_recovery import json_object
        response = json_object(last['response'])
        require(response['ranked_files'] == result['ranking'][:len(paths)]
                and response['reasoning'] == result['reasoning'], 'Final rank is not the actual model response')
        check_inventory(result['ranking'], original, head)
        check_metrics(result, record, saved['ground_truth'])
    else:
        require((case / 'error.json').exists(), 'Failed joint case lacks error diagnostic')
    return len(calls)


def audit_full_report(run, record, saved, inputs):
    result = checked_result(run, record)
    check_identity(result, record)
    require(result['status'] == record['status'], 'Case and record statuses differ')
    if record['status'] != 'ok':
        require(result.get('error'), 'Failed full-report case lacks diagnostic')
        return
    require(result['query'] == inputs['query'] and result['reranker_query_sha256'] == digest(inputs['query'])
            and result['baseline_query'] == saved['query'], 'Full-report text differs')
    original_chunks = [{k: c[k] for k in ('file_path', 'chunk_id', 'embedding_text')} for c in saved['scored_chunks']]
    chunks = result['scored_chunks']
    require([{k: c[k] for k in ('file_path', 'chunk_id', 'embedding_text')} for c in chunks] == original_chunks,
            'Full-report evidence differs from frozen C3 chunks')
    require(result['selected_chunk_ids'] == [c['chunk_id'] for c in original_chunks], 'Selected chunk IDs differ')
    best = {}
    for chunk in chunks:
        require(math.isfinite(chunk['score']), 'Nonfinite reranker score')
        best[chunk['file_path']] = max(best.get(chunk['file_path'], -math.inf), chunk['score'])
    rows = sorted(saved['ranking'], key=lambda r: r['semantic_rank'])
    ordered = sorted([r for r in rows if r['file_path'] in best],
                     key=lambda r: (-best[r['file_path']], r['semantic_rank']))
    expected = [r['file_path'] for r in ordered] + [r['file_path'] for r in saved['ranking'] if r['file_path'] not in best]
    require(ranking_paths(result) == expected, 'File max aggregation or semantic tie order differs')
    check_inventory(ranking_paths(result), ranking_paths(saved))
    check_metrics(result, record, saved['ground_truth'])


def summary(records, total):
    result = aggregate_results(records, 10, total)
    # Suppress successful-only numbers too, preventing a pending cohort from
    # accidentally being presented as a completed experiment.
    result.pop('metrics_successful_only', None)
    if not result['pending_samples'] and result['failed_samples']:
        result['status'] = 'completed_with_failures'
    return result


def analyze(runs, selection_path, output, *, verify_sources=False, repetitions=3000):
    output.mkdir(parents=True, exist_ok=True)
    selection = read(selection_path)
    samples = selection['samples']
    require(samples and len({key(s) for s in samples}) == len(samples), 'Empty or duplicate selection')
    require(len({s['position'] for s in samples}) == len(samples), 'Duplicate selection positions')
    require(len({p.name for p in runs}) == len(runs), 'Run directory names must be unique')
    c3_index = {key(r): r for r in read(C3 / 'samples.json')}
    bl_index = {key(r): r for r in read(BL / 'samples.json')}
    bl_selection = {key(r): r for r in read(BL / 'selection.json')['samples']}
    c3_rows, bl_rows, source_data = [], [], {}
    for sample in samples:
        cr, br = c3_index[key(sample)], bl_index[key(sample)]
        check_identity(cr, sample)
        check_identity(br, sample, position=False)
        check_identity(bl_selection[key(sample)], sample, position=False)
        saved = checked_result(C3, cr) if cr['status'] == 'ok' else None
        if saved:
            check_metrics(saved, cr, saved['ground_truth'])
        if br['status'] == 'ok':
            bd = checked_result(BL, br)
            require(br['metrics'] == ranking_metrics(bd['ranking'], bd['gold_files'], 10), 'BL baseline metrics differ')
            if saved:
                require(bd['gold_files'] == saved['ground_truth'], 'C3/BL gold labels differ')
        inputs_path = BL / 'cases' / f'{br["position"]:04d}' / 'inputs.json'
        inputs = read(inputs_path) if inputs_path.exists() else None
        source_data[sample['position']] = (cr, saved, inputs, inputs_path)
        c3_rows.append({**cr, **{f: sample[f] for f in HASH_FIELDS}})
        bl_rows.append({**br, **{f: sample[f] for f in HASH_FIELDS}})
    methods = {'C3': c3_rows, 'BLAgent': bl_rows}
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'analysis_script_sha256': sha256_file(Path(__file__)),
              'selection_file': str(selection_path), 'selection_sha256': sha256_file(selection_path),
              'cohort': selection.get('cohort'), 'selected_samples': len(samples), 'source_hashes_requested': verify_sources,
              'note': 'Paired exploratory analysis of previously observed development reports; not independent blind testing. '
                      'Joint summaries are reconstructed from saved terminal records, not new inference. '
                      'Generation latency is incremental over cached C3, not full-pipeline latency.',
              'baselines': {m: summary(rows, len(samples)) for m, rows in methods.items()}, 'runs': {}}
    for run in runs:
        protocol = read(run / 'protocol.json')
        manifest = read(run / 'selection.json')
        require(manifest == selection, 'Run selection differs from requested paired cohort')
        joint = 'head' in protocol
        expected_selection = sha256_file(selection_path) if joint else digest(samples)
        require(protocol['selection_sha256'] == expected_selection, 'Protocol selection checksum differs')
        snapshots = []
        original_script(run, protocol, output, snapshots)
        if verify_sources and not joint:
            for path, checksum in protocol['files_sha256'].items():
                # Script has already been matched to its exact snapshot above.
                if Path(path).name == 'pilot_c3_full_report.py':
                    continue
                require(sha256_file(Path(path)) == checksum, 'Frozen source/model checksum differs: ' + path)
        indexed = {}
        if joint:
            paths = sorted((run / 'cases').glob('*/record.json'))
            raw_rows = [read(p) for p in paths]
        else:
            raw_rows = read(run / 'samples.json') if (run / 'samples.json').exists() else []
        allowed_positions = {s['position'] for s in samples}
        for rec in raw_rows:
            require(rec['position'] in allowed_positions and rec['position'] not in indexed, 'Unknown/duplicate terminal case')
            indexed[rec['position']] = rec
        records, missing_gold_recovered, calls = [], [], 0
        for sample in samples:
            rec = indexed.get(sample['position'])
            if rec is None:
                continue
            check_identity(rec, sample)
            require(rec['status'] in ('ok', 'failed'), 'Nonterminal case record')
            cr, saved, inputs, inputs_path = source_data[sample['position']]
            if verify_sources and joint:
                for path, checksum in rec['source_sha256'].items():
                    require(sha256_file(Path(path)) == checksum, 'Frozen input checksum differs: ' + path)
            if saved is None:
                require(rec['status'] == 'failed', 'Failed C3 source became successful without source evidence')
            elif joint:
                calls += audit_joint(run, sample, rec, saved, inputs, protocol)
            else:
                audit_full_report(run, rec, saved, inputs)
            normalized = dict(rec)
            if rec['status'] == 'ok':
                if 'missing_gold' not in normalized:
                    normalized['missing_gold'] = cr['missing_gold']
                    missing_gold_recovered.append(sample['position'])
                else:
                    require(normalized['missing_gold'] == cr['missing_gold'], 'Coverage metadata differs from source C3')
            records.append(normalized)
        aggregated = summary(records, len(samples))
        missing = [s['position'] for s in samples if s['position'] not in indexed]
        run_report = {'run': str(run), 'method': protocol['method'], 'protocol_sha256': sha256_file(run / 'protocol.json'),
                      'audit_status': 'passed' if not missing else 'pending', 'summary': aggregated,
                      'pending_positions': missing, 'audited_calls': calls, 'source_snapshots': snapshots,
                      'missing_gold_restored_in_memory_positions': missing_gold_recovered,
                      'summary_recovery': 'Terminal case records are authoritative; old summaries/protocol/results were not edited. '
                                          'Restored coverage metadata comes from C3 only and does not change ranks or metrics.'}
        report['runs'][run.name] = run_report
        if not missing:
            methods[run.name] = records
    contrasts = {f'{name} minus {baseline}': {name: 1, baseline: -1}
                 for name in methods if name not in ('C3', 'BLAgent') for baseline in ('C3', 'BLAgent')}
    report['paired'] = paired_analyze(methods, contrasts, repetitions=repetitions) if contrasts else None
    report['multiplicity_note'] = ('Holm adjustment covers every reported contrast and metric in this invocation; '
                                   'other adaptive pilot invocations are outside this family. Treat development comparisons as exploratory, '
                                   'not evidence of global SOTA or an independent confirmatory test.')
    report['status'] = 'complete' if all(r['audit_status'] == 'passed' for r in report['runs'].values()) else 'pending'
    write_json(output / 'analysis.json', report)
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['cohort', 'method', 'status', 'successful', 'failed', 'pending', *METRIC_NAMES])
        writer.writeheader()
        summaries = {**report['baselines'], **{name: value['summary'] for name, value in report['runs'].items()}}
        for name, value in summaries.items():
            writer.writerow({'cohort': report['cohort'], 'method': name, 'status': value['status'],
                             'successful': value['successful_samples'], 'failed': value['failed_samples'],
                             'pending': value['pending_samples'], **(value['metrics'] or {})})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-sources', action='store_true', help='Also hash all recorded input/model sources; may be expensive.')
    parser.add_argument('--repetitions', type=int, default=3000)
    args = parser.parse_args()
    require(2000 <= args.repetitions <= 50000, 'Use at least 2000 inference repetitions')
    report = analyze([(PROJECT / p).resolve() for p in args.run], (PROJECT / args.selection).resolve(),
                     (PROJECT / args.output).resolve(), verify_sources=args.verify_sources, repetitions=args.repetitions)
    print(json.dumps({'status': report['status'], 'selected_samples': report['selected_samples'],
                      'runs': {k: {'audit_status': v['audit_status'], 'pending': v['summary']['pending_samples']}
                               for k, v in report['runs'].items()}}, indent=2))


if __name__ == '__main__':
    main()
