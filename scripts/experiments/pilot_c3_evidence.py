"""Run paired, frozen-input ordinary/evidence-grounded C3 head reranking.

This is a development mechanism experiment, not an independent test or a claim
of verified causality. The two methods share all report/code input and budgets.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / p) for p in ('.runtime/blagent-deps', '.runtime/rrl-deps', 'src')]
from bug_localization.batch import aggregate_results, write_json
from bug_localization.blagent_recovery import json_object
from bug_localization.blagent_runtime import ServerGenerator
from bug_localization.bm25 import ranking_metrics
from bug_localization.embeddings import digest
from bug_localization.evidence_reranking import (
    BASELINE_PROMPT, EVIDENCE_PROMPT, build_input, build_messages,
    response_schema, resolve_ranking, validate_response,
)
from bug_localization.snapshot import sha256_file

C3 = PROJECT / 'results/chunk_ablation_reranking/20261112_C2_C3_500/C3'
REPORTS = PROJECT / 'results/blagent/recovery-20261118/dual-v4/full2294'
IDENTITY = ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha',
            'query_sha256', 'labels_sha256')


def read(path):
    return json.loads(path.read_text())


def frozen_write(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen artifact changed: {path}')
    else:
        write_json(path, value)


def infer(generator, payload, variant, folder):
    messages = build_messages(payload, variant)
    schema = response_schema(payload, variant)
    tokens = generator.count_tokens(messages)
    write_json(folder / 'inference_input.json', {**payload, 'prompt_tokens': tokens})
    validation = []
    for attempt in range(2):
        response = generator.complete(messages, stage='C3_' + variant,
                                      temperature=0, response_schema=schema)
        try:
            value = json_object(response)
            errors = validate_response(value, payload, variant)
        except (ValueError, TypeError) as error:
            value, errors = None, [str(error)]
        validation.append({'attempt': attempt + 1, 'errors': errors})
        write_json(folder / 'validation.json', validation)
        if not errors:
            return value
        if attempt == 0:
            # Retry validates formatting/grounding only, never consults outcomes.
            messages = build_messages(payload, variant)
            messages[-1]['content'] += (
                '\n\nOUTPUT VALIDATION FEEDBACK (not code or bug evidence):\n' +
                '\n'.join(errors[:8]) +
                '\nGenerate a complete corrected JSON response. Use only supplied IDs, '
                'exact report quotes and excerpt-local line ranges. Each file appears once.')
    raise ValueError('Invalid structured reranking after two calls: ' + '; '.join(errors[:8]))


def summary(records, total):
    result = aggregate_results(records, 10, total)
    result.pop('metrics_successful_only', None)
    result['failure_policy'] = 'Failures count as zero in the full denominator; no fallback'
    result['pending_samples'] = total - len(records)
    if len(records) != total:
        result['metrics'] = None
        result.pop('metrics_successful_only', None)
    return result


def load_source(sample, baseline):
    source = baseline[sample['position']]
    if any(source[k] != sample[k] for k in IDENTITY):
        raise ValueError('Frozen C3 sample identity mismatch')
    if source['status'] != 'ok':
        return source, None, {}, None
    result_path = C3 / source['result_file']
    report_path = REPORTS / 'cases' / f'{sample["dataset_row"] + 1:04d}' / 'inputs.json'
    if sha256_file(result_path) != source['result_sha256']:
        raise ValueError('C3 source result checksum mismatch')
    saved, original = read(result_path), read(report_path)
    if any(saved[k] != sample[k] for k in IDENTITY):
        raise ValueError('C3 result identity mismatch')
    if original['source_commit'] != sample['before_fix_sha']:
        raise ValueError('Original report source commit mismatch')
    source_hashes = {str(p): sha256_file(p) for p in (result_path, report_path)}
    return source, saved, source_hashes, original['query']


def run(selection, output, head=10):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(selection, output, head)


def _run_locked(selection, output, head):
    started = time.monotonic()
    manifest = read(selection)
    samples = manifest['samples']
    if not samples or len({s['position'] for s in samples}) != len(samples):
        raise ValueError('Empty or duplicate sample selection')
    baseline = {r['position']: r for r in read(C3 / 'samples.json')}
    # Prepare and pin every source before any inference, without using outcomes
    # to select reports. Labels remain outside build_input and model requests.
    prepared = {}
    for sample in samples:
        source, saved, hashes, query = load_source(sample, baseline)
        prepared[sample['position']] = (source, saved, hashes,
                                       None if saved is None else build_input(saved, query, head))
    settings = {
        'generation_model': 'models/Qwen3-8B', 'inference_endpoint': 'http://127.0.0.1:8099',
        'server_manifest': 'models/BLAgent-Qwen3-8B-BF16/manifest.json', 'server_slots': 2,
        'max_context_tokens': 32768, 'max_new_tokens': 2048, 'max_calls': 2,
        'seed': 42, 'request_timeout': 600,
    }
    generator = ServerGenerator(PROJECT, settings)
    sources = [Path(__file__), PROJECT / 'src/bug_localization/evidence_reranking.py',
               PROJECT / 'src/bug_localization/blagent_runtime.py',
               PROJECT / 'src/bug_localization/blagent_recovery.py']
    code_hashes = {str(p.relative_to(PROJECT)): sha256_file(p) for p in sources}
    plan = {
        'experiment': 'C3 matched-input fault-evidence comparison', 'head': head,
        'selection_sha256': sha256_file(selection), 'n': len(samples),
        'variants': ['baseline', 'evidence'], 'code_sha256': code_hashes,
        'generator_identity': generator.identity, 'failure_policy': 'zero; no fallback or sample replacement',
        'primary_contrast': 'evidence - baseline', 'primary_metric': 'MAP@10',
        'analysis': 'paired full denominator; six metrics Holm; project-group and report-paired uncertainty',
        'development_only': True, 'no_outcome_adaptive_prompt_changes': True,
        'shared_input': 'full original report; C3 Top10; original saved chunks; alphabetical IDs and local line numbers',
        'source_sha256': {str(p): v[2] for p, v in prepared.items()},
        'interpretation': 'tests explicit evidence comparison; no new retrieval, context expansion or training',
        'success_criterion': 'exploratory positive MAP difference; independent confirmation required',
    }
    frozen_write(output / 'plan.json', plan)
    snapshot = output / 'implementation'
    snapshot.mkdir(exist_ok=True)
    for path in sources:
        target = snapshot / path.name
        if target.exists() and sha256_file(target) != sha256_file(path):
            raise ValueError('Frozen implementation snapshot changed')
        if not target.exists():
            shutil.copyfile(path, target)
    for variant in plan['variants']:
        folder = output / variant
        folder.mkdir(exist_ok=True)
        frozen_write(folder / 'selection.json', manifest)
        frozen_write(folder / 'protocol.json', {
            **plan, 'method': variant, 'prompt': BASELINE_PROMPT if variant == 'baseline' else EVIDENCE_PROMPT,
            'max_new_tokens': 2048, 'max_calls': 2,
        })
    write_json(output / 'server_properties.json', generator.server_properties)
    # Token budget is checked for the entire cohort before any expensive calls.
    token_audit = []
    for sample in samples:
        payload = prepared[sample['position']][3]
        if payload is not None:
            counts = {v: generator.count_tokens(build_messages(payload, v)) for v in plan['variants']}
            token_audit.append({'position': sample['position'], **counts})
    frozen_write(output / 'input_token_audit.json', token_audit)

    def work(sample, variant):
        case_start = time.monotonic()
        folder = output / variant
        case = folder / 'cases' / f'{sample["position"]:04d}'
        case.mkdir(parents=True, exist_ok=True)
        source, saved, hashes, payload = prepared[sample['position']]
        record_path = case / 'record.json'
        if record_path.exists():
            record = read(record_path)
            if any(record.get(k) != sample[k] for k in IDENTITY) or record['source_sha256'] != hashes:
                raise ValueError('Checkpoint identity/source mismatch')
            for key in ('result', 'inference_input', 'validation'):
                if key + '_file' in record:
                    if sha256_file(folder / record[key + '_file']) != record[key + '_sha256']:
                        raise ValueError('Checkpoint artifact checksum mismatch: ' + key)
            return variant, record
        for call_path in sorted(case.glob('call_*.json')):
            cached = read(call_path)
            if cached.get('status') == 'running' or (
                    cached.get('status') == 'failed' and cached.get('error_type') != 'ValueError'):
                # The runtime refuses failed cached calls with ValueError. Do not
                # turn an interrupted service/request into an algorithm failure.
                raise RuntimeError('Unfinished infrastructure call requires explicit recovery; '
                                   'no scored failure was created: ' + str(call_path))
        record = {**sample, 'source_sha256': hashes, 'missing_gold': source.get('missing_gold', [])}
        g = copy.copy(generator)
        g.start_case(case)
        try:
            if payload is None:
                raise ValueError('C3 source failed; counted as failure in full denominator')
            value = infer(g, payload, variant, case)
            ranking = resolve_ranking(value, payload)
            if ranking[head:] != payload['tail'] or set(ranking) != set(payload['original_ranking']):
                raise ValueError('Candidate inventory or tail changed')
            result = {
                'ranking': ranking, 'reasoning': value['reasoning'], 'structured_output': value,
                'ground_truth': saved['ground_truth'],
                'metrics': ranking_metrics(ranking, saved['ground_truth'], 10), 'model_calls': g.calls,
                'mechanical_validation': 'passed; does not certify semantic or causal correctness',
            }
            write_json(case / 'result.json', result)
            record.update(status='ok', metrics=result['metrics'],
                          result_file=str((case / 'result.json').relative_to(folder)),
                          result_sha256=sha256_file(case / 'result.json'))
        except (OSError, ConnectionError, TimeoutError, RuntimeError) as error:
            write_json(case / 'infrastructure_error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
            raise
        except ValueError as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
            write_json(case / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        for key in ('inference_input', 'validation'):
            path = case / (key + '.json')
            if path.exists():
                record[key + '_file'] = str(path.relative_to(folder))
                record[key + '_sha256'] = sha256_file(path)
        record['model_calls'] = g.calls
        record['elapsed_seconds'] = time.monotonic() - case_start
        write_json(record_path, record)
        return variant, record

    records = {v: [] for v in plan['variants']}
    # Alternate within-pair dispatch order; same shared server, max two workers.
    jobs = [(sample, variant) for sample in samples
            for variant in (plan['variants'] if int(digest(sample['dataset_row'])[:8], 16) % 2 == 0
                            else list(reversed(plan['variants'])))]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(work, sample, variant) for sample, variant in jobs]
        for future in as_completed(futures):
            try:
                variant, record = future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
            records[variant].append(record)
            records[variant].sort(key=lambda r: r['position'])
            write_json(output / variant / 'samples.json', records[variant])
            write_json(output / variant / 'summary.json', summary(records[variant], len(samples)))
            progress = {v: {'completed': len(rs), 'ok': sum(r['status'] == 'ok' for r in rs),
                            'failed': sum(r['status'] == 'failed' for r in rs)} for v, rs in records.items()}
            write_json(output / 'progress.json', {'methods': progress, 'total_per_method': len(samples),
                       'wall_seconds_this_invocation': time.monotonic() - started,
                       'updated_at': datetime.now(timezone.utc).isoformat()})
            print(json.dumps(progress), flush=True)
    write_json(output / 'completion.json', {'status': 'completed', 'n': len(samples),
               'wall_seconds_this_invocation': time.monotonic() - started,
               'at': datetime.now(timezone.utc).isoformat(), 'development_only': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run((PROJECT / args.selection).resolve(), (PROJECT / args.output).resolve())
