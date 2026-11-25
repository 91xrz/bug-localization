"""Matched C3 reranking with lexical versus local-structure context additions.

All original code excerpts are retained. Only the bounded source additions vary;
the ranker, candidate pool, full report and output protocol are shared.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import importlib.util
import inspect
import io
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
import tokenize
import traceback
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / p) for p in ('.runtime/blagent-deps', '.runtime/rrl-deps', 'src')]
from bug_localization.batch import write_json
from bug_localization.blagent_runtime import ServerGenerator
from bug_localization.bm25 import ranking_metrics
from bug_localization.context_evidence import render_segments, select_context
from bug_localization.embeddings import digest
from bug_localization.evidence_reranking import BASELINE_PROMPT, build_input, build_messages, resolve_ranking
from bug_localization.snapshot import sha256_file

SPEC = importlib.util.spec_from_file_location('c3_context_base', PROJECT / 'scripts/experiments/pilot_c3_evidence.py')
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
read, frozen_write, infer, summary = BASE.read, BASE.frozen_write, BASE.infer, BASE.summary
VARIANTS = ('baseline', 'lexical', 'structural')
HEAD, BUDGET = 10, 256


def build_context_input(saved, query, sample, variant, token_len, budget=BUDGET):
    if variant not in VARIANTS:
        raise ValueError('Unknown context selection variant')
    payload = build_input(saved, query, HEAD)
    snapshot_base = PROJECT / 'repos' / sample['repository'] / sample['before_fix_sha']
    manifest_path = snapshot_base / 'snapshot.json'
    snapshot = read(manifest_path)
    if snapshot['repository'] != sample['repository'] or snapshot['commit'] != sample['before_fix_sha']:
        raise ValueError('Source snapshot is not the pinned pre-fix revision')
    source_root = (snapshot_base / 'source').resolve()
    payload.update(source_context={}, base_user_content_sha256=payload['user_content_sha256'],
                   context_budget_per_file=budget, context_token_count=0,
                   source_snapshot={'root': str(source_root), 'manifest_file': str(manifest_path.resolve()),
                                    'manifest_sha256': sha256_file(manifest_path),
                                    'repository': sample['repository'], 'commit': sample['before_fix_sha'],
                                    'archive_file': str((snapshot_base / 'snapshot.zip').resolve()),
                                    'archive_sha256': snapshot['archive_sha256'],
                                    'archive_root': sample['repository'].split('/')[-1] + '-' + sample['before_fix_sha']})
    additions = []
    if variant != 'baseline':
        for file_id, path in payload['catalog']['files'].items():
            context = select_context(source_root, path, query,
                                     [c['text'] for c in payload['evidence'][path]], token_len,
                                     variant=variant, budget_tokens=budget,
                                     expected_sha256=snapshot['files'][path])
            payload['source_context'][path] = context
            payload['context_token_count'] += context['token_count']
            if context['rendered_text']:
                additions.append(context['rendered_text'])
    if additions:
        payload['user_content'] += '\n\nADDITIONAL SOURCE CONTEXT:\n' + '\n\n'.join(additions)
    payload['user_content_sha256'] = digest(payload['user_content'])
    return payload


def verify_source_archive(sample, saved):
    """Check the existing pinned archive and the exact candidate file bytes."""
    base = PROJECT / 'repos' / sample['repository'] / sample['before_fix_sha']
    manifest = read(base / 'snapshot.json')
    if manifest['repository'] != sample['repository'] or manifest['commit'] != sample['before_fix_sha']:
        raise ValueError('Archive manifest does not identify the pre-fix revision')
    archive = base / 'snapshot.zip'
    if sha256_file(archive) != manifest['archive_sha256']:
        raise ValueError('Pinned source archive checksum mismatch')
    expected_root = sample['repository'].split('/')[-1] + '-' + sample['before_fix_sha']
    with zipfile.ZipFile(archive) as handle:
        if {name.split('/')[0] for name in handle.namelist()} != {expected_root}:
            raise ValueError('Source archive root does not match pre-fix commit')
        for row in saved['ranking'][:HEAD]:
            path = row['file_path']
            actual = hashlib.sha256(handle.read(expected_root + '/' + path)).hexdigest()
            if actual != manifest['files'][path]:
                raise ValueError('Candidate source differs from pinned archive: ' + path)


def prepare(selection, token_len):
    manifest = read(selection)
    samples = manifest['samples']
    if not samples or len({s['position'] for s in samples}) != len(samples):
        raise ValueError('Empty or duplicate selection')
    baseline = {r['position']: r for r in read(BASE.C3 / 'samples.json')}
    prepared = {}
    for sample in samples:
        source, saved, hashes, query = BASE.load_source(sample, baseline)
        if saved is None:
            # This mechanism experiment requires saved evidence for all arms.
            # Abort preparation instead of selecting a successful-only subset.
            raise ValueError('Upstream C3 failure in fixed cohort; no reports were removed')
        verify_source_archive(sample, saved)
        payloads = {v: build_context_input(saved, query, sample, v, token_len) for v in VARIANTS}
        prepared[sample['position']] = (source, saved, hashes, payloads)
        print(f'Prepared source evidence {len(prepared)}/{len(samples)}', flush=True)
    return manifest, prepared


def prepared_cache_identity(selection):
    return {'selection_sha256': sha256_file(selection), 'head': HEAD, 'budget': BUDGET,
            'selector_sha256': sha256_file(PROJECT / 'src/bug_localization/context_evidence.py'),
            'original_builder_sha256': sha256_file(PROJECT / 'src/bug_localization/evidence_reranking.py'),
            'renderer_sha256': digest(inspect.getsource(build_context_input)),
            'tokenizer_sha256': {name: sha256_file(PROJECT / 'models/Qwen3-8B' / name)
                                 for name in ('tokenizer.json', 'tokenizer_config.json')},
            'token_counting': 'AutoTokenizer.encode; add_special_tokens=False'}


def freeze_prepared_cache(selection, output):
    samples = read(selection)['samples']
    files = {f'{s["position"]:04d}.json': sha256_file(output / 'prepared_inputs' / f'{s["position"]:04d}.json')
             for s in samples}
    frozen_write(output / 'prepared_cache.json', {**prepared_cache_identity(selection), 'files': files,
                 'scope': 'Source context preparation only; no prior model outputs are reused'})


def load_prepared_cache(selection, output, token_len):
    """Recheck base inputs and every source slice without repeating AST selection.

    Independent analysis replays the frozen selector after inference as well.
    """
    cache = read(output / 'prepared_cache.json')
    if any(cache.get(k) != v for k, v in prepared_cache_identity(selection).items()):
        raise ValueError('Prepared context cache implementation/selection/tokenizer changed')
    manifest = read(selection)
    baseline = {r['position']: r for r in read(BASE.C3 / 'samples.json')}
    prepared = {}
    changed = {'source_context', 'context_token_count', 'user_content', 'user_content_sha256'}
    for sample in manifest['samples']:
        source, saved, hashes, query = BASE.load_source(sample, baseline)
        if saved is None:
            raise ValueError('Upstream C3 failed; cannot consume context cache')
        verify_source_archive(sample, saved)
        name = f'{sample["position"]:04d}.json'
        path = output / 'prepared_inputs' / name
        if sha256_file(path) != cache['files'][name]:
            raise ValueError('Prepared context cache file checksum mismatch')
        payloads = read(path)
        base = build_context_input(saved, query, sample, 'baseline', token_len)
        if set(payloads) != set(VARIANTS) or payloads['baseline'] != base:
            raise ValueError('Cached original report, candidates or original evidence changed')
        snapshot = read(Path(base['source_snapshot']['manifest_file']))
        for variant in ('lexical', 'structural'):
            payload = payloads[variant]
            if {k: v for k, v in payload.items() if k not in changed} != {k: v for k, v in base.items() if k not in changed}:
                raise ValueError('Context cache altered shared original evidence or source identity')
            if set(payload['source_context']) != set(base['candidates']):
                raise ValueError('Context cache changed candidate membership')
            sections, total = [], 0
            for path in sorted(payload['source_context']):
                context = payload['source_context'][path]
                if context['file_path'] != path or context['requested_variant'] != variant:
                    raise ValueError('Cached context file or selection variant differs')
                source_file = Path(base['source_snapshot']['root']) / path
                raw = source_file.read_bytes()
                checksum = hashlib.sha256(raw).hexdigest()
                if checksum != snapshot['files'][path] or checksum != context['source_sha256']:
                    raise ValueError('Cached context pre-fix source checksum differs')
                encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
                lines = raw.decode(encoding).splitlines(keepends=True)
                for segment in context['segments']:
                    start, end = segment['start_line'], segment['end_line']
                    if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
                        raise ValueError('Cached source range is invalid')
                    text = ''.join(lines[start - 1:end])
                    if text != segment['text'] or hashlib.sha256(text.encode()).hexdigest() != segment['text_sha256']:
                        raise ValueError('Cached context is not an exact pre-fix source slice')
                rendered = render_segments(path, context['segments'])
                count = token_len(rendered) if rendered else 0
                if rendered != context['rendered_text'] or count != context['token_count'] or count > BUDGET:
                    raise ValueError('Cached context rendering or token budget differs')
                total += count
                if rendered:
                    sections.append(rendered)
            user = base['user_content'] + ('\n\nADDITIONAL SOURCE CONTEXT:\n' + '\n\n'.join(sections) if sections else '')
            if payload['context_token_count'] != total or payload['user_content'] != user or payload['user_content_sha256'] != digest(user):
                raise ValueError('Cached additional prompt text differs')
        prepared[sample['position']] = (source, saved, hashes, payloads)
    return manifest, prepared


def preparation_report(prepared):
    report = {'n': len(prepared), 'per_file_budget_tokens': BUDGET, 'variants': {},
              'lexical_structural_different_reports': 0, 'lexical_structural_different_files': 0}
    for variant in VARIANTS:
        contexts = [c for p in prepared.values() for c in p[3][variant]['source_context'].values()]
        report['variants'][variant] = {
            'files_with_additions': sum(bool(c['segments']) for c in contexts),
            'total_added_tokens': sum(c['token_count'] for c in contexts),
            'maximum_file_tokens': max((c['token_count'] for c in contexts), default=0),
            'fallback_files': sum(bool(c['fallback_reason']) for c in contexts),
        }
    for _, _, _, payloads in prepared.values():
        a, b = payloads['lexical']['source_context'], payloads['structural']['source_context']
        differences = sum(a[p]['rendered_text'] != b[p]['rendered_text'] for p in a)
        report['lexical_structural_different_reports'] += bool(differences)
        report['lexical_structural_different_files'] += differences
    return report


def run(selection, output, prepare_only=False):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(selection, output, prepare_only)


def run_locked(selection, output, prepare_only):
    started = time.monotonic()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(PROJECT / 'models/Qwen3-8B', local_files_only=True)
    @lru_cache(maxsize=20000)
    def token_len(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    if not prepare_only and (output / 'prepared_cache.json').exists():
        manifest, prepared = load_prepared_cache(selection, output, token_len)
    else:
        manifest, prepared = prepare(selection, token_len)
    samples = manifest['samples']
    prep = preparation_report(prepared)
    write_json(output / 'preparation_summary.json', prep)
    if prepare_only:
        folder = output / 'prepared_inputs'
        folder.mkdir(exist_ok=True)
        for position, (_, _, _, payloads) in prepared.items():
            write_json(folder / f'{position:04d}.json', payloads)
        freeze_prepared_cache(selection, output)
        print(json.dumps(prep), flush=True)
        return
    settings = {'generation_model': 'models/Qwen3-8B', 'inference_endpoint': 'http://127.0.0.1:8099',
                'server_manifest': 'models/BLAgent-Qwen3-8B-BF16/manifest.json', 'server_slots': 2,
                'max_context_tokens': 32768, 'max_new_tokens': 2048, 'max_calls': 2,
                'seed': 42, 'request_timeout': 600}
    generator = ServerGenerator(PROJECT, settings)
    code = [Path(__file__), PROJECT / 'scripts/experiments/pilot_c3_evidence.py',
            PROJECT / 'src/bug_localization/context_evidence.py',
            PROJECT / 'src/bug_localization/evidence_reranking.py',
            PROJECT / 'src/bug_localization/blagent_runtime.py',
            PROJECT / 'src/bug_localization/blagent_recovery.py']
    plan = {
        'experiment': 'C3 bounded local-source context selection', 'head': HEAD,
        'selection_sha256': sha256_file(selection), 'n': len(samples), 'variants': list(VARIANTS),
        'code_sha256': {str(p.relative_to(PROJECT)): sha256_file(p) for p in code},
        'generator_identity': generator.identity, 'prompt': BASELINE_PROMPT,
        'inference_variant': 'baseline', 'context_budget_per_file': BUDGET,
        'max_new_tokens': 2048, 'max_calls': 2,
        'primary_contrast': 'structural - lexical', 'secondary_contrast': 'structural - baseline',
        'primary_metric': 'MAP@10', 'multiplicity': 'two contrasts by six metrics; Holm family=12',
        'failure_policy': 'zero in full denominator; no ranking fallback or case replacement',
        'development_only': True, 'no_outcome_adaptive_changes': True,
        'shared': 'original report, C3 top10 and all original excerpts, ordinary joint prompt and output schema',
        'treatment': '256-token per-file cap on same-file pre-fix source additions; actual lengths reported',
        'source_sha256': {str(p): v[2] for p, v in prepared.items()},
        'prepared_cache_sha256': sha256_file(output / 'prepared_cache.json') if (output / 'prepared_cache.json').exists() else None,
        'prepared_input_sha256': {str(p): {v: digest(payload) for v, payload in entry[3].items()}
                                  for p, entry in prepared.items()},
        'decision': 'Exploratory evidence only; no default replacement based on this development cohort',
    }
    frozen_write(output / 'plan.json', plan)
    snapshot = output / 'implementation'
    snapshot.mkdir(exist_ok=True)
    for path in code:
        target = snapshot / path.name
        if target.exists() and sha256_file(target) != sha256_file(path):
            raise ValueError('Frozen code snapshot changed')
        if not target.exists():
            shutil.copyfile(path, target)
    for variant in VARIANTS:
        folder = output / variant
        folder.mkdir(exist_ok=True)
        frozen_write(folder / 'selection.json', manifest)
        frozen_write(folder / 'protocol.json', {**plan, 'method': variant})
    counts = [{'position': p, **{v: generator.count_tokens(build_messages(x, 'baseline'))
                                for v, x in entry[3].items()}} for p, entry in prepared.items()]
    frozen_write(output / 'input_token_audit.json', counts)
    if any(row[v] + 2048 > 32768 for row in counts for v in VARIANTS):
        raise ValueError('Frozen cohort exceeds context budget; no calls or silent truncation')
    write_json(output / 'server_properties.json', generator.server_properties)

    def work(sample, variant):
        begin = time.monotonic()
        folder = output / variant
        case = folder / 'cases' / f'{sample["position"]:04d}'
        case.mkdir(parents=True, exist_ok=True)
        source, saved, hashes, payloads = prepared[sample['position']]
        payload = payloads[variant]
        record_path = case / 'record.json'
        if record_path.exists():
            record = read(record_path)
            if any(record.get(k) != sample[k] for k in BASE.IDENTITY) or record['source_sha256'] != hashes:
                raise ValueError('Checkpoint sample/source changed')
            for key in ('result', 'inference_input', 'validation'):
                if key + '_file' in record and sha256_file(folder / record[key + '_file']) != record[key + '_sha256']:
                    raise ValueError('Checkpoint artifact changed: ' + key)
            return variant, record
        for path in case.glob('call_*.json'):
            cached = read(path)
            if cached.get('status') == 'running' or (
                    cached.get('status') == 'failed' and cached.get('error_type') != 'ValueError'):
                raise RuntimeError('Interrupted infrastructure call; explicit recovery required: ' + str(path))
        record = {**sample, 'source_sha256': hashes, 'missing_gold': source.get('missing_gold', [])}
        g = copy.copy(generator)
        g.start_case(case)
        try:
            value = infer(g, payload, 'baseline', case)
            ranking = resolve_ranking(value, payload)
            result = {'ranking': ranking, 'reasoning': value['reasoning'], 'structured_output': value,
                      'ground_truth': saved['ground_truth'],
                      'metrics': ranking_metrics(ranking, saved['ground_truth'], 10), 'model_calls': g.calls}
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
        record.update(model_calls=g.calls, elapsed_seconds=time.monotonic() - begin,
                      added_context_tokens=payload['context_token_count'])
        write_json(record_path, record)
        return variant, record

    records = {v: [] for v in VARIANTS}
    permutations = list(itertools.permutations(VARIANTS))
    jobs = [(s, v) for s in samples for v in permutations[int(digest(s['dataset_row'])[:8], 16) % 6]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(work, s, v) for s, v in jobs]
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
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run((PROJECT / args.selection).resolve(), (PROJECT / args.output).resolve(), args.prepare_only)
