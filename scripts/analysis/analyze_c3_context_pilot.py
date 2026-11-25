"""Audit the three-way C3 source-context development pilot without inference.

Structural minus lexical is the primary comparison; structural minus baseline
is secondary. Both use full-denominator failure-zero metrics. The same extra
token cap does not imply equal realized context length, which is also reported.
"""
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import io
from pathlib import Path, PurePosixPath
import sys
import tokenize
import zipfile

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
SPEC = importlib.util.spec_from_file_location('c3_context_common_audit',
    PROJECT / 'scripts/analysis/analyze_c3_evidence_pilot.py')
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)
from bug_localization.batch import write_json
from bug_localization.blagent_recovery import json_object
from bug_localization.bm25 import ranking_metrics
from bug_localization.embeddings import digest
from bug_localization.evidence_reranking import build_input, validate_response
from bug_localization.paired_statistics import analyze as paired_analyze, METRICS, values
from bug_localization.snapshot import sha256_file

read, require, summary = common.read, common.require, common.summary
IDENTITY, METRIC_NAMES = common.IDENTITY, common.METRIC_NAMES
VARIANTS = ('baseline', 'lexical', 'structural')
CONTRASTS = {'structural minus lexical': {'structural': 1, 'lexical': -1},
             'structural minus baseline': {'structural': 1, 'baseline': -1}}


def decode_source(raw):
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    return raw.decode(encoding)


@lru_cache(maxsize=256)
def verified_archive_members(archive_name, expected_sha256, expected_root, paths, stat_stamp):
    """Cache only within one analysis and pin filesystem identity in the key."""
    archive = Path(archive_name)
    require(sha256_file(archive) == expected_sha256, 'Source ZIP archive checksum mismatch')
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        require(names and len(names) == len(set(names)), 'Empty ZIP archive or duplicate member names')
        roots = {PurePosixPath(name).parts[0] for name in names}
        require(roots == {expected_root}, 'Source ZIP root does not identify the pre-fix repository commit')
        result = {}
        for path in paths:
            member = expected_root + '/' + path
            require(member in names, 'Candidate file is absent from original source ZIP')
            result[path] = hashlib.sha256(handle.read(member)).hexdigest()
    return result


def audit_source_context(payload, sample, variant, token_len, *, project=PROJECT):
    """Rebuild exact source slices, rendering, selection and token accounting."""
    from bug_localization.context_evidence import select_context
    require(variant in VARIANTS, 'Unknown source-context variant')
    snapshot = payload['source_snapshot']
    expected_root = project / 'repos' / sample['repository'] / sample['before_fix_sha'] / 'source'
    root = Path(snapshot['root']).resolve(strict=True)
    require(root == expected_root.resolve(strict=True), 'Source root is not the sample pre-fix snapshot')
    manifest_path = Path(snapshot['manifest_file'])
    require(manifest_path.resolve() == (expected_root.parent / 'snapshot.json').resolve(),
            'Source manifest is not beside the pinned snapshot')
    require(sha256_file(manifest_path) == snapshot['manifest_sha256'], 'Source snapshot manifest checksum mismatch')
    manifest = read(manifest_path)
    require(snapshot['repository'] == manifest['repository'] == sample['repository'], 'Source repository differs')
    require(snapshot['commit'] == manifest['commit'] == sample['before_fix_sha'], 'Source commit is not before-fix commit')
    contexts = payload['source_context']
    paths = payload['candidates']
    archive = Path(snapshot['archive_file']).resolve(strict=True)
    require(archive == (expected_root.parent / 'snapshot.zip').resolve(strict=True), 'Source ZIP is outside pinned snapshot directory')
    archive_root = sample['repository'].rsplit('/', 1)[-1] + '-' + sample['before_fix_sha']
    require(snapshot['archive_root'] == archive_root, 'Recorded ZIP root differs from pre-fix commit')
    require(snapshot['archive_sha256'] == manifest['archive_sha256'], 'ZIP metadata hashes differ')
    stat = archive.stat()
    archived = verified_archive_members(str(archive), manifest['archive_sha256'], archive_root,
                                        tuple(sorted(paths)), (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    budget = payload['context_budget_per_file']
    require(type(budget) is int and budget == 256, 'Source-context budget differs from fixed 256-token protocol')
    require(set(contexts) == (set() if variant == 'baseline' else set(paths)), 'Context file inventory differs from fixed head')
    count, files = 0, []
    for path in sorted(paths):
        relative = PurePosixPath(path)
        require(not relative.is_absolute() and all(p not in ('', '.', '..') for p in path.split('/')),
                'Unsafe source path')
        full = root.joinpath(*relative.parts).resolve(strict=True)
        require(full.is_relative_to(root) and full.is_file(), 'Source escapes pinned snapshot')
        raw = full.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        require(checksum == manifest['files'][path], 'Source file hash differs from snapshot manifest')
        require(archived[path] == checksum, 'Extracted source differs from original ZIP member bytes')
        if variant == 'baseline':
            continue
        context = contexts[path]
        require(context['file_path'] == path and context['source_sha256'] == checksum,
                'Extra context source identity differs')
        require(context['requested_variant'] == variant and context['budget_tokens'] == budget,
                'Extra context selection method/budget differs')
        lines = decode_source(raw).splitlines(keepends=True)
        seen_lines, rendered_parts = set(), []
        for segment in context['segments']:
            start, end = segment['start_line'], segment['end_line']
            require(type(start) is int and type(end) is int and 1 <= start <= end <= len(lines),
                    'Extra source segment has invalid repository line bounds')
            numbers = set(range(start, end + 1))
            require(not (seen_lines & numbers), 'Duplicate extra source line')
            seen_lines.update(numbers)
            text = ''.join(lines[start - 1:end])
            require(segment['text'] == text, 'Extra source text differs from pre-fix file lines')
            require(segment['text_sha256'] == hashlib.sha256(text.encode('utf-8')).hexdigest(),
                    'Extra source text checksum differs')
            rendered_parts.append(f'[source lines {start}-{end}]\n{text}')
        rendered = ('Additional source context: ' + path + '\n' + '\n'.join(rendered_parts)) if rendered_parts else ''
        require(context['rendered_text'] == rendered, 'Extra context rendering differs from source slices')
        actual_tokens = token_len(rendered) if rendered else 0
        require(type(context['token_count']) is int and context['token_count'] == actual_tokens
                and 0 <= actual_tokens <= budget, 'Extra context token budget/accounting differs')
        original_texts = [r['text'] for r in payload['evidence'][path]]
        replay = select_context(root, path, payload['query'], original_texts, token_len,
                                variant=variant, budget_tokens=budget, expected_sha256=checksum)
        require(replay == context, 'Saved context differs from deterministic frozen selection')
        count += actual_tokens
        files.append({'file_path': path, 'source_sha256': checksum, 'token_count': actual_tokens,
                      'segments': len(context['segments']), 'source_lines': len(seen_lines),
                      'effective_variant': context['effective_variant'], 'fallback_reason': context['fallback_reason'],
                      'reasons': sorted({reason for s in context['segments'] for reason in s['reason']})})
    require(payload['context_token_count'] == count, 'Total added context token count differs')
    return {'snapshot_manifest': str(manifest_path), 'snapshot_manifest_sha256': snapshot['manifest_sha256'],
            'archive_file': str(archive), 'archive_sha256': manifest['archive_sha256'], 'archive_root': archive_root,
            'repository': sample['repository'], 'before_fix_sha': sample['before_fix_sha'],
            'context_tokens': count, 'files': files}


def audit_input(payload, saved, report, sample, variant, token_len, *, project=PROJECT):
    head = len(payload['candidates'])
    base = build_input(saved, report['query'], head)
    # Common independently reconstructs original blocks and their numbered text.
    common.check_input({**payload, 'user_content': base['user_content'],
                        'user_content_sha256': base['user_content_sha256']}, saved, report, head)
    require(payload['base_user_content_sha256'] == base['user_content_sha256'], 'Original code/report rendering changed')
    source = audit_source_context(payload, sample, variant, token_len, project=project)
    user = base['user_content']
    if variant != 'baseline':
        sections = [payload['source_context'][p]['rendered_text']
                    for p in sorted(payload['source_context']) if payload['source_context'][p]['rendered_text']]
        if sections:
            user += '\n\nADDITIONAL SOURCE CONTEXT:\n' + '\n\n'.join(sections)
    require(payload['user_content'] == user and payload['user_content_sha256'] == digest(user),
            'Actual input contains altered/unapproved additional source context')
    return source


def check_frozen_code(run, protocol):
    for relative, checksum in protocol['code_sha256'].items():
        snapshot = run.parent / 'implementation' / Path(relative).name
        require(sha256_file(snapshot) == checksum, 'Frozen implementation snapshot differs: ' + relative)
        # Replayed selector/validator must match the frozen implementation.
        if Path(relative).name in ('context_evidence.py', 'evidence_reranking.py'):
            require(sha256_file(PROJECT / relative) == checksum, 'Live selector/validator differs from frozen code: ' + relative)


def check_prepared_cache(run, protocol, selection):
    expected = protocol.get('prepared_cache_sha256')
    if expected is None:
        return None
    path = run.parent / 'prepared_cache.json'
    require(sha256_file(path) == expected, 'Prepared source cache manifest checksum differs')
    cache = read(path)
    require(cache['selection_sha256'] == protocol['selection_sha256'] and cache['head'] == protocol['head']
            and cache['budget'] == protocol['context_budget_per_file'], 'Prepared cache cohort or budgets differ')
    require(cache['selector_sha256'] == protocol['code_sha256']['src/bug_localization/context_evidence.py']
            and cache['original_builder_sha256'] == protocol['code_sha256']['src/bug_localization/evidence_reranking.py'],
            'Prepared cache implementation differs')
    for name, checksum in cache['tokenizer_sha256'].items():
        require(checksum == protocol['generator_identity']['source_files_sha256'][name], 'Prepared cache tokenizer differs')
    runner = run.parent / 'implementation/pilot_c3_context.py'
    source = runner.read_text(encoding='utf-8')
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'build_context_input']
    require(len(nodes) == 1, 'Frozen context renderer is ambiguous')
    node = nodes[0]
    renderer = ''.join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    require(digest(renderer) == cache['renderer_sha256'], 'Prepared cache renderer differs from frozen function')
    require(cache['token_counting'] == 'AutoTokenizer.encode; add_special_tokens=False', 'Prepared cache token counting differs')
    wanted = {f'{s["position"]:04d}.json': s['position'] for s in selection['samples']}
    require(set(cache['files']) == set(wanted), 'Prepared cache file inventory differs from cohort')
    for name, position in wanted.items():
        payload_path = run.parent / 'prepared_inputs' / name
        require(sha256_file(payload_path) == cache['files'][name], 'Prepared input cache file checksum differs')
        payloads = read(payload_path)
        require(set(payloads) == set(VARIANTS), 'Prepared cache variant inventory differs')
        for variant, payload in payloads.items():
            require(digest(payload) == protocol['prepared_input_sha256'][str(position)][variant],
                    'Prepared cache and frozen inference payload hashes differ')
    return {'manifest': str(path), 'sha256': expected, 'files': len(wanted),
            'scope': 'Source preparation only; actual inference requests/outputs are audited separately'}


def audit_run(run, selection, selection_path, token_len, *, project=PROJECT):
    protocol = read(run / 'protocol.json')
    variant = protocol['method']
    require(variant in VARIANTS and protocol.get('inference_variant') == 'baseline', 'Invalid context inference variant')
    require(read(run / 'selection.json') == selection, 'Run selection differs')
    require(protocol['selection_sha256'] == sha256_file(selection_path), 'Selection checksum differs')
    check_frozen_code(run, protocol)
    cache_audit = check_prepared_cache(run, protocol, selection)
    samples = {s['position']: s for s in selection['samples']}
    records, inputs, originals, source_audits, coverage, calls_by_position = {}, {}, {}, {}, {}, {}
    for path in sorted((run / 'cases').glob('*/record.json')):
        record = read(path)
        position = record['position']
        require(position in samples and position not in records, 'Unknown or duplicate terminal context case')
        sample = samples[position]
        common.check_identity(record, sample)
        require(record['status'] in ('ok', 'failed'), 'Nonterminal case record')
        for name in ('inference_input', 'validation'):
            if name + '_file' in record:
                require(sha256_file(run / record[name + '_file']) == record[name + '_sha256'], 'Artifact checksum differs: ' + name)
        saved, report = common.source_data(record)
        common.check_identity(saved, sample)
        require(report['source_commit'] == sample['before_fix_sha'], 'Original report source commit differs')
        originals[position] = saved
        coverage[position] = common.source_inventory(saved)
        require(record['missing_gold'] == coverage[position]['missing_gold'], 'Repository coverage metadata differs')
        input_path = path.parent / 'inference_input.json'
        payload = read(input_path) if input_path.exists() else None
        if payload is not None:
            require(len(payload['candidates']) == min(protocol['head'], len(saved['ranking'])), 'Candidate head budget differs')
            if 'prepared_input_sha256' in protocol:
                prepared = {k: v for k, v in payload.items() if k != 'prompt_tokens'}
                require(digest(prepared) == protocol['prepared_input_sha256'][str(position)][variant],
                        'Model input differs from pre-inference frozen payload')
            if 'context_budget_per_file' in protocol:
                require(payload['context_budget_per_file'] == protocol['context_budget_per_file'],
                        'Context cap differs from protocol')
            source_audits[position] = audit_input(payload, saved, report, sample, variant, token_len, project=project)
            if 'added_context_tokens' in record:
                require(record['added_context_tokens'] == payload['context_token_count'], 'Recorded extra context tokens differ')
            inputs[position] = payload
        calls = common.audit_calls(path.parent, {**protocol, 'method': 'baseline'}, payload)
        calls_by_position[position] = calls
        if hasattr(token_len, 'messages'):
            for call in calls:
                require(call['input_tokens'] == token_len.messages(call['request']['messages']),
                        'Actual request token count differs from frozen tokenizer/chat template')
        if calls and payload is not None:
            if 'prompt_tokens' in payload:
                require(calls[0]['input_tokens'] == payload['prompt_tokens'], 'Initial request/payload prompt token counts differ')
            source_audits[position]['initial_prompt_tokens'] = calls[0].get('input_tokens')
        require(record['model_calls'] == len(calls), 'Recorded model call count differs')
        if record['status'] == 'ok':
            require(payload is not None and calls and calls[-1]['status'] == 'ok', 'Successful case lacks final inference evidence')
            result_path = run / record['result_file']
            require(sha256_file(result_path) == record['result_sha256'], 'Result checksum differs')
            result = read(result_path)
            response = json_object(calls[-1]['response'])
            require(response == result['structured_output'], 'Saved response differs from final actual output')
            require(not validate_response(response, payload, 'baseline'), 'Successful output is not a valid ordinary ranking')
            common.check_response(response, payload, result)
            common.check_inventory(result['ranking'], common.ranking_paths(saved), protocol['head'])
            require(result['ground_truth'] == saved['ground_truth'], 'Ground truth changed')
            metrics = ranking_metrics(result['ranking'], saved['ground_truth'], 10)
            require(metrics == result['metrics'] == record['metrics'], 'Recomputed metrics differ')
            require(result['model_calls'] == len(calls), 'Result call count differs')
        else:
            require(record.get('error'), 'Failure lacks explicit diagnostic')
        records[position] = record
    ordered = [records[s['position']] for s in selection['samples'] if s['position'] in records]
    return {'protocol': protocol, 'records': ordered, 'inputs': inputs, 'originals': originals,
            'coverage': coverage, 'source_audits': source_audits, 'calls': calls_by_position,
            'prepared_cache_audit': cache_audit,
            'summary': summary(ordered, len(samples)), 'cost': common.cost_report(ordered, calls_by_position)}


def token_counter(protocol):
    from transformers import AutoTokenizer
    model = PROJECT / protocol['generator_identity']['settings']['generation_model']
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    # Original model/tokenizer files are pinned in the generator identity.
    for name, checksum in protocol['generator_identity']['source_files_sha256'].items():
        if 'tokenizer' in name:
            require(sha256_file(model / name) == checksum, 'Tokenizer source hash differs')
    @lru_cache(maxsize=20000)
    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    def messages(items):
        rendered = tokenizer.apply_chat_template(items, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        return count(rendered)
    count.messages = messages
    return count


def context_budget_report(runs):
    report = {'note': 'All augmented files share a 256-token rendered-context cap. Actual selected tokens may differ; '
                       'wrapper tokens and full model input tokens are separately accounted in model-call costs.', 'methods': {}}
    for variant, run in runs.items():
        cases = run['source_audits']
        tokens = [r['context_tokens'] for r in cases.values()]
        prompt_tokens = [r['initial_prompt_tokens'] for r in cases.values() if r.get('initial_prompt_tokens') is not None]
        files = [f for r in cases.values() for f in r['files']]
        fallback = Counter(f['fallback_reason'] for f in files if f['fallback_reason'])
        reasons = Counter(reason for f in files for reason in f['reasons'])
        report['methods'][variant] = {
            'audited_cases': len(cases), 'total_extra_tokens': sum(tokens),
            'mean_extra_tokens': float(np.mean(tokens)) if tokens else None,
            'initial_model_prompt_tokens_sum': sum(prompt_tokens),
            'initial_model_prompt_tokens_mean': float(np.mean(prompt_tokens)) if prompt_tokens else None,
            'min_extra_tokens': min(tokens) if tokens else None, 'max_extra_tokens': max(tokens) if tokens else None,
            'files_with_context': sum(f['token_count'] > 0 for f in files),
            'files_without_context': sum(f['token_count'] == 0 for f in files),
            'fallback_counts': dict(fallback), 'selection_reason_file_counts': dict(reasons),
            'cases': {str(k): v for k, v in sorted(cases.items())}}
    shared = sorted(set(runs['lexical']['source_audits']) & set(runs['structural']['source_audits']))
    deltas = [runs['structural']['source_audits'][p]['context_tokens'] - runs['lexical']['source_audits'][p]['context_tokens']
              for p in shared]
    report['structural_minus_lexical_extra_tokens'] = {
        'paired_cases': len(shared), 'mean_delta': float(np.mean(deltas)) if deltas else None,
        'mean_absolute_delta': float(np.mean(np.abs(deltas))) if deltas else None,
        'equal_token_cases': sum(d == 0 for d in deltas)}
    prompt_deltas = {}
    for variant in ('lexical', 'structural'):
        positions = sorted(set(runs['baseline']['source_audits']) & set(runs[variant]['source_audits']))
        pairs = [(p, runs[variant]['source_audits'][p].get('initial_prompt_tokens'),
                  runs['baseline']['source_audits'][p].get('initial_prompt_tokens')) for p in positions]
        changes = {str(p): added - base for p, added, base in pairs if added is not None and base is not None}
        prompt_deltas[variant + ' minus baseline'] = {
            'paired_cases': len(changes), 'total_actual_prompt_token_delta': sum(changes.values()),
            'mean_actual_prompt_token_delta': float(np.mean(list(changes.values()))) if changes else None,
            'per_case_delta': changes}
    report['actual_initial_prompt_token_deltas'] = prompt_deltas
    return report


def oracle_head_ranking(original, gold, head):
    """Evaluation-only upper bound: stable gold-first ordering within the head."""
    require(type(head) is int and head > 0, 'Invalid oracle head size')
    require(len(original) == len(set(original)), 'Duplicate oracle candidate')
    labels = set(gold)
    require(labels, 'Oracle requires nonempty evaluation labels')
    fixed = original[:head]
    ranking = [p for p in fixed if p in labels] + [p for p in fixed if p not in labels] + original[head:]
    common.check_inventory(ranking, original, head)
    return ranking


def candidate_head_oracle(runs, samples):
    """Post-hoc evaluation only; never passed to inference or used to tune it."""
    total = len(samples)
    require(all(not run['summary']['pending_samples'] for run in runs.values()), 'Oracle is suppressed until every method completes')
    baseline = runs['baseline']
    head = baseline['protocol']['head']
    indexes = {v: {r['position']: r for r in run['records']} for v, run in runs.items()}
    records, cases = [], []
    for sample in samples:
        position = sample['position']
        saved = baseline['originals'][position]
        original, gold = common.ranking_paths(saved), saved['ground_truth']
        ranking = oracle_head_ranking(original, gold, head)
        metrics = ranking_metrics(ranking, gold, 10)
        records.append({**sample, 'status': 'ok', 'metrics': metrics,
                        'missing_gold': baseline['coverage'][position]['missing_gold']})
        gaps = {}
        for variant, index in indexes.items():
            record = index[position]
            gaps[variant] = {m: metrics[m] - (record['metrics'][m] if record['status'] == 'ok' else 0.) for m in METRICS}
            require(all(d >= -1e-12 for d in gaps[variant].values()), 'Observed score exceeds fixed-head oracle')
        cases.append({**sample, 'oracle_head': ranking[:head], 'tail_unchanged': True,
                      'gold_total': len(set(gold)), 'gold_in_fixed_head': len(set(gold) & set(original[:head])),
                      'gold_outside_fixed_head': sorted(set(gold) - set(original[:head])),
                      'metrics': metrics, 'oracle_minus_method': gaps})
    upper = summary(records, total)
    gaps = {v: {m: upper['metrics'][m] - run['summary']['metrics'][m] for m in METRIC_NAMES} for v, run in runs.items()}
    return {'label': 'candidate_head_oracle', 'posthoc': True, 'evaluation_only': True, 'head': head,
            'rule': 'Use repair labels only after all methods finish to place gold files already in the fixed head first. '
                    'Preserve gold/non-gold relative order within each class and preserve the original tail exactly. '
                    'Never import a file from rank head+1 or later. All gold labels remain in AP/recall denominators.',
            'interpretation': 'Theoretical attainable bound under this fixed candidate head, not a method, new baseline test, '
                              'or evidence of improvement. Gaps include terminal inference failures counted as zero. '
                              'Not used for prompts, parameter selection, or the twelve-test comparison family.',
            'summary': upper, 'oracle_minus_method': gaps, 'cases': cases}


def analyze(baseline, lexical, structural, selection_path, output, *, repetitions=5000,
            token_len=None, project=PROJECT):
    verified_archive_members.cache_clear()
    selection = read(selection_path)
    samples = selection['samples']
    require(samples and len({s['position'] for s in samples}) == len(samples), 'Empty or duplicate selection')
    folders = dict(zip(VARIANTS, (baseline, lexical, structural)))
    if token_len is None:
        token_len = token_counter(read(baseline / 'protocol.json'))
    runs = {v: audit_run(folder, selection, selection_path, token_len, project=project) for v, folder in folders.items()}
    baseline_protocol = runs['baseline']['protocol']
    for run in runs.values():
        for field in ('head', 'generator_identity', 'prompt', 'code_sha256'):
            require(run['protocol'][field] == baseline_protocol[field], 'Paired protocol differs: ' + field)
    paired_positions = sorted(set.intersection(*(set(r['inputs']) for r in runs.values())))
    for position in paired_positions:
        original = runs['baseline']['inputs'][position]
        for variant in ('lexical', 'structural'):
            other = runs[variant]['inputs'][position]
            for field in ('query', 'candidates', 'evidence', 'catalog', 'original_ranking', 'tail',
                          'base_user_content_sha256', 'context_budget_per_file', 'source_snapshot'):
                require(other[field] == original[field], 'Common input changed between context variants: ' + field)
            require(runs[variant]['originals'][position]['ground_truth'] == runs['baseline']['originals'][position]['ground_truth'],
                    'Paired labels differ')
    complete = all(not r['summary']['pending_samples'] for r in runs.values())
    methods = {v: r['records'] for v, r in runs.items()}
    paired = paired_analyze(methods, CONTRASTS, repetitions=repetitions) if complete else None
    clusters = None
    if complete:
        repositories = [s['repository'] for s in samples]
        clusters = {}
        for name, weights in CONTRASTS.items():
            clusters[name] = {m: common.cluster_inference(
                sum(w * values(methods[v], m) for v, w in weights.items()), repositories,
                repetitions=repetitions) for m in METRICS}
        common.holm([stats for contrast in clusters.values() for stats in contrast.values()],
                    'p_two_sided_project_randomization', 'p_holm_twelve_tests')
    projects = {p: {v: summary([r for r in rows if r['repository'] == p], sum(s['repository'] == p for s in samples))
                    for v, rows in methods.items()} for p in sorted({s['repository'] for s in samples})}
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'analysis_script_sha256': sha256_file(Path(__file__)),
        'common_analysis_sha256': sha256_file(PROJECT / 'scripts/analysis/analyze_c3_evidence_pilot.py'),
        'status': 'complete' if complete else 'pending', 'selected_samples': len(samples),
        'selection_sha256': sha256_file(selection_path),
        'note': 'Previously observed development reports. No independent-test or causal-proof claim. '
                'Primary contrast structural minus lexical; secondary structural minus baseline. '
                'Both augmented methods have the same context cap, not necessarily the same realized token count.',
        'audit': {'status': 'passed' if complete else 'pending', 'three_way_input_cases': len(paired_positions),
                  'original_code_blocks_unchanged': True, 'source_lines_and_snapshot_hashes_checked': True,
                  'deterministic_context_selection_replayed': True, 'failure_policy': 'zero in complete cohort; pending metrics suppressed'},
        'runs': {v: {'path': str(folders[v]), 'protocol_sha256': sha256_file(folders[v] / 'protocol.json'),
                     'summary': run['summary'], 'cost': run['cost'], 'prepared_cache_audit': run['prepared_cache_audit']}
                 for v, run in runs.items()},
        'project_summaries': projects, 'paired_report_statistics': paired, 'project_cluster_statistics': clusters,
        'context_budget_audit': context_budget_report(runs),
        'candidate_head_oracle': candidate_head_oracle(runs, samples) if complete else None,
        'coverage_audit': {v: {str(p): row for p, row in run['coverage'].items()} for v, run in runs.items()},
        'multiplicity_note': 'Two contrasts times six metrics form a twelve-test Holm family within each separately '
                             'labeled inference scheme. Prior adaptive experiments are not included; results remain exploratory.'}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'analysis.json', report)
    with (output / 'comparison.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['method', 'successful', 'failed', 'pending', *METRIC_NAMES])
        writer.writeheader()
        for variant, run in runs.items():
            result = run['summary']
            writer.writerow({'method': variant, 'successful': result['successful_samples'], 'failed': result['failed_samples'],
                             'pending': result['pending_samples'], **(result['metrics'] or {})})
    indexes = {v: {r['position']: r for r in rows} for v, rows in methods.items()}
    cases = []
    for sample in samples:
        row = dict(sample)
        for variant, index in indexes.items():
            record = index.get(sample['position'])
            row[variant + '_status'] = record['status'] if record else 'pending'
            row[variant + '_ap'] = (record['metrics']['average_precision_at_k'] if record['status'] == 'ok' else 0.) if record else None
        if complete:
            row['structural_minus_lexical_ap'] = row['structural_ap'] - row['lexical_ap']
            row['structural_minus_baseline_ap'] = row['structural_ap'] - row['baseline_ap']
        cases.append(row)
    write_json(output / 'paired_cases.json', cases)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*VARIANTS, 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=5000)
    args = parser.parse_args()
    require(2000 <= args.repetitions <= 50000, 'Use 2000 to 50000 repetitions')
    paths = [(PROJECT / getattr(args, name)).resolve() for name in (*VARIANTS, 'selection', 'output')]
    result = analyze(*paths, repetitions=args.repetitions)
    import json
    print(json.dumps({'status': result['status'], 'selected_samples': result['selected_samples'],
                      'audit': result['audit']}, indent=2))


if __name__ == '__main__':
    main()
