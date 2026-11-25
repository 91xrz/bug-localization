"""Independently audit the paired C3 evidence pilot without calling a model.

The cohort is previously observed development data. Failures count as zero,
pending cases suppress cohort estimates, and costs include unsuccessful calls.
Project-cluster inference is reported separately from within-project inference.
"""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import aggregate_results, write_json
from bug_localization.bm25 import ranking_metrics
from bug_localization.embeddings import digest
from bug_localization.paired_statistics import analyze as paired_analyze, METRICS, values
from bug_localization.snapshot import sha256_file

IDENTITY = ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha',
            'query_sha256', 'labels_sha256')
METRIC_NAMES = ('MAP@10', 'MRR@10', 'Top-1', 'Top-5', 'Top-10', 'MeanRecall@10')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ranking_paths(value):
    return [v['file_path'] if isinstance(v, dict) else v for v in value['ranking']]


def check_identity(row, sample):
    for field in IDENTITY:
        require(row.get(field) == sample[field], 'Sample identity differs: ' + field)


def check_inventory(ranking, original, head):
    require(len(ranking) == len(original) and len(set(ranking)) == len(ranking)
            and set(ranking) == set(original), 'Candidate inventory changed')
    require(set(ranking[:head]) == set(original[:head]), 'Head is not a candidate permutation')
    require(ranking[head:] == original[head:], 'Untouched tail changed')


def summary(records, total):
    result = aggregate_results(records, 10, total)
    result.pop('metrics_successful_only', None)
    result['failure_policy'] = 'Failures count as zero in the complete cohort; pending cohort metrics are suppressed'
    if not result['pending_samples'] and result['failed_samples']:
        result['status'] = 'completed_with_failures'
    return result


def source_data(record):
    """Identify and hash actual source artifacts, not a mutable summary file."""
    saved, report = None, None
    require(bool(record.get('source_sha256')), 'Missing frozen source hashes')
    for name, checksum in record['source_sha256'].items():
        path = Path(name)
        require(sha256_file(path) == checksum, 'Source checksum mismatch: ' + name)
        value = read(path)
        if isinstance(value, dict) and {'ranking', 'scored_chunks', 'ground_truth'} <= value.keys():
            require(saved is None, 'Ambiguous C3 source')
            saved = value
        elif isinstance(value, dict) and 'query' in value:
            require(report is None, 'Ambiguous full-report source')
            report = value
    require(saved is not None and report is not None, 'Missing C3 ranking or original report source')
    return saved, report


def source_inventory(saved):
    """Resolve repository availability from C3's pinned original inventory.

    C3 ranking/candidates contains the reranking pool, often exactly 50 files.
    It cannot establish whether a gold file existed in the source repository.
    """
    sources = saved.get('source_files_sha256', {})
    paths = [Path(name) for name in sources if Path(name).name == 'candidates.json']
    require(len(paths) == 1, 'C3 source must pin exactly one complete candidates.json inventory')
    path = paths[0]
    require(sha256_file(path) == sources[str(path)], 'Repository candidate inventory checksum mismatch')
    inventory = read(path)
    require(isinstance(inventory, dict) and isinstance(inventory.get('files'), list), 'Invalid complete candidate inventory')
    files = [r['file_path'] for r in inventory['files']]
    require(all(isinstance(p, str) and p for p in files) and len(files) == len(set(files)),
            'Invalid or duplicate repository inventory paths')
    available = set(files)
    require(set(ranking_paths(saved)) <= available, 'C3 ranking contains a file outside the complete inventory')
    reference_audit = None
    if saved.get('reference_file'):
        reference_path = Path(saved['reference_file'])
        require(sha256_file(reference_path) == saved['reference_sha256'], 'C3 reference checksum mismatch')
        reference = read(reference_path)
        require(reference['source_files_sha256'] == sources, 'C3/reference original source hashes differ')
        require(reference['ground_truth'] == saved['ground_truth'], 'C3/reference labels differ')
        for field in IDENTITY:
            if field in saved:
                require(reference.get(field) == saved[field], 'C3/reference sample identity differs: ' + field)
        reference_audit = {'path': str(reference_path), 'sha256': saved['reference_sha256']}
    missing = sorted(set(saved['ground_truth']) - available)
    return {'source_inventory_file': str(path), 'source_inventory_sha256': sources[str(path)],
            'reference': reference_audit, 'available_file_count': len(available),
            'c3_ranking_file_count': len(saved['ranking']), 'missing_gold': missing,
            'gold_available_but_outside_c3_ranking': sorted(
                (set(saved['ground_truth']) & available) - set(ranking_paths(saved))),
            'definition': 'missing_gold = sorted(gold minus complete pinned original repository inventory); '
                          'absence from the C3 Top50 pool is a retrieval miss, not a repository coverage gap'}


def check_input(observed, saved, report, head):
    original = ranking_paths(saved)
    paths = original[:head]
    expected = {p: [{'chunk_id': c['chunk_id'], 'text': c['embedding_text']}
                    for c in saved['scored_chunks'] if c['file_path'] == p] for p in paths}
    require(observed['query'] == report['query'], 'Original report changed')
    require(observed['candidates'] == paths and observed['evidence'] == expected,
            'Candidates or raw C3 evidence changed')
    require(all(expected.values()), 'Candidate without code evidence')
    require(observed['original_ranking'] == original and observed['tail'] == original[head:],
            'Saved original ranking/tail differs')
    # Reconstruct the numbered view from the raw source; hashes alone could
    # otherwise hide extra labels or altered code inside the rendered prompt.
    files, chunks, sections = {}, {}, []
    for number, path in enumerate(sorted(paths), 1):
        file_id = f'F{number:02d}'
        files[file_id] = path
        section = [f'FILE {file_id}: {path}']
        for chunk in expected[path]:
            chunk_id = f'C{len(chunks) + 1:03d}'
            lines = chunk['text'].splitlines()
            chunks[chunk_id] = {'file_id': file_id, **chunk, 'line_count': len(lines)}
            section.append(f'EXCERPT {chunk_id} (local lines 1-{len(lines)}):\n' +
                           '\n'.join(f'{i}: {line}' for i, line in enumerate(lines, 1)))
        sections.append('\n\n'.join(section))
    user = 'BUG REPORT:\n' + report['query'] + '\n\nCANDIDATE EVIDENCE:\n' + '\n\n'.join(sections)
    require(observed['catalog'] == {'files': files, 'chunks': chunks}, 'Numbered evidence catalog differs')
    require(observed['user_content'] == user and observed['user_content_sha256'] == digest(user),
            'Rendered model input differs from allowed raw evidence')


def check_response(response, observed, result):
    """Check the ranking and citations separately from the production validator."""
    files = observed['catalog']['files']
    ranked = response['ranked_ids']
    require(isinstance(ranked, list) and len(ranked) == len(files)
            and len(set(ranked)) == len(files) and set(ranked) == set(files), 'Invalid actual ranked IDs')
    require([files[k] for k in ranked] + observed['tail'] == result['ranking'],
            'Saved ranking does not follow actual model response')
    require(response['reasoning'] == result['reasoning'], 'Saved reasoning differs from response')
    if 'assessments' not in response:
        return
    constraints = response['constraints']
    require(1 <= len(constraints) <= 3, 'Invalid report constraint count')
    ids = [c['id'] for c in constraints]
    require(len(set(ids)) == len(ids) and set(ids) <= {'R1', 'R2', 'R3'}, 'Invalid report constraint IDs')
    for c in constraints:
        require(c['kind'] in ('trigger', 'observed', 'expected', 'anchor'), 'Invalid constraint kind')
        require(isinstance(c['quote'], str) and c['quote'].strip() and len(c['quote']) <= 200
                and c['quote'] in observed['query'], 'Constraint quote is not actual report text')
    assessments = response['assessments']
    assessed = [a['file_id'] for a in assessments]
    require(len(assessed) == len(files) and len(set(assessed)) == len(files)
            and set(assessed) == set(files), 'Missing or duplicate file assessment')
    for row in assessments:
        require(row['state'] in ('support', 'contradiction', 'unknown'), 'Invalid evidence state')
        require(row['reason'].strip(), 'Empty file assessment')
        require(len(row['constraint_ids']) <= 2 and len(set(row['constraint_ids'])) == len(row['constraint_ids'])
                and set(row['constraint_ids']) <= set(ids), 'Undefined or repeated constraint reference')
        require(len(row['evidence']) <= 2, 'Too many code references')
        if row['state'] != 'unknown':
            require(row['constraint_ids'] and row['evidence'], 'Support/contradiction lacks evidence')
        for reference in row['evidence']:
            require(reference['chunk_id'] in observed['catalog']['chunks'], 'Invented code reference')
            chunk = observed['catalog']['chunks'][reference['chunk_id']]
            require(chunk['file_id'] == row['file_id'], 'Code reference belongs to another file')
            require(type(reference['start']) is int and type(reference['end']) is int
                    and 1 <= reference['start'] <= reference['end'] <= chunk['line_count'],
                    'Invalid excerpt-local citation range')


def audit_calls(case, protocol, observed):
    calls = [read(p) for p in sorted(case.glob('call_*.json'))]
    settings = protocol['generator_identity']['settings']
    require(len(calls) <= settings['max_calls'], 'Inference call budget exceeded')
    variant = protocol.get('method')
    validations = []
    previous_errors = None
    for number, call in enumerate(calls):
        request = call['request']
        require(digest(request) == call['request_sha256'], 'Model request checksum mismatch')
        require(request['identity'] == digest(protocol['generator_identity']), 'Generator identity changed')
        require(request['max_new_tokens'] == settings['max_new_tokens'], 'Output budget changed')
        require(request['temperature'] == 0, 'Sampling temperature changed')
        require(call['status'] in ('ok', 'failed'), 'Model call still pending')
        messages = request['messages']
        require(messages[0] == {'role': 'system', 'content': protocol['prompt']}, 'System prompt changed')
        require(len(messages) >= 2 and messages[1]['role'] == 'user', 'Missing user input')
        if number == 0 and observed is not None:
            require(digest(messages[1]['content']) == observed['user_content_sha256'],
                    'Actual initial user content differs from frozen input')
            if 'user_content' in observed:
                require(messages[1]['content'] == observed['user_content'], 'User content text/hash differ')
        if variant in ('baseline', 'evidence') and observed is not None:
            from bug_localization.evidence_reranking import response_schema, validate_response
            from bug_localization.blagent_recovery import json_object
            require(request['response_schema'] == response_schema(observed, variant), 'Response schema changed')
            require(request['stage'] == 'C3_' + variant, 'Request method changed')
            require(len(messages) == 2, 'Unexpected model messages')
            if number:
                require(previous_errors, 'Retry was not justified by mechanical validation failure')
                expected = observed['user_content'] + (
                    '\n\nOUTPUT VALIDATION FEEDBACK (not code or bug evidence):\n' +
                    '\n'.join(previous_errors[:8]) +
                    '\nGenerate a complete corrected JSON response. Use only supplied IDs, '
                    'exact report quotes and excerpt-local line ranges. Each file appears once.')
                require(messages[1]['content'] == expected, 'Retry contains extra evidence or altered feedback')
            if call['status'] == 'ok':
                try:
                    errors = validate_response(json_object(call['response']), observed, variant)
                except (ValueError, TypeError) as error:
                    errors = [str(error)]
                validations.append({'attempt': number + 1, 'errors': errors})
                previous_errors = errors
        if call['status'] == 'ok':
            require('response' in call and call.get('finished'), 'Successful call lacks final response')
    if variant in ('baseline', 'evidence') and validations:
        require(read(case / 'validation.json') == validations, 'Saved mechanical validation differs from actual responses')
    return calls


def cost_report(records, calls_by_position):
    calls = [call for rows in calls_by_position.values() for call in rows]
    durations = [r['elapsed_seconds'] for r in records if 'elapsed_seconds' in r]
    counts = [len(rows) for rows in calls_by_position.values()]
    return {
        'scope': 'Incremental reranking over cached dual retrieval, not end-to-end latency; failures included',
        'model_calls': len(calls), 'failed_model_calls': sum(c['status'] != 'ok' for c in calls),
        'input_tokens': sum(c.get('input_tokens', 0) for c in calls),
        'output_tokens': sum(c.get('output_tokens', 0) for c in calls),
        'calls_missing_input_tokens': sum('input_tokens' not in c for c in calls),
        'calls_missing_output_tokens': sum('output_tokens' not in c for c in calls),
        'maximum_input_tokens': max((c.get('input_tokens', 0) for c in calls), default=0),
        'summed_call_seconds': sum(c.get('elapsed_seconds', 0) for c in calls),
        'summed_case_seconds': sum(durations),
        'case_mean_seconds': float(np.mean(durations)) if durations else None,
        'case_p50_seconds': float(np.quantile(durations, .5)) if durations else None,
        'case_p95_seconds': float(np.quantile(durations, .95)) if durations else None,
        'cases_with_multiple_calls': sum(n > 1 for n in counts),
        'case_call_counts': {str(k): len(v) for k, v in calls_by_position.items()},
    }


def audit_run(run, selection, selection_path):
    protocol = read(run / 'protocol.json')
    require(read(run / 'selection.json') == selection, 'Run selection changed')
    require(protocol['selection_sha256'] == sha256_file(selection_path), 'Selection checksum mismatch')
    for relative, checksum in protocol.get('code_sha256', {}).items():
        snapshot = run.parent / 'implementation' / Path(relative).name
        require(sha256_file(snapshot) == checksum, 'Frozen implementation snapshot checksum mismatch')
        if Path(relative).name == 'evidence_reranking.py':
            require(sha256_file(PROJECT / relative) == checksum,
                    'Live response validator differs from frozen implementation; restore/import exact snapshot before audit')
    head = protocol['head']
    samples = {s['position']: s for s in selection['samples']}
    paths = sorted((run / 'cases').glob('*/record.json'))
    records, inputs, originals, calls_by_position, results, coverage = {}, {}, {}, {}, {}, {}
    for path in paths:
        record = read(path)
        position = record['position']
        require(position in samples and position not in records, 'Unknown or duplicate terminal case')
        check_identity(record, samples[position])
        require(record['status'] in ('ok', 'failed'), 'Nonterminal case record')
        for name in ('inference_input', 'validation'):
            if name + '_file' in record:
                require(sha256_file(run / record[name + '_file']) == record[name + '_sha256'],
                        'Saved artifact checksum mismatch: ' + name)
        require(bool(record.get('source_sha256')),
                'Upstream C3 failure with no frozen source artifacts is not supported by this analyzer; '
                'do not omit it or report a completed cohort')
        saved, report = source_data(record)
        coverage[position] = source_inventory(saved)
        for field in IDENTITY:
            if field in saved:
                require(saved[field] == samples[position][field], 'Source identity differs: ' + field)
        originals[position] = saved
        input_path = path.parent / 'inference_input.json'
        observed = read(input_path) if input_path.exists() else None
        if observed is not None:
            check_input(observed, saved, report, head)
            inputs[position] = observed
        calls = audit_calls(path.parent, protocol, observed)
        calls_by_position[position] = calls
        if record['status'] == 'ok':
            require(observed is not None and calls, 'Successful case lacks input or model calls')
            result_path = run / record['result_file']
            require(sha256_file(result_path) == record['result_sha256'], 'Result checksum mismatch')
            result = read(result_path)
            original = ranking_paths(saved)
            check_inventory(result['ranking'], original, head)
            require(result['ground_truth'] == saved['ground_truth'], 'Gold labels changed')
            metrics = ranking_metrics(result['ranking'], saved['ground_truth'], 10)
            require(metrics == result['metrics'] == record['metrics'], 'Ranking metrics differ')
            require(result['model_calls'] == len(calls), 'Model call count differs')
            require(calls[-1]['status'] == 'ok', 'Successful ranking lacks successful final call')
            from bug_localization.blagent_recovery import json_object
            response = json_object(calls[-1]['response'])
            require(response == result['structured_output'], 'Structured output differs from actual model response')
            check_response(response, observed, result)
            if protocol.get('method') in ('baseline', 'evidence'):
                from bug_localization.evidence_reranking import validate_response
                require(not validate_response(response, observed, protocol['method']),
                        'Successful result fails frozen mechanical validation')
            results[position] = result
        else:
            require(bool(record.get('error')), 'Failed case lacks error detail')
        normalized = dict(record)
        missing = coverage[position]['missing_gold']
        if 'missing_gold' in normalized:
            require(normalized['missing_gold'] == missing, 'Source repository coverage metadata differs')
        normalized['missing_gold'] = missing
        records[position] = normalized
    ordered = [records[s['position']] for s in selection['samples'] if s['position'] in records]
    return {'protocol': protocol, 'records': ordered, 'inputs': inputs, 'originals': originals,
            'results': results, 'calls': calls_by_position, 'coverage': coverage,
            'summary': summary(ordered, len(samples)), 'cost': cost_report(ordered, calls_by_position)}


def cluster_inference(delta, repositories, *, repetitions=5000, seed=20261119):
    """Resample whole projects and exchange method labels by whole project."""
    d = np.asarray(delta, dtype=float)
    require(len(d) and len(d) == len(repositories) and np.isfinite(d).all(), 'Invalid paired differences')
    require(repetitions > 0, 'No inference repetitions')
    names = sorted(set(repositories))
    groups = [np.flatnonzero(np.asarray(repositories) == p) for p in names]
    sums = np.array([d[g].sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(repetitions, len(groups)))
    boots = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    if len(groups) <= 16:
        signs = np.array(list(itertools.product((-1, 1), repeat=len(groups))))
        nulls = signs @ sums / len(d)
        pvalue = float((np.abs(nulls) >= abs(d.mean()) - 1e-15).mean())
        method = 'Exact paired method-label swaps by whole project'
    else:
        signs = rng.choice([-1, 1], size=(repetitions, len(groups)))
        nulls = signs @ sums / len(d)
        pvalue = float((1 + (np.abs(nulls) >= abs(d.mean()) - 1e-15).sum()) / (repetitions + 1))
        method = 'Monte Carlo paired method-label swaps by whole project'
    return {'n_reports': len(d), 'n_projects': len(groups), 'delta': float(d.mean()),
            'ci95_project_cluster_bootstrap': np.quantile(boots, [.025, .975]).tolist(),
            'p_two_sided_project_randomization': pvalue, 'randomization_method': method,
            'randomization_draws': len(nulls), 'seed': seed, 'bootstrap_repetitions': repetitions,
            'note': 'Few development projects limit inference; no independent-test or population-generalization claim'}


def holm(stats, p_key, output_key):
    previous = 0.
    for i, row in enumerate(sorted(stats, key=lambda v: v[p_key])):
        previous = min(1., max(previous, (len(stats) - i) * row[p_key]))
        row[output_key] = previous


def evidence_diagnostics(run):
    states = dict.fromkeys(('support', 'contradiction', 'unknown'), 0)
    top1_states = dict(states)
    references, constraints, first_valid, repaired = 0, 0, 0, 0
    for position, result in run['results'].items():
        response = result['structured_output']
        if 'assessments' not in response:
            continue
        constraints += len(response['constraints'])
        for assessment in response['assessments']:
            states[assessment['state']] += 1
            references += len(assessment['evidence'])
            if assessment['file_id'] == response['ranked_ids'][0]:
                top1_states[assessment['state']] += 1
        if len(run['calls'][position]) == 1:
            first_valid += 1
        else:
            repaired += 1
    return {'successful_cases': len(run['results']), 'assessment_states': states,
            'top1_assessment_states': top1_states, 'report_constraints': constraints,
            'code_references': references, 'valid_after_one_call': first_valid,
            'valid_after_format_correction': repaired,
            'note': 'States are model judgments. Citation/quote validity is mechanically checked; semantic or causal correctness is not certified.'}


def actual_output(call):
    """Read output from this call only, including a budget-exhausted raw reply."""
    for field in ('response', 'raw_response'):
        if isinstance(call.get(field), str) and call[field].strip():
            return call[field], field
    raw = call.get('server_response', {}).get('content')
    if isinstance(raw, str) and raw.strip():
        return raw, 'server_response.content'
    return None, None


def last_output_rank(record, calls, payload, gold):
    """Post-hoc diagnosis only: never repair an order or choose an earlier call."""
    from bug_localization.blagent_recovery import json_object
    row = {k: record[k] for k in IDENTITY}
    row.update(original_primary_status=record['status'], status='failed',
               error_type='RankOnlyDiagnosticFailure', error='No final model output',
               last_call_number=len(calls), last_request_sha256=None,
               rule='Only the last attempted call; no repair, fallback, or best-response selection')
    if not calls or payload is None:
        return row
    final = calls[-1]
    row['last_request_sha256'] = final.get('request_sha256')
    raw, field = actual_output(final)
    row['output_field'] = field
    if raw is None:
        return row
    try:
        value = json_object(raw)
        require(isinstance(value, dict), 'Final output is not a JSON object')
        ids = value.get('ranked_ids')
        files = payload['catalog']['files']
        require(isinstance(ids, list) and all(isinstance(i, str) for i in ids)
                and len(ids) == len(files) and len(set(ids)) == len(files) and set(ids) == set(files),
                'Final output lacks an exact candidate-ID permutation')
        ranking = [files[i] for i in ids] + payload['tail']
        check_inventory(ranking, payload['original_ranking'], len(files))
    except (ValueError, TypeError, KeyError) as error:
        row['error'] = str(error)
        return row
    row.pop('error')
    row.pop('error_type')
    row.update(status='ok', ranking=ranking, metrics=ranking_metrics(ranking, gold, 10),
               missing_gold=list(record.get('missing_gold', [])),
               primary_validation_ignored=record['status'] != 'ok')
    return row


def failure_categories(errors):
    """Allow multiple categories; these describe output validity, not semantics."""
    categories = set()
    for message in errors:
        text = str(message).lower()
        if 'quote must be an exact' in text or 'report substring' in text:
            categories.add('non_exact_report_quote')
        if "another file's excerpt" in text or 'another file' in text:
            categories.add('cross_file_reference')
        if 'reference range outside' in text or 'citation range' in text:
            categories.add('reference_out_of_bounds')
        if 'ranked_ids' in text or 'candidate-id permutation' in text:
            categories.add('incomplete_or_invalid_ranking')
        if 'undefined report constraints' in text or 'constraint ids must be unique' in text:
            categories.add('undefined_or_duplicate_constraints')
        if 'assessments must assess' in text:
            categories.add('missing_or_duplicate_file_assessment')
        if 'requires constraints and code references' in text:
            categories.add('missing_supporting_citation')
        if 'output budget' in text or 'inference truncated' in text:
            categories.add('output_budget_or_truncation')
        if 'context overflow' in text:
            categories.add('input_context_overflow')
        if text.startswith('schema ') or 'json parse:' in text:
            categories.add('schema_or_json')
    if errors and not categories:
        categories.add('other_output_validation_failure')
    return sorted(categories)


def call_validation_errors(call, payload, variant):
    from bug_localization.blagent_recovery import json_object
    from bug_localization.evidence_reranking import validate_response
    errors = [call['error']] if call.get('error') else []
    raw, _ = actual_output(call)
    if raw is None or payload is None:
        return errors or ['No model output available']
    try:
        errors.extend(validate_response(json_object(raw), payload, variant))
    except (ValueError, TypeError) as error:
        errors.append('JSON parse: ' + str(error))
    return errors


def posthoc_rank_only_diagnostic(runs, total):
    """No significance tests or success decision: explain strict-output losses."""
    methods, failures = {}, {}
    for variant, run in runs.items():
        rows, terminal_counts, attempted_counts, cases = [], {}, {}, []
        for record in run['records']:
            position = record['position']
            payload = run['inputs'].get(position)
            calls = run['calls'][position]
            rows.append(last_output_rank(record, calls, payload, run['originals'][position]['ground_truth']))
            attempt_details, all_categories = [], set()
            for number, call in enumerate(calls, 1):
                errors = call_validation_errors(call, payload, variant)
                categories = failure_categories(errors)
                all_categories.update(categories)
                attempt_details.append({'attempt': number, 'categories': categories, 'errors': errors})
            for category in all_categories:
                attempted_counts[category] = attempted_counts.get(category, 0) + 1
            terminal = []
            if record['status'] == 'failed':
                errors = ([record['error']] + (attempt_details[-1]['errors'] if attempt_details else []))
                terminal = failure_categories(errors)
                for category in terminal:
                    terminal_counts[category] = terminal_counts.get(category, 0) + 1
            cases.append({'position': position, 'primary_status': record['status'],
                          'terminal_failure_categories': terminal, 'attempts': attempt_details})
        methods[variant] = {'summary': summary(rows, total), 'cases': rows,
                            'primary_failed_but_final_rank_parseable': sum(
                                r['original_primary_status'] == 'failed' and r['status'] == 'ok' for r in rows)}
        failures[variant] = {'terminal_failed_cases': sum(r['status'] == 'failed' for r in run['records']),
                             'terminal_failure_case_counts_multilabel': terminal_counts,
                             'any_attempt_failure_case_counts_multilabel': attempted_counts,
                             'cases': cases}
    return {
        'label': 'posthoc_rank_only_diagnostic', 'posthoc': True, 'primary_results_unchanged': True,
        'rule': 'Use only the last attempted call from each method and report. Parse its actual response/raw response; '
                'accept only an exact complete ranked_ids permutation, ignoring citation validity. '
                'No repair, previous-response fallback, outcome-based choice, or new inference. Unusable final ranks count zero.',
        'interpretation': 'Secondary diagnostic separating output-validation failures from ranking behavior; '
                          'not a passed experiment or evidence that the method improves localization. '
                          'No significance test or success decision is based on these diagnostic scores.',
        'methods': methods, 'failure_classification': failures,
    }


def analyze(baseline, evidence, selection_path, output, *, repetitions=5000):
    selection = read(selection_path)
    samples = selection['samples']
    require(samples and len({s['position'] for s in samples}) == len(samples), 'Empty or duplicate selection')
    require(len({tuple(s[k] for k in IDENTITY[1:]) for s in samples}) == len(samples), 'Duplicate report identities')
    runs = {name: audit_run(folder, selection, selection_path)
            for name, folder in [('baseline', baseline), ('evidence', evidence)]}
    a, b = runs['baseline'], runs['evidence']
    require(a['protocol']['head'] == b['protocol']['head'], 'Head budgets differ')
    require(a['protocol']['generator_identity'] == b['protocol']['generator_identity'], 'Models or generation budgets differ')
    common = sorted(set(a['inputs']) & set(b['inputs']))
    for position in common:
        for field in ('query', 'candidates', 'evidence', 'original_ranking', 'tail', 'catalog', 'user_content_sha256'):
            require(a['inputs'][position][field] == b['inputs'][position][field], 'Paired model input differs: ' + field)
        require(a['originals'][position]['ground_truth'] == b['originals'][position]['ground_truth'], 'Paired labels differ')
    complete = all(not r['summary']['pending_samples'] for r in runs.values())
    methods = {name: value['records'] for name, value in runs.items()}
    paired = paired_analyze(methods, {'evidence minus baseline': {'evidence': 1, 'baseline': -1}},
                            repetitions=repetitions) if complete else None
    clusters = None
    if complete:
        repositories = [s['repository'] for s in samples]
        clusters = {m: cluster_inference(values(methods['evidence'], m) - values(methods['baseline'], m),
                                          repositories, repetitions=repetitions) for m in METRICS}
        holm(list(clusters.values()), 'p_two_sided_project_randomization', 'p_holm_six_metrics')
    originals = {**a['originals'], **b['originals']}
    coverage = {**a['coverage'], **b['coverage']}
    c3 = []
    for sample in samples:
        saved = originals.get(sample['position'])
        if saved is not None:
            gold = saved['ground_truth']
            c3.append({**sample, 'status': 'ok', 'metrics': ranking_metrics(ranking_paths(saved), gold, 10),
                       'missing_gold': coverage[sample['position']]['missing_gold']})
    project_summaries = {p: {name: summary([r for r in rows if r['repository'] == p],
                                         sum(s['repository'] == p for s in samples))
                             for name, rows in {**methods, 'C3_descriptive': c3}.items()}
                         for p in sorted({s['repository'] for s in samples})}
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'analysis_script_sha256': sha256_file(Path(__file__)),
        'selection_sha256': sha256_file(selection_path), 'selected_samples': len(samples),
        'status': 'complete' if complete else 'pending',
        'note': 'Exploratory development cohort, previously observed; not independent blind testing. '
                'Only evidence-minus-baseline is the primary comparison; C3 is descriptive. '
                'Evidence selection and dual retrieval are held fixed. Incremental latency is not end-to-end latency.',
        'audit': {'status': 'passed' if complete else 'pending', 'paired_input_cases': len(common),
                  'actual_source_hashes_checked': True, 'failure_policy': 'zero, full denominator'},
        'coverage_audit': {'definition': 'Repository coverage uses the complete original candidates.json pinned '
                                         'by C3 source_files_sha256, with its C1 reference hash chain verified. '
                                         'C3 ranking/candidates is only the reranking pool and is not repository inventory.',
                           'cases': {str(k): v for k, v in sorted(coverage.items())},
                           'available_gold_outside_pool_positions': [p for p, v in sorted(coverage.items())
                                                                    if v['gold_available_but_outside_c3_ranking']]},
        'runs': {name: {'path': str(folder), 'protocol_sha256': sha256_file(folder / 'protocol.json'),
                        'summary': runs[name]['summary'], 'cost': runs[name]['cost']}
                 for name, folder in [('baseline', baseline), ('evidence', evidence)]},
        'C3_descriptive': summary(c3, len(samples)), 'project_summaries': project_summaries,
        'paired_report_statistics': paired, 'project_cluster_statistics': clusters,
        'evidence_diagnostics': evidence_diagnostics(b),
        'posthoc_rank_only_diagnostic': posthoc_rank_only_diagnostic(runs, len(samples)) if complete else None,
        'multiplicity_note': 'Six metrics form one Holm family within each separately labeled inference scheme. '
                             'Historical/adaptive experiments are outside these families; all results remain exploratory.',
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'analysis.json', report)
    case_rows = []
    indexes = {name: {r['position']: r for r in rows} for name, rows in methods.items()}
    for sample in samples:
        row = {**sample}
        for name, index in indexes.items():
            record = index.get(sample['position'])
            row[name + '_status'] = record['status'] if record else 'pending'
            row[name + '_ap'] = (record['metrics']['average_precision_at_k'] if record['status'] == 'ok' else 0.) if record else None
            row[name + '_top1'] = (record['metrics']['top_1'] if record['status'] == 'ok' else 0.) if record else None
        row['delta_ap'] = row['evidence_ap'] - row['baseline_ap'] if None not in (row['evidence_ap'], row['baseline_ap']) else None
        case_rows.append(row)
    write_json(output / 'paired_cases.json', case_rows)
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['method', 'successful', 'failed', 'pending', *METRIC_NAMES])
        writer.writeheader()
        for name, result in {**{k: v['summary'] for k, v in runs.items()}, 'C3_descriptive': report['C3_descriptive']}.items():
            writer.writerow({'method': name, 'successful': result['successful_samples'],
                             'failed': result['failed_samples'], 'pending': result['pending_samples'],
                             **(result['metrics'] or {})})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'evidence', 'selection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=5000)
    args = parser.parse_args()
    require(2000 <= args.repetitions <= 50000, 'Use 2000 to 50000 repetitions')
    paths = [(PROJECT / getattr(args, name)).resolve() for name in ('baseline', 'evidence', 'selection', 'output')]
    result = analyze(*paths, repetitions=args.repetitions)
    print(json.dumps({'status': result['status'], 'selected_samples': result['selected_samples'],
                      'audit': result['audit']}, indent=2))


if __name__ == '__main__':
    main()
