"""Audited GenLoc local-model pilot on the shared, pre-fix Python file pool."""
from datetime import datetime, timezone, timedelta
import fcntl
import json
from pathlib import Path
import time
import traceback

from .batch import aggregate_results, write_checkpoints, write_json
from .bm25 import collect_files, prepare_report, ranking_metrics, validate_config
from .embeddings import digest
from .snapshot import get_snapshot, sha256_file
from .vector import select_reports


def shortlist(chunks, scores, chunk_limit=300, file_limit=50):
    """Rank chunks, then preserve each file's first occurrence, as upstream does."""
    import numpy as np
    scores = np.asarray(scores)
    if scores.shape != (len(chunks),) or not np.isfinite(scores).all():
        raise ValueError('Invalid chunk similarity scores')
    order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), i))[:chunk_limit]
    files = list(dict.fromkeys(chunks[i]['file_path'] for i in order))[:file_limit]
    return files, [{'chunk_id': chunks[i]['chunk_id'], 'file_path': chunks[i]['file_path'],
                    'score': float(scores[i]), 'chunk_index': i} for i in order]


def first_query_chunk(query):
    """Upstream splits at 8191 cl100k tokens and uses only the first query result."""
    import tiktoken
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    tokenizer = tiktoken.get_encoding('cl100k_base')
    count = lambda text: len(tokenizer.encode(text))
    parts = RecursiveCharacterTextSplitter(chunk_size=8191, chunk_overlap=0,
                                           length_function=count).split_text(query)
    if not parts:
        raise ValueError('Empty retrieval query')
    return parts[0], {'original_tokens': count(query), 'used_tokens': count(parts[0]),
                      'parts': len(parts), 'truncated': len(parts) > 1,
                      'tokenizer': 'cl100k_base', 'policy': 'author_first_query_chunk'}


def exploration_messages(query, author):
    # The native local server may not combine auto-tool grammar with final JSON
    # grammar. Describe the same upstream output schema in the system message.
    transport = ('\n\nFinal response format: when you are not invoking a tool, return one JSON data '
                 'object with exactly two top-level keys: analysis_of_the_bug_report (your analysis as '
                 'a string) and ranked_list (an array of objects with file and justification strings). '
                 'For example, the data shape is '
                 '{"analysis_of_the_bug_report":"your analysis",'
                 '"ranked_list":[{"file":"exact/repository/path.py","justification":"your evidence"}]}. '
                 'Replace the example values with your actual analysis and ranked files. '
                 'Return the data itself, without Markdown fences or a JSON schema wrapper '
                 '(no top-level type, properties, required, or additionalProperties).')
    return [{'role': 'system', 'content': author.system_prompt + transport},
            {'role': 'user', 'content': '\nGiven a bug report, your goal is to analyze and rank files '
             'by their likelihood of containing the bug. \n\nBug Report:\n' + query + '\n\n'
             'On your first response, invoke one or more of the provided functions and wait for '
             'their results. Return native tool calls in that response; the final-answer JSON '
             'format applies only after tool results are available.'}]


def explore(query, file_data, candidates, author, generator, case, config):
    """Inference only: this function receives neither gold nor patch metadata."""
    from .genloc_tools import Toolbox, normalize_ranking
    toolbox = Toolbox(file_data, candidates, author=author)
    messages = exploration_messages(query, author)
    events = []
    allowed_tools = {t['function']['name'] for t in author.tools}
    final_schema = author.response_format['json_schema']['schema']
    for iteration in range(1, config['max_iterations'] + 1):
        choice = 'required' if iteration == 1 else ('none' if iteration == config['max_iterations'] else 'auto')
        assistant = generator.complete(messages, author.tools, tool_choice=choice,
                                       response_schema=final_schema if choice == 'none' else None,
                                       stage=f'explore_{iteration}')
        calls = assistant.get('tool_calls') or []
        if calls:
            if choice == 'none':
                raise ValueError('Model called a tool in the forced final iteration')
            ids = [call.get('id') for call in calls]
            if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
                raise ValueError('Tool call IDs must be unique, nonempty strings')
            messages.append(assistant)
            for call in calls:
                start = time.monotonic()
                function = call.get('function') or {}
                name, raw_args = function.get('name'), function.get('arguments')
                arguments = None
                try:
                    if name not in allowed_tools:
                        raise ValueError('Unknown exploration tool')
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(arguments, dict):
                        raise ValueError('Tool arguments must be a JSON object')
                    response = toolbox.execute(name, arguments)
                except (ValueError, TypeError, KeyError) as error:
                    # Explicit tool error feedback consumes the existing iteration
                    # budget; it never adds a model call or reads arbitrary files.
                    response = {'error': 'Invalid tool request', 'detail': str(error)}
                event = {'iteration': iteration, 'call_id': call['id'], 'name': name,
                         'raw_arguments': raw_args, 'arguments': arguments, 'response': response,
                         'success': not (isinstance(response, dict) and 'error' in response),
                         'elapsed_seconds': time.monotonic() - start}
                events.append(event)
                write_json(case / 'tool_calls.json', events)
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'name': name,
                                 'content': json.dumps(response, ensure_ascii=False)})
            write_json(case / 'conversation.json', messages)
            continue
        if choice == 'required':
            raise ValueError('First iteration must perform an actual tool call')
        raw = assistant.get('content')
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError('Empty final response')
        write_json(case / 'final_raw.json', {'content': raw, 'iteration': iteration})
        ranking, normalization = normalize_ranking(raw, file_data, author=author)
        if not ranking:
            raise ValueError('No valid file remained after author path recovery')
        messages.append(assistant)
        write_json(case / 'conversation.json', messages)
        write_json(case / 'normalization.json', normalization)
        return {'ranking': ranking, 'iterations': iteration,
                'tool_calls': len(events), 'tool_errors': sum(not e['success'] for e in events),
                'method_body_calls': sum(e['name'] == 'get_method_body' for e in events),
                'ranking_outside_initial_candidates': [p for p in ranking if p not in candidates],
                'normalization': normalization}
    raise ValueError('Exploration exhausted the iteration budget without final output')


def localize(query, documents, author, generator, embedder, case, config):
    from .genloc_tools import build_file_data, build_chunks
    started = time.monotonic()
    file_data = build_file_data(documents, author=author)
    write_json(case / 'file_data.json', file_data)
    chunks = build_chunks(file_data, chunk_tokens=config['chunk_tokens'])
    if not chunks:
        raise ValueError('No method-body chunks in the candidate snapshot')
    write_json(case / 'chunks.json', chunks)
    vectors = embedder.encode([c['embedding_text'] for c in chunks])
    document_audit = embedder.last_audit
    query_text, query_audit = first_query_chunk(query)
    vector, model_query_audit = embedder.query(query_text)
    candidates, top_chunks = shortlist(chunks, vectors @ vector,
                                       config['retrieval_chunk_limit'], config['candidate_limit'])
    retrieval = {'candidates': candidates, 'top_chunks': top_chunks,
                 'initial_ranking': candidates, 'chunks': len(chunks),
                 'files_with_methods': sum(bool(f['methods']) for f in file_data),
                 'files_without_methods': sum(not f['methods'] for f in file_data),
                 'query_text': query_text, 'query_audit': query_audit,
                 'embedding_query_audit': model_query_audit,
                 'embedding_documents_audit': document_audit,
                 'elapsed_seconds': time.monotonic() - started,
                 'policy': config['retrieval_policy']}
    write_json(case / 'retrieval.json', retrieval)
    print(f'  retrieval: {len(chunks)} chunks, {len(candidates)} candidates; starting exploration', flush=True)
    started = time.monotonic()
    result = explore(query, file_data, candidates, author, generator, case, config)
    result.update(initial_ranking=candidates, chunks=len(chunks),
                  retrieval_seconds=retrieval['elapsed_seconds'],
                  exploration_seconds=time.monotonic() - started,
                  query_truncated=query_audit['truncated'])
    return result


def run(project, config, output, *, check_only=False):
    from .genloc_tools import AuthorCode
    base = json.loads((project / config['base_config']).read_text())
    validate_config(base)
    if base['dataset'] != 'swe_bench_formatted' or base['query_fields'] != ['body'] or base['extensions'] != ['.py'] or base['top_k'] != 10:
        raise ValueError('GenLoc pilot requires shared SWE/Python/body-only/@10 inputs')
    if config['max_iterations'] != 9 or config['max_calls'] != 9:
        raise ValueError('This pilot freezes the author-code nine-request stopping policy')
    if config.get('final_output_transport') != 'native-tools-first-call-data-example-v3':
        raise ValueError('Use the v3 pilot configuration for explicit first-call tool transport')
    if (config['chunk_tokens'], config['retrieval_chunk_limit'], config['candidate_limit']) != (300, 300, 50):
        raise ValueError('This pilot freezes author 300-token/300-chunk/50-file retrieval')
    selected, selection = select_reports(project, config, base, max_samples=50)
    original_selection = json.loads((project / config['selection_manifest']).read_text())
    selection['selection_rule'] = original_selection['selection_rule']
    author = AuthorCode(project / config['author_checkout'])
    if check_only:
        return {'selected_samples': len(selected), 'selection_sha256': selection['selection_sha256'],
                'author': author.provenance, 'configuration': config}
    from .genloc_runtime import LocalToolGenerator
    from .genloc_embedding import CodeXEmbed
    folder = (project / output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    embedder = None
    try:
        generator = LocalToolGenerator(project, config)
        embedder = CodeXEmbed(project, config['embedding'])
        modules = ['genloc.py', 'genloc_tools.py', 'genloc_runtime.py', 'genloc_embedding.py',
                   'bm25.py', 'batch.py', 'vector.py', 'snapshot.py']
        identity = {'config': config, 'base': base, 'selection_sha256': selection['selection_sha256'],
                    'author': author.provenance, 'generation': generator.identity,
                    'embedding': embedder.identity,
                    'dependency_environment': json.loads((project / '.runtime/genloc-deps/environment.json').read_text()),
                    'implementation': {p: sha256_file(Path(__file__).with_name(p)) for p in modules}}
        provenance = {'identity': identity, 'identity_sha256': digest(identity),
                      'variant': 'GenLoc-local Qwen3-8B BF16 + CodeXEmbed-400M; common Python pool; exact cosine',
                      'scope': 'six-project development smoke test; one seed, not paper-score reproduction'}
        if (folder / 'provenance.json').exists():
            if json.loads((folder / 'provenance.json').read_text())['identity'] != identity:
                raise ValueError('Resume identity mismatch; use a new output directory')
        else:
            provenance['created_at'] = datetime.now(timezone(timedelta(hours=8))).isoformat()
            write_json(folder / 'provenance.json', provenance)
            write_json(folder / 'selection.json', selection)
            write_json(folder / 'effective_config.json', {'genloc': config, 'base': base})
            frozen_source = folder / 'implementation'
            frozen_source.mkdir(exist_ok=True)
            for name, checksum in identity['implementation'].items():
                source = Path(__file__).with_name(name)
                if sha256_file(source) != checksum:
                    raise ValueError('Adapter implementation changed during startup')
                (frozen_source / name).write_bytes(source.read_bytes())
        records = []

        def checkpoint():
            summary = aggregate_results(records, 10, len(selected))
            if summary['status'] == 'completed' and summary['failed_samples']:
                summary['status'] = 'completed_with_failures'
            summary.update(variant=provenance['variant'], identity_sha256=provenance['identity_sha256'])
            retrieval_records = [r for r in records if 'initial_metrics' in r]
            summary['retrieval_successful_samples'] = len(retrieval_records)
            summary['initial_retrieval_metrics'] = None if summary['pending_samples'] else {
                name: sum(r['initial_metrics'][field] for r in retrieval_records) / len(selected)
                for name, field in [('MAP@10', 'average_precision_at_k'), ('MRR@10', 'reciprocal_rank_at_k'),
                                    ('Top-1', 'top_1'), ('Top-5', 'top_5'), ('Top-10', 'top_10')]}
            write_checkpoints(folder, records, summary)
            return summary

        summary = checkpoint()
        for position, sample in enumerate(selected, 1):
            row = sample['report']
            case = folder / 'cases' / f'{position:04d}'
            case.mkdir(parents=True, exist_ok=True)
            entry = {'position': position, 'dataset_row': sample['dataset_row'], 'repository': row['repo_name'],
                     'issue_id': row['issue_id'], 'before_fix_sha': row['before_fix_sha']}
            terminal = case / 'record.json'
            if terminal.exists():
                record = json.loads(terminal.read_text())
                if any(record.get(k) != v for k, v in entry.items()):
                    raise ValueError('Resume sample mismatch')
                if record['status'] == 'ok' and sha256_file(folder / record['result_file']) != record['result_sha256']:
                    raise ValueError('Result checksum mismatch')
                records.append(record)
                summary = checkpoint()
                continue
            print(f'[{position}/{len(selected)}] GenLoc {row["repo_name"]} #{row["issue_id"]}', flush=True)
            started = time.monotonic()
            record = dict(entry)
            generator.start_case(case)
            gold = None
            try:
                _, gold, query = prepare_report(row, {**base, 'repository': row['repo_name'], 'issue_id': row['issue_id']})
                source, manifest = get_snapshot(project, row['repo_name'], row['before_fix_sha'])
                documents, skipped = collect_files(source, manifest, base)
                inputs = {'query': query, 'source_commit': row['before_fix_sha'],
                          'candidate_files': [{'file_path': d['file_path'], 'sha256': d['sha256']} for d in documents]}
                write_json(case / 'inputs.json', inputs)
                # Labels stay in this evaluation scope, outside localize/explore.
                result = localize(query, documents, author, generator, embedder, case, config)
                pool = {d['file_path'] for d in documents}
                if not set(result['ranking']).issubset(pool):
                    raise ValueError('Final ranking escaped the common source pool')
                result.update(**entry, **inputs, gold_files=gold,
                              metrics=ranking_metrics(result['ranking'], gold, 10),
                              initial_metrics=ranking_metrics(result['initial_ranking'], gold, 10),
                              initial_recall_at_50=len(set(gold) & set(result['initial_ranking'])) / len(set(gold)),
                              missing_gold=sorted(set(gold) - pool), skipped_files=skipped,
                              gold_outside_initial_found=sorted(set(gold) & set(result['ranking']) - set(result['initial_ranking'])))
                write_json(case / 'result.json', result)
                record.update(status='ok', metrics=result['metrics'], initial_metrics=result['initial_metrics'],
                              missing_gold=result['missing_gold'], indexed_files=len(documents),
                              query_truncated=result['query_truncated'],
                              result_file=(case / 'result.json').relative_to(folder).as_posix(),
                              result_sha256=sha256_file(case / 'result.json'))
                print('  final ranking:', result['ranking'], flush=True)
            except Exception as error:
                record.update(status='failed', error_type=type(error).__name__, error=str(error))
                if gold and (case / 'retrieval.json').exists():
                    initial = json.loads((case / 'retrieval.json').read_text())['initial_ranking']
                    record['initial_metrics'] = ranking_metrics(initial, gold, 10)
                write_json(case / 'error.json', {**record, 'traceback': traceback.format_exc()})
                print('  FAILED:', type(error).__name__, str(error), flush=True)
            record.update(elapsed_seconds=time.monotonic() - started, model_calls=generator.calls)
            write_json(terminal, record)
            records.append(record)
            summary = checkpoint()
            limit = config.get('stop_after_failures', 2)
            if limit and summary['pending_samples'] and len(records) >= limit and all(r['status'] == 'failed' for r in records[-limit:]):
                summary['status'] = 'stopped_after_consecutive_failures'
                write_checkpoints(folder, records, summary)
                break
        return folder, summary
    finally:
        if embedder is not None:
            embedder.close()
        lock.close()
