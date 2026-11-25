"""Run the author's RRL agent with live reformulation and a local model backend.

Author source: gencau/boatse-query-reformulation (Apache-2.0), frozen below.
The default 7B / Q4_K_M CPU run is a functional adaptation, not paper scores.
"""

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request
from unittest.mock import patch

from .batch import aggregate_results, write_checkpoints, write_json
from .blaze import digest
from .bm25 import collect_files, prepare_report, ranking_metrics, validate_config as validate_base
from .snapshot import get_snapshot, sha256_file
from .vector import select_reports

AUTHOR_REVISION = '50903acf3d6124514ff4f4c179cfd67a3e8ea414'


class BudgetExceeded(BaseException):
    """Escape author retry loops, which deliberately catch ordinary Exceptions."""


def validate_config(config, base):
    validate_base(base)
    if base['top_k'] != 10:
        raise ValueError('This RRL adapter exports a top-10 file ranking')
    endpoint = urllib.parse.urlsplit(config['endpoint'])
    if endpoint.scheme != 'http' or endpoint.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('RRL smoke uses an explicitly local model endpoint')
    for field, low, high in [('max_calls', 4, 100), ('max_output_tokens', 128, 4096),
                             ('context_tokens', 4096, 32768), ('timeout_seconds', 1, 1800)]:
        if type(config.get(field)) is not int or not low <= config[field] <= high:
            raise ValueError('Invalid RRL ' + field)
    if config['max_output_tokens'] >= config['context_tokens']:
        raise ValueError('RRL output budget leaves no room for the prompt')


def load_author(project, config):
    root = (project / config['author_checkout']).resolve()
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != AUTHOR_REVISION:
        raise ValueError('RRL author checkout does not match the pinned revision')
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', 'HEAD', '--', 'bug_localization/src'],
                   check=True, stdout=subprocess.DEVNULL)
    package = root / 'bug_localization'
    sys.path[:0] = [str(package), str(package / 'src/baselines')]
    author = importlib.import_module('src.baselines.agent_workflow_defined_set')
    author_tools = importlib.import_module(author.exec_tool.__module__)
    path_utils = importlib.import_module(author.is_missing_path.__module__)
    # Use the actual model tokenizer for source previews and context accounting.
    author.tk.TokenizationUtils.PROFILE_NAME_TO_PROVIDER_AND_MODEL[config['model']] = {
        'model_provider': 'huggingface', 'model_name': str((project / config['tokenizer_path']).resolve()),
        'context_size': config['context_tokens']}
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted((package / 'src').rglob('*.py'))}
    return author, author_tools, path_utils, {'repository': 'https://github.com/gencau/boatse-query-reformulation',
                                             'revision': revision, 'source_sha256': files}


def safe_candidate_path(repo_dir, path, allowed):
    if not isinstance(path, str) or path not in allowed:
        return None
    root = Path(repo_dir).resolve()
    candidate = (root / path).resolve()
    return str(candidate) if candidate.is_relative_to(root) and candidate.is_file() else None


class LocalChat:
    def __init__(self, config, tokenizer, folder):
        self.config, self.tokenizer, self.folder = config, tokenizer, folder
        self.calls, self.failed_calls, self.truncated_calls = 0, [], []

    def __call__(self, messages, model_kwargs=None, timeout_s=None):
        if self.calls >= self.config['max_calls']:
            raise BudgetExceeded('RRL exceeded the configured model-call budget')
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=False)
        if len(tokens) + self.config['max_output_tokens'] > self.config['context_tokens']:
            raise BudgetExceeded('RRL prompt plus reserved output exceeds the configured context; no silent truncation')
        self.calls += 1
        body = {'model': self.config['model'], 'messages': messages, 'temperature': 0,
                'max_tokens': self.config['max_output_tokens'], 'seed': 42,
                'repeat_penalty': 1.05, 'stream': False}
        request_path = self.folder / f'call_{self.calls:03d}.json'
        audit = {'request': body, 'prompt_tokens_local': len(tokens), 'status': 'running'}
        write_json(request_path, audit)
        started = time.monotonic()
        try:
            request = urllib.request.Request(self.config['endpoint'].rstrip('/') + '/chat/completions',
                                              data=json.dumps(body).encode(),
                                              headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=self.config['timeout_seconds']) as response:
                result = json.load(response)
            choice = result['choices'][0]
            text = choice['message']['content']
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Local RRL model returned an empty response')
            if choice.get('finish_reason') == 'length':
                self.truncated_calls.append(self.calls)
            audit.update(status='ok', response=result)
            return text
        except Exception as error:
            self.failed_calls.append(self.calls)
            audit.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            audit['elapsed_seconds'] = time.monotonic() - started
            write_json(request_path, audit)


def run_agent(author, author_tools, path_utils, *, config, query, repo_dir, candidates, folder):
    from .rrl_index import search_index

    tokenizer = author.tk.TokenizationUtils(config['model'])
    chat = LocalChat(config, tokenizer._tokenizer, folder)
    logger = author.RunLogger(log_dir=str(folder / 'author_logs'), run_id='sample', model_name=config['model'])
    allowed_reads = set(candidates)
    if (repo_dir / 'README.md').is_file():
        allowed_reads.add('README.md')

    def validate_files(paths, directory):
        valid, invalid = [], []
        for path in paths or []:
            if safe_candidate_path(directory, path, candidates):
                valid.append(path)
            else:
                invalid.append((str(path), 'outside the shared source-file candidates or invalid path'))
        return valid, invalid

    prompt = author.AgentContextPrompt()
    extraction_messages = [{'role': 'system', 'content': prompt.get_system_prompt()},
                           {'role': 'user', 'content': prompt.get_base_extract_prompt(issue_description=query)}]
    extracted = author.parse_json_safe(chat(extraction_messages))
    retrieval_fields = {'explanation', 'path', 'filename', 'identifiers', 'code snippet', 'code_snippet', 'error_message'}
    if not isinstance(extracted, dict) or not any(extracted.get(key) for key in retrieval_fields):
        raise ValueError('RRL live query reformulation did not return a nonempty JSON object')
    write_json(folder / 'reformulation.json', {'query_sha256': digest(query), 'information': extracted,
                                              'source': 'live generation from this report only'})
    ctx = author.RunContext(repo_name=config['_repository'], repo_dir=str(repo_dir), dataset='swe',
                            issue_index=0, issue_description=query, model_name=config['model'], num_files=10,
                            tk=tokenizer, extracted_option='all', viewed_files=[], viewed_files_full={},
                            bm25_results=[], extracted_info=[])

    def current_extraction(issue_idx, dataset, option='all'):
        if issue_idx != 0 or option != 'all':
            raise ValueError('RRL reformulation is bound to this one report and the all-fields setting')
        return extracted

    guarded_path = lambda directory, path: safe_candidate_path(directory, path, allowed_reads)
    with patch.object(author, 'run_with_timeout', chat), \
            patch.object(author_tools, 'extract_relevant', current_extraction), \
            patch.object(author_tools, 'search_index', lambda query, repo_dir, top_k=10:
                         search_index(repo_dir, query, top_k)), \
            patch.object(author_tools, 'safe_repo_path', guarded_path), \
            patch.object(path_utils, 'safe_repo_path', guarded_path), \
            patch.object(author, 'validate_ranked_files', validate_files):
        try:
            messages = [{'role': 'system', 'content': author.SYSTEM_PROMPT.format(n=ctx.num_files)}]
            first, args1, result1 = author._run_turn(
                messages=messages, prompt_tmpl=author.PROMPT_TURN_1,
                prompt_kwargs={'issue_description': query, 'function_calls': author.FUNCTION_CALLS_TURN_1,
                               'repo_name': ctx.repo_name, 'n': ctx.num_files},
                step='turn_1', ctx=ctx, logger=logger, content_limit=author.CONTENT_LIMIT)
            messages.append({'role': 'assistant', 'content': author.to_assistant_content(result1)})
            second, args2, result2 = author._run_turn(
                messages=messages, prompt_tmpl=author.PROMPT_TURN_2,
                prompt_kwargs={'tool_call': first, 'args': args1,
                               'function_calls': author.FUNCTION_CALLS_TURN_2, 'n': ctx.num_files},
                step='turn_2', ctx=ctx, logger=logger, content_limit=author.CONTENT_LIMIT)
            final = author._turn3_loop(messages=messages, tool_called=second, tool_args=args2, tool_result=result2,
                                       ctx=ctx, logger=logger, content_limit=author.CONTENT_LIMIT)
            initial = final.get('ranked_files', [])
            reviewed = author._self_evaluate_and_adopt(messages=messages, ranked_files=initial, ctx=ctx, logger=logger)
            ranked = reviewed if reviewed else initial
            valid, invalid = validate_files(ranked, str(repo_dir))
            if invalid or not valid:
                raise ValueError('RRL did not produce a valid nonempty candidate-file ranking')
            if not ctx.extracted_info or not ctx.bm25_results:
                raise ValueError('RRL agent skipped query reformulation or retrieval; flow is incomplete')
            if chat.failed_calls or chat.truncated_calls:
                raise ValueError('RRL model calls failed or exhausted output tokens; inspect the saved calls')
            return {'ranking': list(dict.fromkeys(valid))[:10], 'initial_ranking': initial,
                    'model_calls': chat.calls, 'viewed_files': ctx.viewed_files,
                    'unique_viewed_file_count': len(set(ctx.viewed_files)),
                    'bm25_results': ctx.bm25_results, 'extracted_info': ctx.extracted_info,
                    'self_review_completed': True, 'reprompts': logger.summary['reprompts']}
        finally:
            logger.close()


def run(project, config, output_directory=None):
    from .rrl_index import prepare_index, set_runtime

    set_runtime(project)
    base = json.loads((project / config['base_config']).read_text())
    validate_config(config, base)
    selected, selection = select_reports(project, config, base,
                                         max_samples=5000 if config.get('selection_manifest') else 50)
    author, author_tools, path_utils, source_identity = load_author(project, config)
    with urllib.request.urlopen(config['endpoint'].rstrip('/') + '/models', timeout=5) as response:
        served = json.load(response)
    if config['model'] not in [model['id'] for model in served['data']]:
        raise ValueError('The local server does not expose the configured RRL model')
    model_path = project / config['model_file']
    server_root = config['endpoint'].rstrip('/').removesuffix('/v1')
    with urllib.request.urlopen(server_root + '/props', timeout=5) as response:
        properties = json.load(response)
    if Path(properties.get('model_path', '')).resolve() != model_path.resolve():
        raise ValueError('Local server model_path does not match the configured GGUF')
    manifest = json.loads((model_path.parent / 'manifest.json').read_text())
    model_hash = sha256_file(model_path)
    if model_hash != manifest['gguf_sha256']:
        raise ValueError('RRL GGUF changed since its conversion manifest was created')
    model_identity = {'path': str(model_path.resolve()), 'sha256': model_hash, 'preparation_manifest': manifest,
                      'tokenizer': {name: sha256_file(project / config['tokenizer_path'] / name)
                                    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json')}}
    now = datetime.now(timezone(timedelta(hours=8)))
    folder = project / (output_directory or Path('results/rrl') / now.strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    context = {'created_at': now.isoformat(), 'variant': config.get('runtime_variant', manifest['variant']),
               'dataset': base['dataset'], 'split': base['split'], 'selection_sha256': selection['selection_sha256'],
               'author': source_identity, 'model': model_identity, 'served_models': served,
               'server_properties': properties,
               'implementation_sha256': {name: sha256_file(Path(__file__).with_name(name))
                                         for name in ('rrl.py', 'rrl_index.py')},
               'dependencies': {name: version(name) for name in ('pyserini', 'transformers', 'langchain-text-splitters')},
               'scope': 'one-report or small functional check; model/backend and shared candidates adapted; not paper reproduction'}
    write_json(folder / 'selection.json', selection)
    write_json(folder / 'effective_config.json', {'base': base, 'rrl': config})
    write_json(folder / 'provenance.json', context)
    records = []
    write_checkpoints(folder, records, {**context, **aggregate_results(records, 10, len(selected))})
    for position, sample in enumerate(selected, 1):
        row = sample['report']
        record = {'position': position, 'dataset_row': sample['dataset_row'], 'repository': row['repo_name'],
                  'issue_id': row['issue_id'], 'before_fix_sha': row['before_fix_sha']}
        case = folder / 'cases' / f'{position:04d}'
        case.mkdir(parents=True)
        started = time.monotonic()
        print(f"[{position}/{len(selected)}] RRL {row['repo_name']} #{row['issue_id']}", flush=True)
        try:
            effective = {**base, 'repository': row['repo_name'], 'issue_id': row['issue_id']}
            _, gold, query = prepare_report(row, effective)
            source, manifest = get_snapshot(project, row['repo_name'], row['before_fix_sha'],
                                            **({'source_provider': base['source_provider'], 'source_dataset': base['dataset']}
                                               if base.get('source_provider') else {}))
            documents, skipped = collect_files(source, manifest, base)
            candidates = {d['file_path'] for d in documents}
            workspace = case / 'workspace'
            for document in documents:
                target = workspace / document['file_path']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(document['content'], encoding='utf-8')
            if (source / 'README.md').is_file():
                shutil.copyfile(source / 'README.md', workspace / 'README.md')
            inventory = [{'file_path': d['file_path'], 'sha256': d['sha256']} for d in documents]
            write_json(case / 'inputs.json', {'query': query, 'query_fields': base['query_fields'],
                                             'query_sha256': digest(query), 'candidates': inventory,
                                             'auxiliary_readme_sha256': sha256_file(workspace / 'README.md')
                                                if (workspace / 'README.md').is_file() else None,
                                             'candidate_fingerprint': digest(inventory), 'skipped_files': skipped})
            with (case / 'author.stdout.log').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
                index_audit = prepare_index(project, workspace, documents)
                result = run_agent(author, author_tools, path_utils, config={**config, '_repository': row['repo_name']},
                                   query=query, repo_dir=workspace, candidates=candidates, folder=case)
            metrics = ranking_metrics(result['ranking'], gold, 10)
            output = {**record, **result, 'index': index_audit, 'metrics': metrics,
                      'gold_files': gold, 'missing_gold': sorted(set(gold) - candidates),
                      'indexed_files': len(documents), 'candidate_fingerprint': digest(inventory)}
            write_json(case / 'result.json', output)
            record.update(status='ok', metrics=metrics, missing_gold=output['missing_gold'], indexed_files=len(documents),
                          result_file=(case / 'result.json').relative_to(folder).as_posix())
            print(f"  completed: {result['model_calls']} model calls, AP@10={metrics['average_precision_at_k']:.4f}", flush=True)
        except (Exception, BudgetExceeded) as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
            write_json(case / 'error.json', {**record, 'traceback': traceback.format_exc()})
            print(f'  Failed: {type(error).__name__}: {error}', flush=True)
        record['elapsed_seconds'] = time.monotonic() - started
        records.append(record)
        summary = {**context, **aggregate_results(records, 10, len(selected))}
        write_checkpoints(folder, records, summary)
    print('RRL results:', folder, flush=True)
    return folder, summary
