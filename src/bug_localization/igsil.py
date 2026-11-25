"""IGSIL/CoSIL author file-level localization on the project's pinned inputs."""
from datetime import datetime, timezone, timedelta
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types
import urllib.request

from .batch import aggregate_results, write_checkpoints, write_json
from .bm25 import collect_files, prepare_report, ranking_metrics, validate_config
from .snapshot import get_snapshot, sha256_file
from .vector import select_reports

REVISION = '0568e423735b399d5b089996961fea9ae142e4c7'


class LocalDecoder:
    def __init__(self, *, endpoint, alias, tokenizer, folder, temperature, max_tokens, max_calls=12):
        self.endpoint, self.alias, self.tokenizer, self.folder = endpoint, alias, tokenizer, folder
        self.temperature, self.max_new_tokens, self.max_context_tokens = temperature, min(max_tokens or 2048, 2048), 32768
        self.calls, self.max_calls = 0, max_calls

    def codegen(self, message, num_samples=1, **kwargs):
        if num_samples != 1 or self.calls >= self.max_calls or kwargs.get('tools'):
            raise ValueError('Unsupported CoSIL model call or call budget exceeded')
        messages = message if isinstance(message, list) else [{'role': 'user', 'content': message}]
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        if len(tokens) + self.max_new_tokens > self.max_context_tokens:
            raise ValueError('IGSIL prompt exceeds local context; no silent truncation')
        self.calls += 1
        payload = {'model': self.alias, 'messages': messages, 'temperature': self.temperature,
                   'max_tokens': self.max_new_tokens, 'seed': 42, 'stream': False}
        audit = {'request': payload, 'prompt_tokens': len(tokens), 'status': 'running'}
        path = self.folder / f'call_{self.calls:03d}.json'
        write_json(path, audit)
        started = time.monotonic()
        try:
            req = urllib.request.Request(self.endpoint + '/chat/completions', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=600) as response:
                reply = json.load(response)
            choice = reply['choices'][0]
            audit.update(status='ok', response=reply)
            if choice.get('finish_reason') != 'stop':
                raise ValueError('IGSIL model output did not finish normally')
            return [{'response': choice['message']['content'], 'usage': reply.get('usage', {})}]
        except Exception as error:
            audit.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            audit['elapsed_seconds'] = time.monotonic() - started
            write_json(path, audit)


def make_structure(documents, parse_python_file):
    structure = {}
    for document in documents:
        parts = document['file_path'].split('/')
        current = structure
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        classes, functions, lines = parse_python_file('', document['content'])
        # Match the author parser output without executing source or cloning worktrees.
        current[parts[-1]] = {'classes': classes, 'functions': functions, 'text': lines}
    return structure


def run(project, config, output=None):
    from elasticsearch import Elasticsearch
    from transformers import AutoTokenizer
    base = json.loads((project / config['base_config']).read_text())
    validate_config(base)
    if base['extensions'] != ['.py'] or base['query_fields'] != ['body'] or base['top_k'] != 10:
        raise ValueError('IGSIL requires the pinned Python/body-only/@10 comparison protocol')
    checkout = (project / config['author_checkout']).resolve()
    if subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip() != REVISION:
        raise ValueError('Unexpected IGSIL author revision')
    subprocess.run(['git', '-C', str(checkout), 'diff', '--exit-code', 'HEAD', '--', 'CoSIL', 'get_repo_structure'],
                   check=True, stdout=subprocess.DEVNULL)
    selected, selection = select_reports(project, config, base, max_samples=5000 if config.get('selection_manifest') else 50)
    sys.path[:0] = [str(checkout), str(checkout / 'CoSIL/fl')]
    # The upstream checkout is a namespace package; its sibling CoSIL.py
    # otherwise shadows the package when the legacy FL_tools imports are used.
    package = types.ModuleType('CoSIL')
    package.__path__ = [str(checkout / 'CoSIL')]
    sys.modules['CoSIL'] = package
    author = importlib.import_module('CoSIL.fl.CoSIL')
    parser = importlib.import_module('get_repo_structure.get_repo_structure')
    tools = importlib.import_module('FL_tools')
    preprocess = importlib.import_module('CoSIL.util.preprocess_data')
    tokenizer = AutoTokenizer.from_pretrained(str(project / config['tokenizer_path']), local_files_only=True)
    server_root = config['endpoint'].removesuffix('/v1')
    with urllib.request.urlopen(server_root + '/props', timeout=5) as response:
        props = json.load(response)
    model = (project / config['model_file']).resolve()
    if Path(props['model_path']).resolve() != model:
        raise ValueError('IGSIL service model path mismatch')
    manifest = json.loads((model.parent / 'manifest.json').read_text())
    if sha256_file(model) != manifest['gguf_sha256']:
        raise ValueError('IGSIL model checksum mismatch')
    now = datetime.now(timezone(timedelta(hours=8)))
    folder = (project / (output or Path('results/igsil') / now.strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    context = {'created_at': now.isoformat(), 'dataset': base['dataset'], 'split': base['split'],
               'selection_sha256': selection['selection_sha256'], 'author_revision': REVISION,
               'variant': 'IGSIL file-level Qwen2.5-Coder-7B Q4_K_M llama.cpp CUDA',
               'model_sha256': manifest['gguf_sha256'], 'server_properties': props,
               'implementation_sha256': sha256_file(Path(__file__)),
               'protocol': 'Author file-level graph reflection, up to 5 ordered files; no test-file exclusion under shared candidate protocol; failed reports count zero'}
    write_json(folder / 'selection.json', selection)
    write_json(folder / 'effective_config.json', {'base': base, 'igsil': config})
    write_json(folder / 'provenance.json', context)
    work = folder / 'work'; (work / 'repo_structures').mkdir(parents=True)
    os.chdir(work)
    records = []
    write_checkpoints(folder, records, {**context, **aggregate_results(records, 10, len(selected))})
    for pos, sample in enumerate(selected, 1):
        row = sample['report']; case = folder / 'cases' / f'{pos:04d}'; case.mkdir(parents=True)
        rec = {'position': pos, 'dataset_row': sample['dataset_row'], 'repository': row['repo_name'],
               'issue_id': row['issue_id'], 'before_fix_sha': row['before_fix_sha']}
        started = time.monotonic()
        print(f"[{pos}/{len(selected)}] IGSIL {row['repo_name']} #{row['issue_id']}", flush=True)
        try:
            _, gold, query = prepare_report(row, {**base, 'repository': row['repo_name'], 'issue_id': row['issue_id']})
            source, manifest_source = get_snapshot(project, row['repo_name'], row['before_fix_sha'])
            docs, skipped = collect_files(source, manifest_source, base)
            structure = make_structure(docs, parser.parse_python_file)
            instance = 'case_' + str(pos)
            write_json(work / 'repo_structures' / (instance + '.json'), {'structure': structure})
            # The author code reads the saved structure using this exact relative path.
            author_files, _, _ = preprocess.extract_structure(structure)
            expected = {d['file_path'] for d in docs}
            if {item[0] for item in author_files} != expected:
                raise ValueError('IGSIL structure does not match shared candidate paths')
            tokenizer_args = {'endpoint': config['endpoint'], 'alias': config['model'], 'tokenizer': tokenizer, 'folder': case}
            holder = []
            def make_model(model, logger, max_tokens=None, temperature=0, **kwargs):
                decoder = LocalDecoder(**tokenizer_args, temperature=temperature, max_tokens=max_tokens)
                holder.append(decoder)
                return decoder
            sys.modules['CoSIL.util.model'] = types.SimpleNamespace(make_model=make_model)
            logger = logging.getLogger('igsil-' + str(pos));logger.setLevel(logging.INFO); logger.propagate = False
            handler = logging.FileHandler(case / 'author.log', encoding='utf-8');logger.addHandler(handler)
            try:
                predicted, artifact, trajectory = author.CoSIL(instance, structure, query, config['model'], logger).file_localize_with_g()
            finally:
                handler.close();logger.removeHandler(handler)
            if not isinstance(predicted, list) or not predicted or any(path not in expected for path in predicted):
                raise ValueError('IGSIL did not return a nonempty valid file ranking')
            if len(predicted) != len(set(predicted)):
                predicted = list(dict.fromkeys(predicted))
            if len(predicted) > 5:
                raise ValueError('IGSIL author Top-5 output exceeded the frozen limit')
            metrics = ranking_metrics(predicted, gold, 10)
            candidates = [{'file_path': d['file_path'], 'sha256': d['sha256']} for d in docs]
            result = {**rec, 'query': query, 'gold_files': gold, 'ranking': predicted, 'metrics': metrics,
                      'missing_gold': sorted(set(gold) - expected), 'candidate_files': candidates,
                      'skipped_files': skipped, 'model_calls': sum(x.calls for x in holder),
                      'author_artifact': artifact, 'author_trajectory': trajectory}
            write_json(case / 'result.json', result)
            rec.update(status='ok', metrics=metrics, missing_gold=result['missing_gold'], indexed_files=len(docs),
                       result_file=(case / 'result.json').relative_to(folder).as_posix())
            print('  ranking:', predicted, flush=True)
        except Exception as error:
            rec.update(status='failed', error_type=type(error).__name__, error=str(error))
            write_json(case / 'error.json', {**rec, 'traceback': traceback.format_exc()})
            print('  Failed:', type(error).__name__, str(error), flush=True)
        rec['elapsed_seconds'] = time.monotonic() - started
        records.append(rec)
        summary = {**context, **aggregate_results(records, 10, len(selected))}
        write_checkpoints(folder, records, summary)
    print('IGSIL results:', folder, flush=True)
    return folder, summary
