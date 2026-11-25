"""BLAgent author-agent adaptation to the project's pinned Python evaluation.

The author checkout supplies query prompts, the ReAct agent, skeleton extraction
and evidence pruning. Retrieval/model changes are explicit, never paper scores.
"""
from datetime import datetime, timedelta, timezone
import fcntl
import importlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .batch import aggregate_results, write_checkpoints, write_json
from .bm25 import collect_files, prepare_report, ranking_metrics, validate_config
from .embeddings import digest, LocalEmbedder
from .model_files import required_model_files
from .snapshot import get_snapshot, sha256_file
from .vector import select_reports

REVISION = '9f91f63e50aec8ad87a11e11148b086dfcd5e66d'
AUTHOR_FILES = ('blagent/agent/ranker_agent.py', 'blagent/agent/prompts/__init__.py',
                'blagent/util/query_augmentation.py', 'blagent/util/signature_extractor.py',
                'blagent/util/code_util.py', 'blagent/util/code_pruner.py')


def author_modules(project, config):
    checkout = (project / config['author_checkout']).resolve()
    revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('Unexpected BLAgent author revision')
    subprocess.run(['git', '-C', str(checkout), 'diff', '--exit-code', 'HEAD', '--', *AUTHOR_FILES], check=True)
    sys.path.insert(0, str(checkout))
    modules = {name: importlib.import_module('blagent.' + path) for name, path in (
        ('agent', 'agent.ranker_agent'), ('prompts', 'agent.prompts'),
        ('queries', 'util.query_augmentation'), ('pruner', 'util.code_pruner'))}
    return modules, {p: sha256_file(checkout / p) for p in AUTHOR_FILES}


def parse_ranking(text, candidates):
    """EAR must be a permutation; unknown/duplicated/omitted files are failures."""
    import re
    match = re.search(r'```(?:json)?\s*(.*?)```', text, re.S)
    value = json.loads(match.group(1) if match else text)
    paths = value['ranked_files']
    if (not isinstance(paths, list) or any(not isinstance(p, str) for p in paths)
            or len(paths) != len(candidates) or len(set(paths)) != len(paths)
            or set(paths) != set(candidates)):
        raise ValueError('EAR must return exactly the supplied files, without duplicates')
    return paths


def validate_scores(scores, candidates):
    if not isinstance(scores, dict) or not scores or not set(scores).issubset(candidates):
        raise ValueError('SAS must return a nonempty score dictionary within the candidate pool')
    if any(type(s) not in (float, int) or not math.isfinite(s) or not 0 <= s <= 10 for s in scores.values()):
        raise ValueError('SAS scores must be finite numbers in [0, 10]')
    # Python's stable sorting preserves the author's JSON order for ties.
    return sorted(scores, key=scores.get, reverse=True)


def candidate_union(t0, t1):
    # Author code, unlike the paper's prose, allows up to 30 total candidates.
    return list(dict.fromkeys(t0[:15] + t1[:15]))


def make_structure(documents):
    structure = {}
    for doc in documents:
        node = structure
        parts = doc['file_path'].split('/')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = {'text': doc['content'].split('\n')}
    return structure


class Generator:
    """Local, audited Qwen generation; no network/API and no silent truncation."""
    def __init__(self, project, config):
        from transformers import AutoTokenizer
        self.config = config
        self.path = project / config['generation_model']
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True, trust_remote_code=False)
        self.identity = {'files': {n: sha256_file(self.path / n) for n in required_model_files(self.path)},
                         'torch': version('torch'), 'transformers': version('transformers'),
                         'accelerate': version('accelerate'), 'thinking': False, 'dtype': 'bfloat16',
                         'sampling': 'per-request seeded sampling', 'settings': config}
        self.model = None

    def start_case(self, folder):
        self.folder = folder
        self.calls = 0

    def count_tokens(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def complete(self, messages, *, temperature=0, stop=None, stage='generation'):
        import torch
        from transformers import AutoModelForCausalLM, set_seed
        self.calls += 1
        if self.calls > self.config['max_calls']:
            raise ValueError('BLAgent per-case model call budget exceeded')
        request = {'messages': messages, 'temperature': temperature, 'stop': stop, 'stage': stage,
                   'max_new_tokens': self.config['max_new_tokens'], 'identity': digest(self.identity)}
        key = digest(request)
        target = self.folder / f'call_{self.calls:03d}.json'
        if target.exists():
            record = json.loads(target.read_text())
            if record['request'] != request or record['request_sha256'] != key:
                raise ValueError('BLAgent call checkpoint does not match current request')
            if record['status'] != 'ok':
                raise ValueError('Previous model call failed; use a separate output for retries')
            return record['response']
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        length = inputs.input_ids.shape[1]
        record = {'request': request, 'request_sha256': key, 'input_tokens': length, 'status': 'running'}
        started = time.monotonic()
        write_json(target, record)
        try:
            if length + self.config['max_new_tokens'] > self.config['max_context_tokens']:
                raise ValueError(f'BLAgent context overflow: {length} input tokens; no silent truncation')
            if self.model is None:
                print('Loading local BLAgent generation model on two GPUs', flush=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.path, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
                    attn_implementation='sdpa', device_map='auto',
                    max_memory={int(k): v for k, v in self.config['gpu_max_memory'].items()}).eval()
                if any(str(v) in ('cpu', 'disk') for v in self.model.hf_device_map.values()):
                    raise ValueError('Insufficient GPU memory; refusing implicit CPU/disk offload')
            device = self.model.get_input_embeddings().weight.device
            inputs = inputs.to(device)
            set_seed((self.config['seed'] + int(key[:8], 16)) % (2**32))
            kwargs = {'do_sample': temperature > 0, 'max_new_tokens': self.config['max_new_tokens'],
                      'pad_token_id': self.tokenizer.eos_token_id, 'logits_to_keep': 1}
            if temperature > 0:
                kwargs.update(temperature=temperature, top_p=0.9, top_k=40)
            if stop:
                kwargs.update(stop_strings=stop, tokenizer=self.tokenizer)
            with torch.inference_mode():
                output = self.model.generate(**inputs, **kwargs)
            tokens = output[0, length:]
            raw = self.tokenizer.decode(tokens, skip_special_tokens=True)
            stopped = any(s in raw for s in (stop or []))
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            finished = bool(len(tokens)) and (int(tokens[-1]) in eos or stopped)
            record.update(raw_response=raw, output_tokens=len(tokens), finished=finished,
                          device_map={k: str(v) for k, v in self.model.hf_device_map.items()})
            if not finished:
                raise ValueError('BLAgent generation exhausted the output token budget')
            answer = raw
            for marker in stop or []:
                answer = answer.split(marker, 1)[0]
            if not answer.strip():
                raise ValueError('BLAgent generated an empty response')
            record.update(status='ok', response=answer.strip())
            return record['response']
        except Exception as error:
            record.update(status='failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            write_json(target, record)

    def chat_completion(self, model_name, prompt, system_prompt, temperature):
        return {'content': self.complete([{'role': 'system', 'content': system_prompt},
                                         {'role': 'user', 'content': prompt}], temperature=temperature, stage='query')}


def agent_llm(generator, temperature, callbacks):
    from langchain_core.language_models.llms import LLM
    class AuditedLLM(LLM):
        @property
        def _llm_type(self):
            return 'blagent-local-qwen3'

        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            return generator.complete([{'role': 'user', 'content': prompt}],
                                      temperature=temperature, stop=stop, stage='SAS')
    return AuditedLLM(callbacks=callbacks)


def retrieve(documents, queries, embedder, chunk_cache):
    import numpy as np
    chunks = []
    for document in documents:
        file_chunks, _ = chunk_cache.get(document, 'blagent_ast')
        chunks.extend(file_chunks)
    vectors = embedder.encode([c['embedding_text'] for c in chunks])
    rankings, evidence, audits = [], [], []
    for query in queries:
        vector, audit = embedder.query(query)
        scores = vectors @ vector
        if not np.isfinite(scores).all():
            raise ValueError('Invalid retrieval scores')
        order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i]['file_path'], i))
        # Author retrieves 1000 chunks and max-pools to at most 100 files.
        paths = list(dict.fromkeys(chunks[i]['file_path'] for i in order[:1000]))[:100]
        rankings.append(paths)
        per_file = {}
        for i in order:
            path = chunks[i]['file_path']
            if len(per_file.setdefault(path, [])) < 5:
                per_file[path].append({'chunk_id': chunks[i]['chunk_id'], 'code': chunks[i]['code'],
                                       'score': float(scores[i]), 'start_line': chunks[i]['start_line'],
                                       'end_line': chunks[i]['end_line']})
        evidence.append(per_file)
        audits.append(audit)
    return rankings, evidence, audits, chunks


def localize(query, documents, modules, generator, embedder, chunk_cache, case, config):
    """Explicit input boundary: no patch, gold paths, or evaluation artifacts."""
    augmentor = modules['queries'].QueryAugmentor(generator, model_name='qwen3:8b')
    queries = [augmentor.augment(query, temperature=config['query_temperature'], type=i) for i in range(2)]
    if any(not q.strip() for q in queries):
        raise ValueError('Both BLAgent query transformations must succeed')
    retrieval_started = time.monotonic()
    if config.get('retrieval_lock'):
        # Separate inference workers share content-addressed caches. Serialize
        # short retrieval/cache writes, while the model server batches requests.
        with Path(config['retrieval_lock']).open('a') as retrieval_lock:
            fcntl.flock(retrieval_lock, fcntl.LOCK_EX)
            rankings, evidence, audits, chunks = retrieve(documents, queries, embedder, chunk_cache)
    else:
        rankings, evidence, audits, chunks = retrieve(documents, queries, embedder, chunk_cache)
    candidates = candidate_union(*rankings)
    write_json(case / 'retrieval.json', {'queries': queries, 'query_audits': audits, 'rankings': rankings,
                                       'candidates': candidates, 'chunks': chunks, 'evidence': evidence,
                                       'elapsed_including_lock_seconds': time.monotonic()-retrieval_started})
    agent_module = modules['agent']
    agent_module.OllamaLLM = lambda **kw: agent_llm(generator, kw['temperature'], kw['callbacks'])
    instance = agent_module.SWEBenchRankerAgent(make_structure(documents),
        {'problem_statement': query, 'retrieved_files_t0': rankings[0], 'retrieved_files_t1': rankings[1]},
        model_name='qwen3:8b', temperature=config['agent_temperature'])
    instance.agent.max_iterations = config['max_agent_iterations']
    instance.agent.return_intermediate_steps = True
    inspected = []
    original_tool = instance.tools[0].func
    def guarded_read(path):
        clean = path.replace('"', '').strip()
        if clean not in candidates:
            raise ValueError('Agent requested a file outside the retrieved candidates')
        value = original_tool(path)
        inspected.append({'file_path': clean, 'skeleton': value})
        write_json(case / 'tool_calls.json', inspected)
        return value
    instance.tools[0].func = guarded_read
    robust = config.get('recovery_protocol') == 'bounded-v2'
    if robust:
        from .blagent_recovery import guarded_tool
        instance.tools[0].func = guarded_tool(original_tool, candidates, inspected, [], case)
        if config.get('require_skeleton_before_final'):
            from .blagent_recovery import inspection_required_parser
            instance.agent.agent.output_parser = inspection_required_parser(
                instance.agent.agent.output_parser, inspected, case)
    # run() uses Chain.run(), which requires one output key; preserve the author's
    # executor and use tool/call audit files instead of adding a second output.
    instance.agent.return_intermediate_steps = False
    scores, usage = instance.run(max_files=15)
    if robust:
        from .blagent_recovery import recover_scores
        scores, phase1 = recover_scores(scores, query, candidates, inspected, generator, case, config, validate_scores)
    else:
        phase1 = validate_scores(scores, candidates)
    if not inspected:
        raise ValueError('SAS returned scores without any real skeleton inspection')
    write_json(case / 'phase1.json', {'scores': scores, 'ranking': phase1, 'author_token_usage': usage,
                                    'inspected_files': [v['file_path'] for v in inspected]})
    head = phase1[:5]
    sources = {d['file_path']: d['content'] for d in documents}
    contexts = {}
    for path in head:
        anchors = [c['code'] for c in evidence[0][path]]
        contexts[path] = modules['pruner'].CodePruner(sources[path], anchors).prune()
    aggregated = ''.join(f'### File: {path} ###\n{contexts[path]}\n\n' for path in head)
    prompt = modules['prompts'].EVIDENCE_ANCHORED_RERANKER_PROMPT.format(
        num_file_paths=len(head), problem_statement=query, aggregated_code_text=aggregated)
    write_json(case / 'phase2_context.json', {'query_type_for_anchors': 0, 'contexts': contexts})
    if robust:
        from .blagent_recovery import evidence_rerank
        ear_modules = {**modules, 'ear_skeletons': {p: agent_module.extract_signature_tree(sources[p]) for p in head}}
        reranked = evidence_rerank(query, head, contexts, evidence[0], ear_modules, generator, case, config, parse_ranking)
    else:
        response = generator.complete([{'role': 'system', 'content': 'You are a helpful assistant that reranks code files.'},
                                       {'role': 'user', 'content': prompt}], temperature=config['ear_temperature'], stage='EAR')
        reranked = parse_ranking(response, head)
    final = reranked + phase1[len(head):]
    return {'queries': queries, 'candidates': candidates, 'phase1_ranking': phase1, 'ranking': final,
            'model_calls': generator.calls, 'inspected_files': [v['file_path'] for v in inspected],
            'retrieved_files_t0': rankings[0], 'retrieved_files_t1': rankings[1]}


def run(project, config, output, *, check_only=False):
    from .ast_chunking import RoutingChunkCache
    base = json.loads((project / config['base_config']).read_text())
    validate_config(base)
    if base['dataset'] != 'swe_bench_formatted' or base['query_fields'] != ['body'] or base['extensions'] != ['.py'] or base['top_k'] != 10:
        raise ValueError('BLAgent adapter requires the shared SWE/Python/body-only/@10 protocol')
    selected, selection = select_reports(project, config, base, max_samples=5000 if config.get('selection_manifest') else 50)
    modules, author_hashes = author_modules(project, config)
    embedding_config = json.loads((project / config['semantic_config']).read_text())
    if check_only:
        for root in (project / config['generation_model'], project / embedding_config['model_path']):
            required_model_files(root)
        return {'selected_samples': len(selected), 'selection_sha256': selection['selection_sha256'],
                'author_revision': REVISION, 'author_files_sha256': author_hashes}
    folder = (project / output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if config.get('generation_backend') == 'llama_server':
        from .blagent_runtime import ServerGenerator
        generator = ServerGenerator(project, config)
    else:
        generator = Generator(project, config)
    embedder = LocalEmbedder(project, embedding_config)
    cache = RoutingChunkCache(project, embedder.tokenizer, embedding_config, embedder.fingerprint)
    implementation_files = ['blagent.py', 'ast_chunking.py', 'embeddings.py', 'chunking.py', 'languages.py',
                            'bm25.py', 'vector.py', 'batch.py', 'snapshot.py', 'model_files.py',
                            'blagent_recovery.py', 'blagent_runtime.py']
    identity = {'config': config, 'base': base, 'embedding': embedder.identity, 'generation': generator.identity,
                'selection_sha256': selection['selection_sha256'], 'author_revision': REVISION,
                'author_files_sha256': author_hashes,
                'implementation': {p: sha256_file(Path(__file__).with_name(p)) for p in implementation_files},
                'dependencies': {p: version(p) for p in ('langchain', 'langchain-core', 'libcst')}}
    provenance = folder / 'provenance.json'
    if provenance.exists():
        saved = json.loads(provenance.read_text())
        if saved['identity'] != identity:
            raise ValueError('BLAgent resume identity mismatch; choose a new output directory')
    else:
        saved = {'identity': identity, 'identity_sha256': digest(identity),
                 'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                 'variant': ('BLAgent author SAS/EAR + Qwen3-8B BF16 non-thinking + shared Qwen AST64 retrieval; '
                             + ('bounded-v2 recovery + llama.cpp two-slot backend; ' if config.get('recovery_protocol') else '')
                             + 'NOT paper reproduction')}
        write_json(provenance, saved)
        write_json(folder / 'effective_config.json', {'blagent': config, 'base': base, 'semantic': embedding_config})
        write_json(folder / 'selection.json', selection)
        if hasattr(generator, 'server_properties'):
            write_json(folder / 'server_properties.json', generator.server_properties)
    records = []
    def checkpoint():
        summary = aggregate_results(records, 10, len(selected))
        if summary['status'] == 'completed' and summary['failed_samples']:
            summary['status'] = 'completed_with_failures'
        summary.update(variant=saved['variant'], identity_sha256=saved['identity_sha256'])
        write_checkpoints(folder, records, summary)
        return summary
    try:
        summary = checkpoint()
        for position, sample in enumerate(selected, 1):
            row = sample['report']
            case = folder / 'cases' / f'{position:04d}'
            case.mkdir(parents=True, exist_ok=True)
            terminal = case / 'record.json'
            entry = {'position': position, 'dataset_row': sample['dataset_row'], 'repository': row['repo_name'],
                     'issue_id': row['issue_id'], 'before_fix_sha': row['before_fix_sha']}
            if terminal.exists():
                rec = json.loads(terminal.read_text())
                if any(rec.get(k) != v for k, v in entry.items()):
                    raise ValueError('Resume sample identity mismatch')
                if rec['status'] == 'ok' and sha256_file(folder / rec['result_file']) != rec['result_sha256']:
                    raise ValueError('BLAgent result checksum mismatch')
                records.append(rec); summary = checkpoint(); continue
            print(f'[{position}/{len(selected)}] BLAgent {row["repo_name"]} #{row["issue_id"]}', flush=True)
            started = time.monotonic()
            rec = dict(entry)
            generator.start_case(case)
            try:
                _, gold, query = prepare_report(row, {**base, 'repository': row['repo_name'], 'issue_id': row['issue_id']})
                source, manifest = get_snapshot(project, row['repo_name'], row['before_fix_sha'])
                docs, skipped = collect_files(source, manifest, base)
                inputs = {'query': query, 'candidate_files': [{'file_path': d['file_path'], 'sha256': d['sha256']} for d in docs],
                          'source_commit': row['before_fix_sha']}
                write_json(case / 'inputs.json', inputs)
                result = localize(query, docs, modules, generator, embedder, cache, case, config)
                expected = {d['file_path'] for d in docs}
                metrics = ranking_metrics(result['ranking'], gold, 10)
                result.update(**entry, **inputs, gold_files=gold, metrics=metrics,
                    phase1_metrics=ranking_metrics(result['phase1_ranking'], gold, 10),
                    missing_gold=sorted(set(gold) - expected), skipped_files=skipped)
                write_json(case / 'result.json', result)
                rec.update(status='ok', metrics=metrics, missing_gold=result['missing_gold'], indexed_files=len(docs),
                           result_file=(case / 'result.json').relative_to(folder).as_posix(), result_sha256=sha256_file(case / 'result.json'))
                print('  final ranking:', result['ranking'], flush=True)
            except Exception as error:
                rec.update(status='failed', error_type=type(error).__name__, error=str(error))
                if config.get('recovery_protocol') and (
                        isinstance(error, (ConnectionError, TimeoutError, OSError, RuntimeError))):
                    rec['infrastructure_failure'] = True
                write_json(case / 'error.json', {**rec, 'traceback': traceback.format_exc()})
                print('  FAILED:', type(error).__name__, str(error), flush=True)
            rec.update(elapsed_seconds=time.monotonic() - started, model_calls=generator.calls)
            write_json(terminal, rec)
            records.append(rec)
            summary = checkpoint()
            if rec.get('infrastructure_failure') and summary['pending_samples']:
                summary['status'] = 'stopped_infrastructure_failure'
                write_checkpoints(folder, records, summary)
                break
            # Infrastructure/model failures should not waste a full 500-case run.
            stop_count = config.get('stop_after_failures', 3)
            if stop_count and summary['pending_samples'] and len(records) >= stop_count and all(r['status'] == 'failed' for r in records[-stop_count:]):
                summary['status'] = 'stopped_after_three_consecutive_failures'
                write_checkpoints(folder, records, summary)
                break
        return folder, summary
    finally:
        embedder.close()
        lock.close()
