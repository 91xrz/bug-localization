"""Local Qwen3 reranking of saved semantic candidates, with paired evaluation."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import csv
import json
import math
from pathlib import Path
import time
import traceback

from .batch import aggregate_results, write_json
from .bm25 import ranking_metrics
from .embeddings import digest
from .snapshot import sha256_file

PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
          'and the Instruct provided. Note that the answer can only be "yes" or "no".'
          '<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_report_experiment_case(source, record):
    """Adapt only the original-report q0 control; never consume enhanced queries."""
    case = source / 'cases' / f"{record['position']:03d}"
    result_path, retrieval_path = case / 'q0_result.json', case / 'q0_retrieval.json'
    saved, retrieval = read_json(result_path), read_json(retrieval_path)
    if saved['ranking'] != retrieval['ranking'] or saved['metrics'] != record['metrics']:
        raise ValueError('Original-report result differs from saved retrieval/metrics')
    candidates_path = case / 'candidates.json'
    candidates = read_json(candidates_path)['files']
    result = {**record, **retrieval, **saved, 'candidate_fingerprint': digest(candidates)}
    query = {'body': retrieval['query'], **retrieval['query_info']}
    chunks_path = case / 'chunks.json'
    chunks = read_json(chunks_path)
    if any(chunk['route'] != 'blagent_ast' for chunk in chunks):
        raise ValueError('Report-experiment adapter requires AST chunks')
    scores = {'scores': retrieval['chunk_scores'], 'sha256': retrieval['chunk_scores_sha256']}
    return case, result, candidates, query, chunks, scores, (result_path, retrieval_path, candidates_path, chunks_path)


def load_direct_hypothesis_case(source, assets, record):
    """Select candidates with hypothesis scores, but rerank using only the original report."""
    position = f"{record['position']:03d}"
    case, asset_case = source / 'cases' / position, assets / 'cases' / position
    result_path, retrieval_path = case / 'direct_result.json', case / 'direct_retrieval.json'
    query_path = source / (position + '_queries.json')
    saved, retrieval, generated = read_json(result_path), read_json(retrieval_path), read_json(query_path)
    if generated['generation_error'] is not None or not generated['query'].strip():
        raise ValueError('Direct hypothesis is empty or generation failed')
    if saved['ranking'] != retrieval['ranking'] or saved['metrics'] != record['metrics']:
        raise ValueError('Direct hypothesis ranking/metrics differ from saved retrieval')
    gold_path = asset_case / 'q0_result.json'
    if saved['ground_truth'] != read_json(gold_path)['ground_truth']:
        raise ValueError('Direct hypothesis labels differ from original-report control')
    candidates_path, chunks_path = asset_case / 'candidates.json', asset_case / 'chunks.json'
    candidates, chunks = read_json(candidates_path)['files'], read_json(chunks_path)
    if any(chunk['route'] != 'blagent_ast' for chunk in chunks):
        raise ValueError('Direct hypothesis adapter requires AST chunks')
    original_query_path = asset_case / 'q0_retrieval.json'
    original = read_json(original_query_path)
    result = {**record, **retrieval, **saved, 'candidate_fingerprint': digest(candidates),
              'retrieval_query_info': retrieval['query_info']}
    query = {'body': original['query'], **original['query_info']}
    scores = {'scores': retrieval['chunk_scores'], 'sha256': retrieval['chunk_scores_sha256']}
    return case, result, candidates, query, chunks, scores, (
        result_path, retrieval_path, query_path, candidates_path, chunks_path, gold_path, original_query_path)


def validate_config(config):
    for name in ('top_n_files', 'chunks_per_file', 'batch_size', 'max_tokens', 'evaluation_k'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(name + ' must be a positive integer')
    if config['evaluation_k'] < 10 or config['top_n_files'] < config['evaluation_k']:
        raise ValueError('Require top_n_files >= evaluation_k >= 10')
    if not config['instruction'].strip():
        raise ValueError('instruction cannot be empty')


class LocalReranker:
    """Official yes/no prompt; no generation, downloads, or remote model code."""

    def __init__(self, project, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        validate_config(config)
        self.config = config
        path = (project / config['model_path']).resolve()
        device = torch.device(config['device'])
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; explicitly select cpu if needed')
        dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                       trust_remote_code=False, padding_side='left')
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.no_id, self.yes_id = [self.tokenizer.encode(word, add_special_tokens=False)
                                  for word in ('no', 'yes')]
        if len(self.no_id) != 1 or len(self.yes_id) != 1 or self.no_id == self.yes_id:
            raise ValueError('Expected distinct single-token yes/no labels')
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, dtype=dtype,
            attn_implementation='sdpa').to(device).eval()
        if config['max_tokens'] > self.model.config.max_position_embeddings:
            raise ValueError('max_tokens exceeds model context')
        self.identity = {
            'path': str(path), 'files_sha256': {name: sha256_file(path / name) for name in
                ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')},
            'torch': version('torch'), 'transformers': version('transformers'),
            'dtype': str(dtype), 'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'score': 'logit(yes) - logit(no); probability = sigmoid(score)',
            'prompt_prefix': PREFIX, 'prompt_suffix': SUFFIX,
        }

    def score(self, query, documents):
        import torch

        rows = []
        for start in range(0, len(documents), self.config['batch_size']):
            texts = [f"<Instruct>: {self.config['instruction']}\n<Query>: {query}\n<Document>: {doc}"
                     for doc in documents[start:start + self.config['batch_size']]]
            ids = self.tokenizer(texts, add_special_tokens=False, padding=False,
                                 truncation=False)['input_ids']
            ids = [self.prefix + tokens + self.suffix for tokens in ids]
            # Never silently remove code or expand the baseline report context.
            if any(len(tokens) > self.config['max_tokens'] for tokens in ids):
                raise ValueError('Reranker input exceeds max_tokens; increase budget, no silent truncation')
            inputs = self.tokenizer.pad({'input_ids': ids}, padding=True, return_tensors='pt')
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                scores = logits[:, self.yes_id[0]] - logits[:, self.no_id[0]]
                probabilities = torch.sigmoid(scores)
            for score, probability, tokens in zip(scores.tolist(), probabilities.tolist(), ids):
                if not math.isfinite(score):
                    raise ValueError('Non-finite reranker score')
                rows.append({'score': score, 'probability': probability, 'input_tokens': len(tokens)})
        return rows


def select_chunks(ranking, chunks, scores, top_n, per_file):
    paths = [row['file_path'] for row in ranking]
    if len(set(paths)) != len(paths) or [r['rank'] for r in ranking] != list(range(1, len(paths) + 1)):
        raise ValueError('Expected unique, consecutive baseline file ranking')
    lookup = {chunk['chunk_id']: chunk for chunk in chunks}
    if len(lookup) != len(chunks) or len(scores) != len(chunks):
        raise ValueError('Duplicate or incomplete chunk inventory')
    grouped, seen = defaultdict(list), set()
    for row in scores:
        chunk_id = row['chunk_id']
        if chunk_id not in lookup or chunk_id in seen or lookup[chunk_id]['file_path'] != row['file_path']:
            raise ValueError('Chunk score inventory mismatch')
        if not math.isfinite(row['cosine_similarity']):
            raise ValueError('Non-finite semantic score')
        seen.add(chunk_id)
        grouped[row['file_path']].append(row)
    if set(grouped) != set(paths):
        raise ValueError('Chunk and file inventories differ')
    selected = []
    for path in paths[:top_n]:
        best = sorted(grouped[path], key=lambda row: (-row['cosine_similarity'], row['chunk_id']))[:per_file]
        selected.extend({**lookup[row['chunk_id']], 'semantic_similarity': row['cosine_similarity']} for row in best)
    return selected


def rerank(ranking, selected, scored):
    if len(selected) != len(scored):
        raise ValueError('Incomplete reranker scores')
    best = {}
    for chunk, score in zip(selected, scored):
        if not math.isfinite(score['score']):
            raise ValueError('Non-finite reranker score')
        path = chunk['file_path']
        if path not in best or score['score'] > best[path]['score']:
            best[path] = {**score, 'best_chunk_id': chunk['chunk_id']}
    head, tail = [], []
    for row in ranking:
        path = row['file_path']
        item = {'file_path': path, 'semantic_rank': row['rank'], 'semantic_score': row['score'],
                'reranked': path in best, 'score': None}
        if path in best:
            head.append({**item, **best[path]})
        else:
            tail.append(item)
    head.sort(key=lambda row: (-row['score'], row['semantic_rank']))
    return [{**row, 'rank': rank} for rank, row in enumerate(head + tail, 1)]


def validate_resume(folder, source, config, routes, samples):
    saved = read_json(folder / 'effective_config.json')
    if saved['reranker'] != config or Path(saved['source_run']).resolve() != source.resolve() or saved['routes'] != routes:
        raise ValueError('Resume must keep source, routes and reranker settings unchanged')
    if read_json(folder / 'selection.json')['samples'] != samples:
        raise ValueError('Resume selection changed')
    for name, checksum in saved['source_metadata_sha256'].items():
        if sha256_file(source / name) != checksum:
            raise ValueError('Resume source metadata changed')
    records = read_json(folder / 'samples.json')
    if set(records) != set(routes):
        raise ValueError('Resume routes changed')
    counts = set()
    for pair in records.values():
        for name in ('baseline', 'reranked'):
            counts.add(len(pair[name]))
            for sample, record in zip(samples, pair[name]):
                if any(sample[key] != record[key] for key in ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha')):
                    raise ValueError('Resume checkpoint is not a prefix of the selection')
    if len(counts) != 1 or next(iter(counts)) > len(samples):
        raise ValueError('Resume requires complete paired sample checkpoints across all routes')
    return saved, records, next(iter(counts))


def run_reranking(project, source, config, routes, limit=None, resume_folder=None, output_folder=None):
    validate_config(config)
    source = (project / source).resolve()
    selection = read_json(source / 'selection.json')
    direct_hypothesis = config.get('source_query_route') == 'direct_hypothesis'
    assets = (project / config['semantic_assets_run']).resolve() if direct_hypothesis else source
    effective = read_json(assets / 'effective_config.json')
    metadata_paths = [source / 'selection.json', assets / 'effective_config.json']
    if direct_hypothesis:
        if config.get('reranker_query_source') != 'original_report':
            raise ValueError('Hypothesis retrieval requires an explicit original_report reranker query; do not resume legacy hypothesis-query runs')
        if selection['samples'] != read_json(assets / 'selection.json')['samples']:
            raise ValueError('Direct hypothesis and semantic assets have different selections')
        audit = read_json(source / 'audit.json')
        if not all(audit[key] for key in ('selection_verified', 'embedding_identity_matches', 'candidate_and_chunk_inputs_match')):
            raise ValueError('Direct hypothesis source was not verified against semantic assets')
        metadata_paths.extend([assets / 'selection.json', source / 'samples.json', source / 'audit.json'])
    samples = selection['samples']
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError('limit must be positive')
        samples = samples[:limit]
    if not samples or not routes or len(set(routes)) != len(routes):
        raise ValueError('Require samples and unique routes')
    report_experiment = not (source / 'blagent_ast/samples.json').exists() and (source / 'samples.json').exists()
    if report_experiment:
        if routes != ['blagent_ast'] or effective['semantic']['chunking_routes'] != ['blagent_ast']:
            raise ValueError('Report-experiment input supports only the original-report AST baseline')
        source_records = {'blagent_ast': read_json(source / 'samples.json')['direct_hypothesis' if direct_hypothesis else 'q0']}
    else:
        source_records = {route: read_json(source / route / 'samples.json') for route in routes}
    for records in source_records.values():
        if len(records) != len(selection['samples']):
            raise ValueError('Source run is incomplete')
        for sample, record in zip(selection['samples'], records):
            if any(sample[key] != record[key] for key in ('position', 'dataset_row', 'repository', 'issue_id', 'before_fix_sha')):
                raise ValueError('Source sample order/identity mismatch')
    if resume_folder is not None:
        folder = (project / resume_folder).resolve()
        saved, records, start_index = validate_resume(folder, source, config, routes, samples)
    else:
        folder = (project / output_folder if output_folder is not None else
                  project / 'results/reranking' / datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f'))
        folder.mkdir(parents=True)
        write_json(folder / 'selection.json', {**selection, 'samples': samples, 'selection_sha256': digest(samples),
                                             'source_selection_sha256': selection['selection_sha256']})
        records = {route: {'baseline': [], 'reranked': []} for route in routes}
        start_index = 0
    model = LocalReranker(project, config)
    if resume_folder is not None:
        if saved['model'] != model.identity:
            raise ValueError('Resume model identity/environment changed')
        history_path = folder / 'resume_history.json'
        history = read_json(history_path) if history_path.exists() else []
        history.append({'time': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'completed_samples': start_index,
                        'implementation_sha256': sha256_file(Path(__file__))})
        write_json(history_path, history)
    else:
        write_json(folder / 'effective_config.json', {
        'reranker': config, 'model': model.identity, 'source_run': str(source), 'source_config': effective,
        'routes': routes, 'source_query_route': 'direct_hypothesis' if direct_hypothesis else ('q0' if report_experiment else 'original'),
        'implementation_sha256': sha256_file(Path(__file__)),
        'source_metadata_sha256': {str(path): sha256_file(path) for path in metadata_paths},
        'reranker_query_source': 'original_report',
        'query_policy': 'Original report as seen by the original-report semantic encoder, without embedding instruction wrapper',
        'file_score': 'Max reranker logit difference over semantic top chunks; ties keep semantic rank',
        'tail_policy': 'Unselected files appended in original order; score=null (not comparable)',
    })
    k = config['evaluation_k']

    def checkpoint():
        summary = {'source_run': str(source), 'selected_samples': len(samples), 'evaluation_k': k, 'routes': {}}
        for route, pair in records.items():
            stats = {name: aggregate_results(rows, k, len(samples)) for name, rows in pair.items()}
            if all(stats[name]['metrics'] is not None for name in stats):
                stats['delta'] = {key: stats['reranked']['metrics'][key] - value
                                  for key, value in stats['baseline']['metrics'].items()}
            changes = [b['metrics']['average_precision_at_k'] - a['metrics']['average_precision_at_k']
                       for a, b in zip(pair['baseline'], pair['reranked']) if a['status'] == b['status'] == 'ok']
            stats['paired_ap'] = {'wins': sum(x > 1e-12 for x in changes), 'ties': sum(abs(x) <= 1e-12 for x in changes),
                                  'losses': sum(x < -1e-12 for x in changes), 'paired_successes': len(changes)}
            stats['reranking_seconds'] = sum(row.get('elapsed_seconds', 0) for row in pair['reranked'])
            summary['routes'][route] = stats
        complete = all(len(pair['reranked']) == len(samples) for pair in records.values())
        failed = any(row['status'] != 'ok' for pair in records.values() for row in pair['reranked'])
        summary['status'] = ('completed_with_failures' if failed else 'completed') if complete else 'running'
        write_json(folder / 'samples.json', records)
        write_json(folder / 'summary.json', summary)
        with (folder / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['route', 'variant', 'successful_samples', 'failed_samples',
                f'MAP@{k}', f'MRR@{k}', 'Top-1', 'Top-5', 'Top-10', f'MeanRecall@{k}'])
            writer.writeheader()
            for route, stats in summary['routes'].items():
                for name in ('baseline', 'reranked'):
                    writer.writerow({'route': route, 'variant': name,
                        **{key: stats[name][key] for key in ('successful_samples', 'failed_samples')},
                        **(stats[name]['metrics'] or {})})
        return summary

    summary = checkpoint()
    print('Reranking results: ' + str(folder), flush=True)
    for index, sample in enumerate(samples):
        if index < start_index:
            continue
        for route in routes:
            started = time.monotonic()
            record = source_records[route][index]
            baseline = dict(record)
            try:
                if record['status'] != 'ok':
                    raise ValueError('Source semantic retrieval failed: ' + record.get('error', 'unknown'))
                if direct_hypothesis:
                    case, result, candidates, query, chunks, scores, input_paths = load_direct_hypothesis_case(source, assets, record)
                elif report_experiment:
                    case, result, candidates, query, chunks, scores, input_paths = load_report_experiment_case(source, record)
                else:
                    result_path = source / record['result_file']
                    case = result_path.parent
                    result = read_json(result_path)
                    candidates = read_json(case / 'candidates.json')['files']
                    query = read_json(case / 'query.json')
                    chunks_path = case / (route + '_chunks.json')
                    chunks = read_json(chunks_path)['chunks']
                    scores_path = source / result['chunk_scores_file']
                    scores = read_json(scores_path)
                    input_paths = (result_path, chunks_path, scores_path, case / 'query.json', case / 'candidates.json')
                for key in ('repository', 'issue_id', 'before_fix_sha'):
                    if result[key] != sample[key]:
                        raise ValueError('Result identity mismatch: ' + key)
                baseline['metrics'] = ranking_metrics([r['file_path'] for r in result['ranking']], result['ground_truth'], k)
                if digest(candidates) != result['candidate_fingerprint'] or {r['file_path'] for r in result['ranking']} != {r['file_path'] for r in candidates}:
                    raise ValueError('Candidate fingerprint/inventory mismatch')
                wrapper = 'Instruct: ' + effective['semantic']['query_instruction'] + '\nQuery: '
                if not query['embedding_text'].startswith(wrapper):
                    raise ValueError('Unknown semantic query format')
                query_text = query['embedding_text'][len(wrapper):]
                if not query['body'].startswith(query_text):
                    raise ValueError('Semantic query differs from saved query text')
                if digest(scores['scores']) != scores['sha256'] or scores['sha256'] != result['chunk_scores_sha256']:
                    raise ValueError('Semantic scores checksum mismatch')
                chosen = select_chunks(result['ranking'], chunks, scores['scores'],
                                       config['top_n_files'], config['chunks_per_file'])
                scored = model.score(query_text, [chunk['embedding_text'] for chunk in chosen])
                ranking = rerank(result['ranking'], chosen, scored)
                metrics = ranking_metrics([row['file_path'] for row in ranking], result['ground_truth'], k)
                target = folder / 'cases' / case.name
                target.mkdir(parents=True, exist_ok=True)
                output = target / (route + '_result.json')
                write_json(output, {**sample, 'route': route, 'ground_truth': result['ground_truth'],
                    'missing_gold': result['missing_gold'], 'candidate_fingerprint': result['candidate_fingerprint'],
                    'query': query_text, 'query_truncated_by_semantic_encoder': query['truncated'],
                    'reranker_query_source': 'original_report',
                    'reranker_query_file': str((assets / 'cases' / case.name / 'q0_retrieval.json').resolve()) if direct_hypothesis else None,
                    'retrieval_query_info': result.get('retrieval_query_info'),
                    'baseline_metrics': baseline['metrics'], 'metrics': metrics, 'ranking': ranking,
                    'source_files_sha256': {str(p.resolve()): sha256_file(p) for p in input_paths},
                    'scored_chunks': [{**{key: chunk[key] for key in ('file_path', 'chunk_id', 'semantic_similarity', 'embedding_text')}, **score}
                                      for chunk, score in zip(chosen, scored)]})
                updated = {**sample, 'status': 'ok', 'metrics': metrics, 'missing_gold': result['missing_gold'],
                           'scored_pairs': len(scored), 'result_file': str(output.relative_to(folder))}
                print(f"[{index + 1}/{len(samples)}] {route} {sample['repository']} #{sample['issue_id']}: "
                      f"AP@{k} {baseline['metrics']['average_precision_at_k']:.4f} -> {metrics['average_precision_at_k']:.4f}", flush=True)
            except Exception as error:
                updated = {**sample, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                           'traceback': traceback.format_exc()}
                print(f"[{index + 1}/{len(samples)}] {route} FAILED: {error}", flush=True)
            updated['elapsed_seconds'] = time.monotonic() - started
            records[route]['baseline'].append(baseline)
            records[route]['reranked'].append(updated)
            summary = checkpoint()
    return folder, summary
