"""Actual warm-service request latency, all models resident, uncached queries/generation."""
import argparse
import os
from pathlib import Path
import random
import sqlite3
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bug_localization.batch import write_json
from bug_localization.embeddings import LocalEmbedder, digest
from bug_localization.reranking import LocalReranker, rerank, read_json
from bug_localization.chunk_ablation import select_evidence
from bug_localization.mixed_recall import mixed_candidates
from bug_localization.bm25 import collect_files, prepare_report, ranking_metrics
from bug_localization.vector import retrieve
from bug_localization.ast_chunking import RoutingChunkCache
from bug_localization.snapshot import sha256_file
from run_direct_hypothesis import Generator
from run_c3_supplement import configure_pipeline, now


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait', action='store_true')
    args = p.parse_args(); source, folder = args.source.resolve(), args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    selection = read_json(source / 'selection.json')['samples']
    # Choose before viewing outcomes. Fixed nine reports, three repetitions, alternating order.
    selected_positions = sorted(random.Random(20261112).sample(range(1, len(selection) + 1), min(9, len(selection))))
    protocol = {'source_protocol_sha256': sha256_file(source / 'protocol.json'), 'implementation_sha256': sha256_file(Path(__file__)),
        'positions': selected_positions, 'repetitions': 3, 'seed': 20261112, 'methods': ['A', 'C3'],
        'hardware_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'timing': 'Synchronized request wall time, all three models resident; warm snapshot/code-vector/index preparation excluded and reported separately; generation and query embedding caches bypassed; original report used for reranking; method order alternates; failures retained.',
        'currency_rate': None}
    if (folder / 'protocol.json').exists() and read_json(folder / 'protocol.json') != protocol:
        raise ValueError('Benchmark protocol changed')
    write_json(folder / 'protocol.json', protocol)
    if args.wait:
        write_json(folder / 'status.json', {'status': 'waiting_for_holdout', 'updated_at': now()})
        while True:
            status = read_json(source / 'status.json')
            if status['status'] == 'completed': break
            if status['status'] == 'failed':
                raise RuntimeError('Holdout failed; benchmark not started')
            time.sleep(20)
    import torch
    import numpy as np
    from elasticsearch import Elasticsearch
    import run_reranker500 as pipeline
    configure_pipeline(pipeline)
    config = read_json(source / 'protocol.json')['config']
    semantic, base, selected, _ = pipeline.context(config)
    write_json(folder / 'status.json', {'status': 'loading_models', 'updated_at': now()})
    start = time.perf_counter()
    encoder = LocalEmbedder(PROJECT, semantic); encoder.load_model()
    reranker = LocalReranker(PROJECT, read_json(PROJECT / config['reranker_config']))
    generator = Generator(read_json(PROJECT / config['generation_config']))
    generator.folder = folder / ('warmup_generation_' + str(time.time_ns())); generator.folder.mkdir(exist_ok=True)
    generator.generate('A helper returns an incorrect result when given an empty list. Expected an empty result.')
    reranker.score('Incorrect empty list handling.', ['def example(items):\n    return items'])
    encoder.query('Incorrect empty list handling.')
    torch.cuda.synchronize()
    write_json(folder / 'model_loading_and_warmup.json', {'seconds': time.perf_counter() - start,
        'generator': generator.identity, 'encoder': encoder.identity, 'reranker': reranker.identity,
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__})
    cache = RoutingChunkCache(PROJECT, encoder.tokenizer, semantic, encoder.fingerprint)
    records = read_json(folder / 'samples.json') if (folder / 'samples.json').exists() else []
    done = {(r['position'], r['repetition'], r['method']) for r in records}
    def query_uncached(body):
        database = encoder.database
        try:
            encoder.database = sqlite3.connect(':memory:')
            encoder.database.execute('CREATE TABLE vectors (key TEXT PRIMARY KEY, data BLOB, checksum TEXT)')
            return encoder.query(body)
        finally:
            encoder.database.close(); encoder.database = database
    try:
        with Elasticsearch(base['endpoint'], request_timeout=120) as client:
            if client.info()['cluster_name'] != base['cluster_name']:
                raise ValueError('Unexpected ES cluster')
            for position in selected_positions:
                meta = selection[position - 1]; row = selected[position - 1]['report']
                preparation_start = time.perf_counter()
                _, gold, body = prepare_report(row, {**base, 'repository': row['repo_name'], 'issue_id': row['issue_id']})
                root, manifest = pipeline.get_snapshot(PROJECT, row['repo_name'], row['before_fix_sha'])
                documents, _ = collect_files(root, manifest, base)
                chunks = read_json(source / 'new320/semantic/cases' / f'{position:03d}' / 'chunks.json')
                before = encoder.misses
                vectors = encoder.encode([c['embedding_text'] for c in chunks])
                if encoder.misses != before:
                    raise ValueError('Expected prepared code vectors; benchmark must not regenerate them')
                identity = {'repository': row['repo_name'], 'commit': row['before_fix_sha'], 'route': 'blagent_ast',
                    'model_fingerprint': encoder.fingerprint, **cache.index_identity(documents, 'blagent_ast')}
                fingerprint = digest({'identity': identity, 'chunks': [c['chunk_id'] for c in chunks],
                    'vectors_sha256': __import__('hashlib').sha256(vectors.astype('<f4').tobytes()).hexdigest()})
                index = 'bl-vector-blagent_ast-' + row['before_fix_sha'][:12] + '-' + fingerprint[:16]
                existed = client.indices.exists(index=index)
                try:
                    vector, _ = encoder.query(body)
                    prepared = retrieve(client, documents, chunks, vectors, vector, identity, 'max', semantic['aggregation_temperature'])
                    if prepared['index'] != index: raise ValueError('Unexpected prepared index')
                    preparation_seconds = time.perf_counter() - preparation_start
                    for repeat in range(1, 4):
                        for method in (['A', 'C3'] if (repeat + position) % 2 else ['C3', 'A']):
                            if (position, repeat, method) in done: continue
                            generator.folder = folder / 'generation' / f'{position:03d}_{repeat}_{time.time_ns()}'
                            generator.folder.mkdir(parents=True)
                            record = {**meta, 'repetition': repeat, 'method': method, 'preparation_seconds': preparation_seconds}
                            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
                            try:
                                generation = None
                                if method == 'C3':
                                    generation = generator.generate(row['body'])
                                    if generation['generation_error']:
                                        raise ValueError(generation['generation_error'])
                                after_generation = time.perf_counter()
                                q, qi = query_uncached(body)
                                original = retrieve(client, documents, chunks, vectors, q, identity, 'max', semantic['aggregation_temperature'])
                                hypothesis = original
                                if method == 'C3':
                                    qh, _ = query_uncached(generation['query'])
                                    hypothesis = retrieve(client, documents, chunks, vectors, qh, identity, 'max', semantic['aggregation_temperature'])
                                after_retrieval = time.perf_counter()
                                candidates = original['ranking'][:50] if method == 'A' else mixed_candidates(original['ranking'], hypothesis['ranking'], 'hypothesis')
                                head, chosen = select_evidence(candidates, original['ranking'], chunks, original['chunk_scores'],
                                    hypothesis['chunk_scores'], 'original' if method == 'A' else 'mixed')
                                query = qi['embedding_text'].split('\nQuery: ', 1)[1]
                                scored = reranker.score(query, [c['embedding_text'] for c in chosen])
                                ranking = rerank(head, chosen, scored)
                                torch.cuda.synchronize(); elapsed = time.perf_counter() - started
                                record.update(status='ok', request_wall_seconds=elapsed,
                                    generation_seconds=after_generation - started, retrieval_seconds=after_retrieval - after_generation,
                                    selection_and_rerank_seconds=time.perf_counter() - after_retrieval,
                                    generation_input_tokens=generation['input_tokens'] if generation else 0,
                                    generation_output_tokens=generation['output_tokens'] if generation else 0,
                                    reranker_input_tokens=sum(r['input_tokens'] for r in scored),
                                    metrics=ranking_metrics([r['file_path'] for r in ranking], gold, 10))
                            except Exception as error:
                                torch.cuda.synchronize()
                                record.update(status='failed', request_wall_seconds=time.perf_counter() - started,
                                    error=str(error), traceback=traceback.format_exc())
                            record['peak_allocated_gpu_bytes'] = torch.cuda.max_memory_allocated()
                            records.append(record); write_json(folder / 'samples.json', records)
                            write_json(folder / 'status.json', {'status': 'running', 'processed': len(records),
                                'selected_attempts': len(selected_positions) * 6, 'updated_at': now()})
                            print(position, repeat, method, record['status'], flush=True)
                finally:
                    if not existed and client.indices.exists(index=index):
                        ownership = client.indices.get_mapping(index=index)[index]['mappings'].get('_meta', {})
                        if ownership.get('project') != 'bug-localization' or ownership.get('fingerprint') != fingerprint:
                            raise ValueError('Unexpected index ownership; refusing cleanup')
                        client.indices.delete(index=index)
        summaries = {}
        for method in ('A', 'C3'):
            rows = [r for r in records if r['method'] == method]
            ts = [r['request_wall_seconds'] for r in rows if r['status'] == 'ok']
            summaries[method] = {'attempts': len(rows), 'successes': len(ts), 'failures': len(rows) - len(ts),
                'mean_seconds': float(np.mean(ts)) if ts else None, 'p50_seconds': float(np.median(ts)) if ts else None,
                'p95_seconds': float(np.quantile(ts, .95)) if ts else None,
                'total_measured_device_wall_hours_including_failures': sum(r['request_wall_seconds'] for r in rows) / 3600,
                'currency_cost': None}
        write_json(folder / 'summary.json', {'methods': summaries, 'protocol': protocol,
            'note': 'Repeated latency samples are not independent accuracy observations. Currency conversion needs an explicit hourly device rate; electricity additionally needs energy measurement.'})
        write_json(folder / 'status.json', {'status': 'completed', 'updated_at': now()})
    finally:
        encoder.close()

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
        write_json(output / 'status.json', {'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc(), 'updated_at': now()})
        raise
