"""Process-isolated file chunking with deterministic result ordering."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path

_cache = None


def _initialize(project, config, fingerprint):
    global _cache
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    from .dependency_context import DependencyRoutingCache
    path = (Path(project) / config['model_path']).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                               trust_remote_code=False, padding_side='left')
    tokenizer.model_max_length = 10**9
    _cache = DependencyRoutingCache(Path(project), tokenizer, config, fingerprint)


def _chunk(item):
    document, route = item
    hits, misses = _cache.hits, _cache.misses
    chunks, parents = _cache.get(document, route)
    return chunks, parents, _cache.hits - hits, _cache.misses - misses


def create_pool(project, config, fingerprint):
    return ProcessPoolExecutor(max_workers=config.get('chunk_workers', 1),
                               mp_context=multiprocessing.get_context('spawn'),
                               initializer=_initialize, initargs=(str(project), config, fingerprint))


def chunk_documents(pool, documents, route):
    # map preserves input order, independently of worker completion order.
    return pool.map(_chunk, ((document, route) for document in documents), chunksize=1)
