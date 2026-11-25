"""Label-free, deterministic experimental improvements of original-report retrieval."""
from collections import defaultdict
import math

import numpy as np

from .lightweight_hypothesis import anchors, normalize


def rank(scores, original=None, limit=50):
    if not all(math.isfinite(float(s)) for s in scores.values()):
        raise ValueError('Nonfinite file score')
    return sorted(scores, key=lambda p: (-scores[p], original[p] if original else 0, p))[:limit]


def aggregation_scores(chunk_scores, config):
    grouped = defaultdict(list)
    for row in chunk_scores:
        grouped[row['file_path']].append(float(row['cosine_similarity']))
    output = {'A': {}}
    for path, values in grouped.items():
        a = np.sort(np.asarray(values, dtype=np.float64))[::-1]
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite chunk similarity')
        m = float(a[0])
        output['A'][path] = m
        for k in config['aggregation']['top_mean_k']:
            output.setdefault(f'top{k}_mean', {})[path] = float(a[:k].mean())
        for weight in config['aggregation']['max_top3_mix']:
            output.setdefault(f'max_top3_w{weight:g}', {})[path] = (1-weight)*m + weight*float(a[:3].mean())
        for tau in config['aggregation']['lse_temperature']:
            output.setdefault(f'lse_t{tau:g}', {})[path] = m + tau*math.log(float(np.exp((a-m)/tau).mean()))
        for penalty in config['length_penalty']:
            output.setdefault(f'length_p{penalty:g}', {})[path] = m - penalty*math.log(len(a))
    return output


def grounded_scores(rows, body, symbols, weight, depth):
    pool = rows[:depth]
    paths = [r['file_path'] for r in pool]
    # Normalization uses the first 50 scores, including for the deeper pool.
    evidence, matches = anchors(body, {p: symbols[p] for p in paths})
    base = normalize(rows)
    return {p: base[p]+weight*evidence[p] for p in paths}, evidence, matches


def feedback_scores(query, vectors, feedback_files, beta):
    q = np.asarray(query, dtype=np.float64)
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2 or not len(v) or v.shape[1:] != q.shape:
        raise ValueError('Invalid feedback vector shapes')
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError('Nonfinite feedback vectors')
    changed = (1-beta)*q + beta*v[:feedback_files].mean(axis=0)
    norm = np.linalg.norm(changed)
    if norm < 1e-12:
        raise ValueError('Degenerate feedback query')
    return v @ (changed/norm)


def family(name):
    if name == 'A':
        return 'baseline'
    if name.startswith('length_'):
        return 'length'
    if name.startswith('anchor_'):
        return 'anchor'
    if name.startswith('prf_'):
        return 'prf'
    if name.startswith('combo_'):
        return 'combination'
    return 'aggregation'
