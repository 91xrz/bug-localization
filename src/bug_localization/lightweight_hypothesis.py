"""CPU-only post-processing of functional-query retrieval; no label access."""
import builtins
from collections import Counter
import keyword
import math
import re

WORD = re.compile(r'[A-Za-z_][A-Za-z_0-9]*')
DEFINITION = re.compile(r'^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z_0-9]*)\b', re.M)
IGNORE = set(keyword.kwlist) | set(dir(builtins)) | {'self', 'cls', 'args', 'kwargs', 'init', 'main'}


def definitions(text):
    """Lightweight declaration scanner, not an execution or full Python parser."""
    return {s for s in DEFINITION.findall(text) if len(s) >= 3 and s not in IGNORE
            and not (s.startswith('__') and s.endswith('__'))}


def clean_report(text):
    text = re.sub(r'<!--[\s\S]*?-->', ' ', text)
    return re.sub(r'https?://[^\s<>]+', ' ', text)


def anchors(report, symbols):
    """File scores in [0,1], plus auditable matches. Inputs contain no labels."""
    report = clean_report(report)
    words = {w for w in WORD.findall(report) if len(w) >= 3 and w not in IGNORE}
    df = Counter(w for ss in symbols.values() for w in ss if w in words)
    weights = {w: math.log((len(symbols) + 1) / (n + 1)) for w, n in df.items()}
    total = sum(weights.values())
    out, matches = {}, {}
    for path, ss in symbols.items():
        matched = sorted(ss & weights.keys())
        value = sum(weights[w] for w in matched) / total if total else 0.
        module = path[:-3].replace('/', '.') if path.endswith('.py') else path
        if module.endswith('.__init__'):
            module = module[:-9]
        # Require a multi-component module; bare common words are not paths.
        explicit = path in report or ('.' in module and re.search(
            r'(?<![\w.])' + re.escape(module) + r'(?![\w])', report) is not None)
        out[path] = 1. if explicit else value
        matches[path] = {'identifiers': matched, 'explicit_path_or_module': explicit}
    return out, matches


def normalize(rows):
    high = rows[0]['score']
    low = rows[min(49, len(rows)-1)]['score']
    if high <= low:
        return {r['file_path']: 0. for r in rows}
    return {r['file_path']: max(0., min(1., (r['score']-low)/(high-low))) for r in rows}


def primary_pool(a, b, count=40, limit=50):
    selected = [r['file_path'] for r in b[:count]]
    for row in a:
        if len(selected) >= limit:
            break
        if row['file_path'] not in selected:
            selected.append(row['file_path'])
    return selected


def balanced_pool(a, b, limit=50):
    selected = []
    for x, y in zip(b, a):
        for row in (x, y):
            if row['file_path'] not in selected:
                selected.append(row['file_path'])
            if len(selected) == limit:
                return selected
    return selected


def rank_variants(a, b, anchor, alphas, weights):
    ar = {r['file_path']: r['rank'] for r in a}
    br = {r['file_path']: r['rank'] for r in b}
    assert ar.keys() == br.keys()
    an, bn = normalize(a), normalize(b)
    union = {r['file_path'] for rows in (a[:50], b[:50]) for r in rows}
    output = {'A': [r['file_path'] for r in a[:50]], 'B': [r['file_path'] for r in b[:50]]}
    pool = primary_pool(a, b)
    output['current_RRF60'] = sorted(pool, key=lambda p: (-(1/(60+ar[p])+1/(60+br[p])), p))
    output['balanced_pool_Border'] = sorted(balanced_pool(a, b), key=br.get)
    # Diagnostic larger-budget bound. Not a deployable fixed50 comparator.
    output['union_ceiling_Border'] = sorted(union, key=br.get)
    for alpha in alphas:
        for weight in weights:
            if alpha == 0 and weight == 0:
                continue
            score = {p: (1-alpha)*bn[p] + alpha*an[p] + weight*anchor[p] for p in union}
            rank = sorted(union, key=lambda p: (-score[p], br[p], ar[p], p))
            output[f'score_a{alpha:g}_e{weight:g}'] = rank[:50]
            if alpha == 0 and weight > 0:
                output[f'B50_anchor_e{weight:g}'] = [p for p in rank if br[p] <= 50]
    return output
