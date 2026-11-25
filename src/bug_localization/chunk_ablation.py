"""Select original, hypothesis, or mixed evidence for a fixed file pool."""
from collections import defaultdict

from .reranking import select_chunks


def select_evidence(candidates, original_ranking, chunks, original_scores, hypothesis_scores, policy):
    if policy not in ('original', 'hypothesis', 'mixed'):
        raise ValueError('Unknown chunk selection policy')
    paths = {r['file_path'] for r in candidates}
    if len(paths) != len(candidates):
        raise ValueError('Duplicate candidate files')
    head = [r for r in original_ranking if r['file_path'] in paths]
    if len(head) != len(paths):
        raise ValueError('Candidate missing from original ranking')
    ordered = head + [r for r in original_ranking if r['file_path'] not in paths]
    ordered = [{**row, 'rank': i} for i, row in enumerate(ordered, 1)]
    # Both calls validate complete, finite, unique chunk-score inventories.
    selected = {name: select_chunks(ordered, chunks, scores, len(head), 3)
                for name, scores in (('original', original_scores), ('hypothesis', hypothesis_scores))}
    groups = {name: defaultdict(list) for name in selected}
    lookup_scores = {name: {s['chunk_id']: s['cosine_similarity'] for s in scores}
                     for name, scores in (('original', original_scores), ('hypothesis', hypothesis_scores))}
    for name, rows in selected.items():
        for row in rows:
            groups[name][row['file_path']].append(row)
    chosen = []
    for file in head:
        path = file['file_path']
        if policy == 'mixed':
            options = [('original', r) for r in groups['original'][path][:1]]
            options += [('hypothesis', r) for r in groups['hypothesis'][path]]
        else:
            options = [(policy, r) for r in groups[policy][path]]
        seen = set()
        for route, row in options:
            if row['chunk_id'] in seen:
                continue
            seen.add(row['chunk_id'])
            chosen.append({**row, 'selected_by': route,
                'original_similarity': lookup_scores['original'][row['chunk_id']],
                'hypothesis_similarity': lookup_scores['hypothesis'][row['chunk_id']]})
            if len(seen) == 3:
                break
    return ordered[:len(head)], chosen
