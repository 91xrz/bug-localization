"""Deterministic primary-40 / secondary-padding file recall, without label access."""
from .reranking import select_chunks


def mixed_candidates(original, hypothesis, primary, primary_count=40, total=50):
    if primary not in ('original', 'hypothesis') or not 0 < primary_count <= total:
        raise ValueError('Invalid mixed recall policy')
    routes = {'original': original, 'hypothesis': hypothesis}
    for rows in routes.values():
        paths = [r['file_path'] for r in rows]
        if len(set(paths)) != len(paths) or [r['rank'] for r in rows] != list(range(1, len(rows) + 1)):
            raise ValueError('Require unique, consecutive complete rankings')
    ranks = {name: {r['file_path']: r['rank'] for r in rows} for name, rows in routes.items()}
    if set(ranks['original']) != set(ranks['hypothesis']):
        raise ValueError('Recall routes must have the same file inventory')
    secondary = 'original' if primary == 'hypothesis' else 'hypothesis'
    selected, seen = [], set()
    for route, rows in ((primary, routes[primary][:primary_count]), (secondary, routes[secondary])):
        for row in rows:
            path = row['file_path']
            if path in seen:
                continue
            if len(selected) == total:
                break
            seen.add(path)
            selected.append({'file_path': path, 'candidate_rank': len(selected) + 1,
                             'introduced_by': route, 'is_primary': route == primary,
                             'original_rank': ranks['original'][path],
                             'hypothesis_rank': ranks['hypothesis'][path]})
    if len(selected) != min(total, len(original)):
        raise ValueError('Mixed recall did not fill the candidate pool')
    return selected


def original_chunks(candidates, original, chunks, scores, per_file=3):
    """Fix evidence and tie order to original retrieval for both C and D."""
    paths = {r['file_path'] for r in candidates}
    head = [r for r in original if r['file_path'] in paths]
    if len(head) != len(candidates):
        raise ValueError('Candidates missing from original ranking')
    ordered = head + [r for r in original if r['file_path'] not in paths]
    ordered = [{**r, 'rank': i} for i, r in enumerate(ordered, 1)]
    chosen = select_chunks(ordered, chunks, scores, len(head), per_file)
    return ordered[:len(head)], chosen
