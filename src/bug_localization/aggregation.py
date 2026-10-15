"""File rankings from one complete inventory of raw chunk cosine similarities."""

from collections import defaultdict
import math

AGGREGATIONS = ('max', 'top3_mean', 'normalized_logsumexp')


def validate_temperature(temperature):
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError('aggregation_temperature must be a finite positive number')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('aggregation_temperature must be a finite positive number')


def aggregate_chunk_scores(chunk_scores, temperature=0.1):
    """Compute all three methods from exactly the same finite cosine scores.

    Input rows contain file_path, chunk_id and cosine_similarity. Scores are
    unshifted (ES's +1 has already been removed). Never truncate the inventory
    before grouping: LogSumExp's normalization uses every chunk in the file.
    """
    validate_temperature(temperature)
    grouped = defaultdict(list)
    seen = set()
    for row in chunk_scores:
        path, chunk_id, score = row['file_path'], row['chunk_id'], row['cosine_similarity']
        if not isinstance(path, str) or not path or not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError('Each chunk requires a file path and unique chunk ID')
        if chunk_id in seen:
            raise ValueError('Duplicate chunk ID in score inventory: ' + chunk_id)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('Chunk cosine similarities must be finite numbers')
        seen.add(chunk_id)
        grouped[path].append((float(score), chunk_id))
    if not grouped:
        raise ValueError('Cannot aggregate an empty chunk inventory')
    rankings = {method: [] for method in AGGREGATIONS}
    for path, values in grouped.items():
        values.sort(key=lambda value: (-value[0], value[1]))
        maximum = values[0][0]
        top = values[:3]
        # Subtract the maximum BEFORE dividing by temperature. No positive
        # exponent occurs, even for tiny temperatures or large input values.
        lse = maximum + temperature * (
            math.log(math.fsum(math.exp((score - maximum) / temperature) for score, _ in values))
            - math.log(len(values)))
        scores = {'max': maximum, 'top3_mean': math.fsum(score for score, _ in top) / len(top),
                  'normalized_logsumexp': lse}
        evidence = {'file_path': path, 'chunk_count': len(values), 'best_chunk_id': values[0][1],
                    'best_chunk_similarity': maximum, 'top_chunk_ids': [chunk_id for _, chunk_id in top]}
        for method, score in scores.items():
            rankings[method].append({**evidence, 'score': score})
    for method, ranking in rankings.items():
        ranking.sort(key=lambda item: (-item['score'], item['file_path']))
        rankings[method] = [{'rank': rank, **item} for rank, item in enumerate(ranking, 1)]
    return rankings
