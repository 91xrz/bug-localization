import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bug_localization.aggregation import AGGREGATIONS, aggregate_chunk_scores
from bug_localization.bm25 import ranking_metrics
from bug_localization.vector import checkpoint, read_chunk_scores, validate_vector_config


def rows(scores, path='a.py'):
    return [{'file_path': path, 'chunk_id': path + '/' + str(i), 'cosine_similarity': value}
            for i, value in enumerate(scores)]


class FileAggregationTests(unittest.TestCase):
    def test_three_methods_use_the_complete_same_inventory(self):
        values = [0.9, 0.8, 0.7, -0.8, -0.9]
        result = aggregate_chunk_scores(rows(values))
        self.assertEqual(result['max'][0]['score'], 0.9)
        self.assertAlmostEqual(result['top3_mean'][0]['score'], 0.8)
        expected = 0.1 * math.log(sum(math.exp(x / 0.1) for x in values) / len(values))
        self.assertAlmostEqual(result['normalized_logsumexp'][0]['score'], expected)
        for ranking in result.values():
            self.assertEqual(ranking[0]['chunk_count'], 5)
        # Low-scoring chunks must affect normalized LSE even when outside top 3.
        shorter = aggregate_chunk_scores(rows(values[:3]))
        self.assertLess(result['normalized_logsumexp'][0]['score'], shorter['normalized_logsumexp'][0]['score'])

    def test_top_three_uses_actual_count_for_short_files(self):
        result = aggregate_chunk_scores(rows([0.8, 0.2]) + rows([-0.4], 'b.py'))
        by_path = {item['file_path']: item for item in result['top3_mean']}
        self.assertAlmostEqual(by_path['a.py']['score'], 0.5)
        self.assertEqual(by_path['b.py']['score'], -0.4)
        for method in AGGREGATIONS:
            self.assertEqual(next(r for r in result[method] if r['file_path'] == 'b.py')['score'], -0.4)

    def test_normalization_is_invariant_to_repeating_the_whole_distribution(self):
        first = aggregate_chunk_scores(rows([0.7, 0.2]))
        repeated = aggregate_chunk_scores(rows([0.7, 0.2] * 20))
        self.assertAlmostEqual(first['normalized_logsumexp'][0]['score'], repeated['normalized_logsumexp'][0]['score'])
        equal = aggregate_chunk_scores(rows([-0.3] * 100))
        self.assertAlmostEqual(equal['normalized_logsumexp'][0]['score'], -0.3)

    def test_logsumexp_is_stable_and_bounded_even_at_extreme_temperatures(self):
        for tau in (0.05, 0.1, 0.2, 1e-300):
            result = aggregate_chunk_scores(rows([1000.0, 999.0, -1000.0]), tau)
            score = result['normalized_logsumexp'][0]['score']
            self.assertTrue(math.isfinite(score))
            self.assertGreaterEqual(score, sum([1000.0, 999.0, -1000.0]) / 3)
            self.assertLessEqual(score, 1000.0)

    def test_file_ties_use_path_and_chunk_ties_use_id(self):
        result = aggregate_chunk_scores(list(reversed(rows([0.5, 0.5], 'b.py'))) + rows([0.5], 'a.py'))
        for ranking in result.values():
            self.assertEqual([r['file_path'] for r in ranking], ['a.py', 'b.py'])
            self.assertEqual(ranking[1]['best_chunk_id'], 'b.py/0')

    def test_rejects_invalid_temperatures_scores_and_duplicate_chunks(self):
        for tau in (0, -0.1, float('nan'), float('inf'), True, '0.1'):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                aggregate_chunk_scores(rows([0.5]), tau)
        for value in (float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                aggregate_chunk_scores(rows([value]))
        for values in ([], rows([0.5]) * 2):
            with self.assertRaises(ValueError):
                aggregate_chunk_scores(values)

    def test_configuration_rejects_invalid_aggregation_before_running(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'configs/retrieval/vector.json').read_text())
        base = json.loads((root / 'configs/retrieval/bm25.json').read_text())
        for changes in ({'file_aggregation': 'mean'}, {'aggregation_temperature': 0}):
            with self.assertRaises(ValueError):
                validate_vector_config({**config, **changes}, base)


class ScoredInventoryTests(unittest.TestCase):
    def query_vector(self):
        return SimpleNamespace(tolist=lambda: [1.0, 0.0])

    def test_scroll_keeps_every_chunk_and_restores_canonical_input_order(self):
        chunks = rows([0.8, 0.5, 0.3, -0.1])
        hits = [{'_source': {'file_path': r['file_path'], 'chunk_id': r['chunk_id']},
                 '_score': r['cosine_similarity'] + 1} for r in reversed(chunks)]
        with patch('elasticsearch.helpers.scan', return_value=iter(hits)) as scan:
            actual = read_chunk_scores(None, 'index', chunks, self.query_vector())
        self.assertEqual([r['chunk_id'] for r in actual], [r['chunk_id'] for r in chunks])
        for row, expected in zip(actual, chunks):
            self.assertAlmostEqual(row['cosine_similarity'], expected['cosine_similarity'])
        query = scan.call_args.kwargs['query']
        self.assertTrue(query['track_scores'])
        self.assertNotIn('collapse', query)

    def test_missing_duplicate_or_wrong_file_scores_are_rejected(self):
        chunks = rows([0.5, 0.2])
        hit = {'_source': {'file_path': 'a.py', 'chunk_id': 'a.py/0'}, '_score': 1.5}
        wrong = {'_source': {'file_path': 'other.py', 'chunk_id': 'a.py/0'}, '_score': 1.5}
        for hits in ([hit], [hit, hit], [wrong]):
            with patch('elasticsearch.helpers.scan', return_value=iter(hits)), self.assertRaises(ValueError):
                read_chunk_scores(None, 'index', chunks, self.query_vector())

    def test_nine_way_summary_preserves_failure_and_pending_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            context = {'file_aggregation': 'max', 'aggregation_temperature': 0.1, 'selection_sha256': 'selection'}
            metrics = ranking_metrics(['a.py'], ['a.py'], 10)
            good = {'position': 1, 'repository': 'owner/repo', 'issue_id': 1, 'status': 'ok',
                    'metrics': metrics, 'missing_gold': [], 'aggregation_metrics': {m: metrics for m in AGGREGATIONS}}
            bad = {'position': 2, 'repository': 'owner/repo', 'issue_id': 2, 'status': 'failed',
                   'error_type': 'TestError', 'error': 'failed'}
            records = {route: [good] for route in ('dynamic', 'method', 'fixed')}
            summary = checkpoint(Path(directory), context, records, 10, 2, SimpleNamespace(hits=0, misses=0))
            self.assertEqual(len(summary['aggregation_routes']), 9)
            self.assertTrue(all(v['metrics'] is None for v in summary['aggregation_routes'].values()))
            for route in records:
                records[route].append(bad)
            summary = checkpoint(Path(directory), context, records, 10, 2, SimpleNamespace(hits=0, misses=0))
            self.assertTrue(all(v['metrics']['MAP@10'] == 0.5 for v in summary['aggregation_routes'].values()))
            self.assertTrue(all(v['failed_samples'] == 1 for v in summary['aggregation_routes'].values()))


if __name__ == '__main__':
    unittest.main()
