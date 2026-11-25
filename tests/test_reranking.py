from pathlib import Path
import sys
import unittest
import importlib.util
import tempfile
import json
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.reranking import (rerank, select_chunks, validate_config, load_report_experiment_case,
                                       load_direct_hypothesis_case, validate_resume)
from bug_localization.bm25 import ranking_metrics
from bug_localization.batch import aggregate_results


class RerankingTests(unittest.TestCase):
    def setUp(self):
        self.ranking = [{'rank': i, 'file_path': path, 'score': 1 / i}
                        for i, path in enumerate(['a.py', 'b.py', 'c.py'], 1)]
        self.chunks = [{'chunk_id': key, 'file_path': path, 'embedding_text': key}
                       for key, path in [('a1', 'a.py'), ('a2', 'a.py'), ('b1', 'b.py'), ('c1', 'c.py')]]
        self.scores = [{**chunk, 'cosine_similarity': score}
                       for chunk, score in zip(self.chunks, [0.2, 0.9, 0.8, 0.7])]

    def test_candidates_use_semantic_top_files_and_best_chunks(self):
        chosen = select_chunks(self.ranking, self.chunks, self.scores, 2, 1)
        self.assertEqual([r['chunk_id'] for r in chosen], ['a2', 'b1'])
        ranking = rerank(self.ranking, chosen, [{'score': -4}, {'score': -1}])
        self.assertEqual([r['file_path'] for r in ranking], ['b.py', 'a.py', 'c.py'])
        self.assertEqual([r['rank'] for r in ranking], [1, 2, 3])
        self.assertIsNone(ranking[-1]['score'])
        self.assertEqual(self.ranking[0]['file_path'], 'a.py')

    def test_max_aggregation_and_ties_preserve_semantic_order(self):
        chosen = select_chunks(self.ranking, self.chunks, self.scores, 2, 3)
        ranking = rerank(self.ranking, chosen, [{'score': 1}, {'score': 5}, {'score': 5}])
        self.assertEqual([r['file_path'] for r in ranking], ['a.py', 'b.py', 'c.py'])
        self.assertEqual(ranking[0]['best_chunk_id'], 'a1')

    def test_score_inventory_corruption_is_rejected(self):
        for scores in [self.scores[:-1], self.scores[:-1] + [self.scores[0]],
                       [{**self.scores[0], 'file_path': 'wrong.py'}] + self.scores[1:]]:
            with self.assertRaises(ValueError):
                select_chunks(self.ranking, self.chunks, scores, 2, 1)
        with self.assertRaises(ValueError):
            rerank(self.ranking, self.chunks, [{'score': 1}])
        with self.assertRaises(ValueError):
            rerank(self.ranking, self.chunks[:1], [{'score': float('nan')}])

    def test_missing_labels_and_failures_remain_in_denominators(self):
        metrics = ranking_metrics(['a.py', 'b.py'], ['a.py', 'missing.py'], 10)
        self.assertEqual(metrics['average_precision_at_k'], 0.5)
        summary = aggregate_results([
            {'status': 'ok', 'metrics': metrics, 'missing_gold': ['missing.py']},
            {'status': 'failed', 'repository': 'repo', 'issue_id': 1, 'error_type': 'Error', 'error': 'failed'}], 10, 2)
        self.assertEqual(summary['metrics']['MAP@10'], 0.25)

    def test_invalid_settings(self):
        config = {'top_n_files': 50, 'chunks_per_file': 3, 'batch_size': 16,
                  'max_tokens': 1536, 'evaluation_k': 10, 'instruction': 'Retrieve buggy code'}
        for key, value in [('top_n_files', 9), ('chunks_per_file', 0), ('batch_size', True), ('max_tokens', -1)]:
            with self.assertRaises(ValueError):
                validate_config({**config, key: value})

    def test_standard_semantic_cli_can_chain_reranking(self):
        spec = importlib.util.spec_from_file_location('vector_cli', PROJECT / 'scripts/retrieval/run_vector_batch.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        source = PROJECT / 'results/vector/example'
        with patch.object(sys, 'argv', ['run_vector_batch.py', '--routes', 'blagent_ast', '--rerank', '--device', 'cpu']), \
                patch.object(cli, 'run_semantic_batch', return_value=(source, {'routes': {'blagent_ast': {'failed_samples': 0}}})), \
                patch('bug_localization.reranking.run_reranking', return_value=(source, {'status': 'completed'})) as run:
            cli.main()
        self.assertEqual(run.call_args.args[1], source)
        self.assertEqual(run.call_args.args[2]['device'], 'cpu')
        self.assertEqual(run.call_args.args[3], ['blagent_ast'])

    def test_report_experiment_adapter_uses_only_original_report(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            case = source / 'cases/001'
            case.mkdir(parents=True)
            payloads = {
                'q0_result.json': {'ranking': self.ranking, 'metrics': {'k': 10}, 'ground_truth': ['a.py']},
                'q0_retrieval.json': {'ranking': self.ranking, 'query': 'original report',
                    'query_info': {'embedding_text': 'original report', 'truncated': False},
                    'chunk_scores': self.scores, 'chunk_scores_sha256': 'checksum'},
                'chunks.json': [{**c, 'route': 'blagent_ast'} for c in self.chunks],
                'candidates.json': {'files': [{'file_path': r['file_path']} for r in self.ranking]},
                'hypothesis_all_retrieval.json': {'query': 'must not be used'},
            }
            for name, value in payloads.items():
                (case / name).write_text(json.dumps(value))
            record = {'position': 1, 'metrics': {'k': 10}, 'missing_gold': []}
            loaded = load_report_experiment_case(source, record)
            self.assertEqual(loaded[3]['body'], 'original report')
            self.assertEqual(loaded[5]['scores'], self.scores)
            payloads['q0_result.json']['ranking'] = self.ranking[::-1]
            (case / 'q0_result.json').write_text(json.dumps(payloads['q0_result.json']))
            with self.assertRaises(ValueError):
                load_report_experiment_case(source, record)

    def test_resume_preserves_settings_and_requires_paired_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            sample = {'position': 1, 'dataset_row': 5, 'repository': 'repo', 'issue_id': 7, 'before_fix_sha': 'abc'}
            config = {'top_n_files': 50}
            saved = {'reranker': config, 'source_run': str(folder), 'routes': ['blagent_ast'], 'source_metadata_sha256': {}}
            (folder / 'effective_config.json').write_text(json.dumps(saved))
            (folder / 'selection.json').write_text(json.dumps({'samples': [sample]}))
            records = {'blagent_ast': {'baseline': [sample], 'reranked': [sample]}}
            (folder / 'samples.json').write_text(json.dumps(records))
            self.assertEqual(validate_resume(folder, folder, config, ['blagent_ast'], [sample])[2], 1)
            with self.assertRaises(ValueError):
                validate_resume(folder, folder, {'top_n_files': 20}, ['blagent_ast'], [sample])
            records['blagent_ast']['reranked'] = []
            (folder / 'samples.json').write_text(json.dumps(records))
            with self.assertRaises(ValueError):
                validate_resume(folder, folder, config, ['blagent_ast'], [sample])

    def test_direct_adapter_uses_original_report_but_hypothesis_candidate_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            source, assets = Path(temp) / 'direct', Path(temp) / 'assets'
            for root in (source, assets):
                (root / 'cases/001').mkdir(parents=True)
            payloads = {
                source / '001_queries.json': {'query': 'frozen functional hypothesis', 'generation_error': None},
                source / 'cases/001/direct_result.json': {
                    'ranking': self.ranking, 'metrics': {'k': 10}, 'ground_truth': ['a.py']},
                source / 'cases/001/direct_retrieval.json': {'ranking': self.ranking,
                    'query_info': {'embedding_text': 'frozen functional hypothesis', 'truncated': False},
                    'chunk_scores': self.scores, 'chunk_scores_sha256': 'hypothesis-scores'},
                assets / 'cases/001/q0_result.json': {'ground_truth': ['a.py']},
                assets / 'cases/001/q0_retrieval.json': {'query': 'original bug report',
                    'query_info': {'embedding_text': 'original bug', 'truncated': True},
                    'chunk_scores': [{'score': 'must not replace hypothesis scores'}]},
                assets / 'cases/001/chunks.json': [{**c, 'route': 'blagent_ast'} for c in self.chunks],
                assets / 'cases/001/candidates.json': {'files': [{'file_path': r['file_path']} for r in self.ranking]},
            }
            for path, value in payloads.items():
                path.write_text(json.dumps(value))
            record = {'position': 1, 'metrics': {'k': 10}, 'missing_gold': []}
            loaded = load_direct_hypothesis_case(source, assets, record)
            self.assertEqual(loaded[3]['body'], 'original bug report')
            self.assertEqual(loaded[3]['embedding_text'], 'original bug')
            self.assertTrue(loaded[3]['truncated'])
            self.assertEqual(loaded[5]['sha256'], 'hypothesis-scores')
            self.assertEqual(loaded[5]['scores'], self.scores)
            self.assertEqual(loaded[1]['ranking'], self.ranking)
            self.assertEqual(loaded[1]['retrieval_query_info']['embedding_text'], 'frozen functional hypothesis')
            self.assertIn(assets / 'cases/001/chunks.json', loaded[-1])
            (assets / 'cases/001/q0_result.json').write_text(json.dumps({'ground_truth': ['wrong.py']}))
            with self.assertRaises(ValueError):
                load_direct_hypothesis_case(source, assets, record)


if __name__ == '__main__':
    unittest.main()
