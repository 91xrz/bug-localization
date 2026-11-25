import importlib.util
from pathlib import Path
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mixed_runner', PROJECT / 'scripts/experiments/run_mixed_recall500.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MixedAuditTests(unittest.TestCase):
    def test_complete_audit_preserves_failure_and_rejects_hypothesis_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            target = folder / 'C'
            (target / 'cases').mkdir(parents=True)
            input_file = folder / 'source.json'
            runner.write_json(input_file, {'source': 'immutable fixture'})
            original = [{'rank': i + 1, 'file_path': str(i), 'score': 1 / (i + 1)} for i in range(70)]
            hypothesis = [{**r, 'rank': i + 1} for i, r in enumerate(reversed(original))]
            chunks = [{'chunk_id': str(i), 'file_path': str(i), 'embedding_text': f'code {i}'} for i in range(70)]
            chunk_scores = [{'chunk_id': str(i), 'file_path': str(i), 'cosine_similarity': 0.5} for i in range(70)]
            case = {'original': {'ranking': original, 'chunk_scores': chunk_scores},
                    'hypothesis': {'ranking': hypothesis}, 'chunks': chunks, 'gold': ['0'],
                    'query': 'original report', 'inventory': {str(i) for i in range(70)}, 'paths': [input_file]}
            candidates = runner.mixed_candidates(original, hypothesis, 'hypothesis')
            head, chosen = runner.original_chunks(candidates, original, chunks, chunk_scores)
            scored = [{'score': 0.0, 'probability': 0.5, 'input_tokens': 80} for _ in chosen]
            ranking = runner.rerank(head, chosen, scored)
            metrics = runner.ranking_metrics([r['file_path'] for r in ranking], ['0'], 10)
            pool = runner.pool_metrics(candidates, ['0'])
            records, samples = [], []
            for pos in range(1, 501):
                sample = {'position': pos, 'dataset_row': pos, 'repository': 'fixture/repo', 'issue_id': pos,
                          'before_fix_sha': 'fixture', 'query_sha256': 'query', 'labels_sha256': 'labels'}
                samples.append(sample)
                if pos == 308:
                    records.append({**sample, 'status': 'failed', 'error_type': 'SourceDependencyFailure', 'error': 'Input exceeds limit'})
                    continue
                path = target / 'cases' / f'{pos:03d}.json'
                runner.write_json(path, {**sample, 'query': case['query'], 'reranker_query_source': 'original_report',
                    'ground_truth': ['0'], 'candidates': candidates, 'pool_metrics': pool, 'ranking': ranking, 'metrics': metrics,
                    'scored_chunks': [{**c, **s} for c, s in zip(chosen, scored)],
                    'source_files_sha256': {str(input_file.resolve()): runner.sha256_file(input_file)}})
                records.append({**sample, 'status': 'ok', 'metrics': metrics, 'pool_metrics': pool,
                                'missing_gold': [], 'result_file': str(path.relative_to(target))})
            runner.write_json(target / 'samples.json', records)

            class Source:
                config = {'methods': {'C': 'hypothesis'}}

                def __init__(self):
                    self.samples = samples

                def failures(self, sample):
                    return [{'status': 'failed'}] if sample['position'] == 308 else []

                def load(self, sample):
                    return case

            result = runner.audit(folder, 'C', Source())
            self.assertEqual(result['verified_successes'], 499)
            self.assertEqual(result['expected_source_failures'], [308])
            self.assertEqual(result['unexpected_failures'], [])
            self.assertEqual(runner.aggregate_results(records, 10)['metrics']['MAP@10'], 499 / 500)
            path = target / records[0]['result_file']
            saved = runner.read_json(path)
            runner.write_json(path, {**saved, 'query': 'generated functional hypothesis'})
            with self.assertRaisesRegex(ValueError, 'original report'):
                runner.audit(folder, 'C', Source())
            runner.write_json(path, {**saved, 'candidates': saved['candidates'][:-1]})
            with self.assertRaisesRegex(ValueError, 'primary40'):
                runner.audit(folder, 'C', Source())


if __name__ == '__main__':
    unittest.main()
