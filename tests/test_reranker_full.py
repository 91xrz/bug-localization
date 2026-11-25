"""Streaming dependency isolation, scientific audits and resumable outputs."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
import run_reranker_full as runner


class FullRerankerTest(unittest.TestCase):
    def test_query_rejects_hypothesis_or_wrong_instruction(self):
        result = {'query': 'original report extended',
                  'query_info': {'embedding_text': 'Instruct: test\nQuery: original report'}}
        self.assertEqual(runner.original_query(result, 'test'), 'original report')
        result['query_info']['embedding_text'] = 'Instruct: test\nQuery: hypothetical fix'
        with self.assertRaisesRegex(ValueError, 'original report'):
            runner.original_query(result, 'test')
        with self.assertRaisesRegex(ValueError, 'instruction'):
            runner.original_query(result, 'other')

    def test_overlapping_pool_keeps_h40_then_original_supplement(self):
        original = {'ranking': [{'file_path': str(i)} for i in range(80)]}
        hypothesis = {'ranking': [{'file_path': str(i)} for i in range(20, 80)] + original['ranking'][:20]}
        pool = runner.candidate_pool(original, hypothesis)
        self.assertEqual([r['file_path'] for r in pool], [str(i) for i in range(20, 60)] + [str(i) for i in range(10)])

    def test_dependency_failure_keeps_original_control_and_resume_checks_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); src = root / 'source'; out = root / 'out'
            save = runner.source.save
            metas = [{'position': i+1, 'dataset_row': i, 'repository': 'x/y', 'issue_id': i,
                      'before_fix_sha': 'commit'} for i in range(2)]
            save(src / 'protocol.json', {'semantic': {'query_instruction': 'test'}})
            save(out / 'protocol.json', {'files_sha256': {}, 'reranker': {}, 'reference_model': {},
                'config': {'source_run': str(src), 'physical_gpu': 0, 'minimum_free_mib': 1, 'poll_seconds': 0}})
            save(out / 'selection.json', {'samples': metas, 'development_rows': [0]})
            inventory = [{'file_path': p, 'sha256': p} for p in ('a', 'b')]
            chunks = [{'file_path': p, 'chunk_id': p, 'embedding_text': p} for p in ('a', 'b')]
            save(src / 'assets/shared/candidates.json', {'files': inventory})
            save(src / 'assets/shared/chunks.json', chunks)
            scores = [{'file_path': p, 'chunk_id': p, 'cosine_similarity': s} for p, s in [('a', .9), ('b', .8)]]
            for meta in metas:
                for method in ('S0', 'H0'):
                    if meta['position'] == 1 and method == 'H0':
                        record = {**meta, 'status': 'failed', 'error': 'input too long', 'error_type': 'InputTooLong'}
                    else:
                        ranking = [{'file_path': p, 'rank': i, 'score': 1/i} for i, p in enumerate(('a', 'b'), 1)]
                        value = {'ranking': ranking, 'chunk_scores': scores, 'chunk_scores_sha256': runner.digest(scores),
                                 'query': 'original report' if method == 'S0' else 'hypothesis',
                                 'query_info': {'embedding_text': 'Instruct: test\nQuery: original report'}}
                        record = runner.source.record(src, method, meta, value, inventory, ['b'], [], assets='assets/shared')
                    save(src / method / 'cases' / f"{meta['position']:04d}" / 'record.json', record)
            queries = []
            class Model:
                identity = {}
                def __init__(self, *args): pass
                def score(self, query, documents):
                    queries.append(query)
                    return [{'score': 1.0} for _ in documents]
            with patch.object(runner.source, 'context'), patch.object(runner.source, 'gpu_memory', return_value={0: 100}), \
                 patch.object(runner, 'LocalReranker', Model), patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}):
                runner.run(out)
                self.assertEqual(queries, ['original report'] * 3)
                self.assertEqual(runner.source.read(out / 'R0/summary.json')['successful_samples'], 2)
                full = runner.source.read(out / 'FULL/summary.json')
                self.assertEqual(full['failed_samples'], 1)
                self.assertEqual(full['metrics']['MRR@10'], .25)
                runner.run(out)
                self.assertEqual(len(queries), 3)  # no re-scoring after resume
                saved = runner.source.read(out / 'R0/cases/0001/result.json')
                case = runner.load_case(src, runner.source.resumed(src, 'S0', metas[0]), None, 'test')
                bad = copy.deepcopy(saved); bad['scored_chunks'][0]['embedding_text'] = 'changed'
                with self.assertRaisesRegex(ValueError, 'source text'):
                    runner.audit_original(case, bad)
                save(src / 'assets/shared/chunks.json', [])
                with self.assertRaisesRegex(ValueError, 'Checkpoint source changed'):
                    runner.run(out)


if __name__ == '__main__':
    unittest.main()
