import copy
from pathlib import Path
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'scripts/experiments'))
from bug_localization.chunk_ablation import select_evidence
from bug_localization.reranking import rerank
from bug_localization.bm25 import ranking_metrics
from run_chunk_ablation500 import audit_case, pool_metrics


class ChunkAblationTests(unittest.TestCase):
    def setUp(self):
        self.ranking = [{'file_path': p, 'rank': i, 'score': 1 / i} for i, p in enumerate(('a', 'b', 'c'), 1)]
        self.candidates = [{'file_path': p, 'is_primary': True} for p in ('c', 'b', 'a')]
        self.chunks = [{'file_path': p, 'chunk_id': p + str(i), 'embedding_text': p + str(i)}
                       for p, n in [('a', 4), ('b', 2), ('c', 1)] for i in range(n)]
        self.original = [{**c, 'cosine_similarity': s} for c, s in zip(self.chunks, [.9, .8, .7, .6, .9, .8, .9])]
        self.hypothesis = [{**c, 'cosine_similarity': s} for c, s in zip(self.chunks, [.8, .9, .6, .7, .8, .9, .9])]

    def select(self, policy):
        return select_evidence(self.candidates, self.ranking, self.chunks, self.original, self.hypothesis, policy)

    def test_hypothesis_top3_and_original_control(self):
        head, chosen = self.select('hypothesis')
        self.assertEqual([r['file_path'] for r in head], ['a', 'b', 'c'])
        self.assertEqual([r['chunk_id'] for r in chosen], ['a1', 'a0', 'a3', 'b1', 'b0', 'c0'])
        self.assertEqual([r['chunk_id'] for r in self.select('original')[1]], ['a0', 'a1', 'a2', 'b0', 'b1', 'c0'])

    def test_mixed_overlap_pads_from_next_hypothesis_chunk(self):
        _, chosen = self.select('mixed')
        self.assertEqual([r['chunk_id'] for r in chosen], ['a0', 'a1', 'a3', 'b0', 'b1', 'c0'])
        self.assertEqual([r['selected_by'] for r in chosen], ['original', 'hypothesis', 'hypothesis', 'original', 'hypothesis', 'original'])
        self.assertEqual(len(set(r['chunk_id'] for r in chosen)), len(chosen))
        self.assertEqual(chosen[0]['original_similarity'], .9)
        self.assertEqual(chosen[0]['hypothesis_similarity'], .8)

    def test_mixed_without_overlap_keeps_exact_original1_hypothesis2(self):
        self.hypothesis[0]['cosine_similarity'] = .1
        self.assertEqual([r['chunk_id'] for r in self.select('mixed')[1]][:3], ['a0', 'a1', 'a3'])

    def test_tied_semantic_scores_use_chunk_id(self):
        for r in self.hypothesis:
            r['cosine_similarity'] = .5
        self.hypothesis.reverse()
        self.assertEqual([r['chunk_id'] for r in self.select('hypothesis')[1]][:3], ['a0', 'a1', 'a2'])

    def test_corrupt_inventory_rejected(self):
        self.hypothesis[-1] = self.hypothesis[0]
        with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
            self.select('mixed')

    def test_independent_audit_checks_inputs_and_padding(self):
        for policy in ('hypothesis', 'mixed'):
            head, chosen = self.select(policy)
            ranking = rerank(head, chosen, [{'score': 1.0} for _ in chosen])
            gold = ['b', 'missing']
            metrics = ranking_metrics([r['file_path'] for r in ranking], gold, 10)
            pool = pool_metrics(self.candidates, gold)
            case = {'query': 'original report', 'candidates': self.candidates, 'gold': gold,
                    'source_hashes': {}, 'reference_sha256': 'refhash', 'reference_file': '/reference.json',
                    'original': {'ranking': self.ranking, 'chunk_scores': self.original},
                    'hypothesis': {'chunk_scores': self.hypothesis}, 'chunks': self.chunks, 'inventory': {'a', 'b', 'c'}}
            saved = {'query': case['query'], 'reranker_query_source': 'original_report', 'chunk_policy': policy,
                     'candidates': self.candidates, 'ground_truth': gold, 'source_files_sha256': {},
                     'reference_sha256': 'refhash', 'reference_file': '/reference.json',
                     'scored_chunks': [{**r, 'score': 1.0} for r in chosen], 'ranking': ranking,
                     'metrics': metrics, 'pool_metrics': pool}
            record = {'metrics': metrics, 'pool_metrics': pool, 'missing_gold': ['missing']}
            audit_case(case, saved, record, policy)
            bad = copy.deepcopy(saved)
            bad['query'] = 'functional hypothesis'
            with self.assertRaisesRegex(ValueError, 'original report'):
                audit_case(case, bad, record, policy)
            bad = copy.deepcopy(saved)
            bad['scored_chunks'][2] = bad['scored_chunks'][0]
            with self.assertRaisesRegex(ValueError, 'duplicate-padding'):
                audit_case(case, bad, record, policy)
            bad = copy.deepcopy(saved)
            bad['ranking'].reverse()
            with self.assertRaisesRegex(ValueError, 'Max aggregation'):
                audit_case(case, bad, record, policy)


if __name__ == '__main__':
    unittest.main()
