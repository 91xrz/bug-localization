import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.embeddings import digest
from bug_localization.hybrid import full_lexical_ranking, fuse_files, verify_case


def lexical():
    return {'repository': 'owner/repo', 'issue_id': 1, 'before_fix_sha': 'a' * 40,
            'indexed_files': 2, 'index': 'owned-index', 'index_fingerprint': 'fingerprint',
            'query_text': 'broken response', 'query': {'term': {'content': 'response'}},
            'candidate_files': [{'file_path': 'a.py', 'sha256': 'a'}, {'file_path': 'b.py', 'sha256': 'b'}],
            'ground_truth': ['a.py', 'missing.py'], 'ground_truth_missing_from_candidates': ['missing.py'],
            'total_matching_files': 2, 'results': [{'rank': 1, 'file_path': 'b.py', 'score': 9.0}]}


def semantic(bm):
    return {**{k: bm[k] for k in ('repository', 'issue_id', 'before_fix_sha', 'indexed_files', 'ground_truth')},
            'missing_gold': bm['ground_truth_missing_from_candidates'],
            'candidate_fingerprint': digest(bm['candidate_files']),
            'ranking': [{'rank': 1, 'file_path': 'a.py', 'aggregated_similarity': 0.8},
                        {'rank': 2, 'file_path': 'b.py', 'aggregated_similarity': 0.7}]}


class HybridTests(unittest.TestCase):
    def test_missing_lexical_match_has_zero_contribution_and_scores_do_not_set_rrf_ranks(self):
        bm = [{'rank': 1, 'file_path': 'b.py', 'score': 1e6}]
        sem = semantic(lexical())['ranking']
        fused = fuse_files(bm, sem, 60)
        self.assertEqual([r['file_path'] for r in fused], ['b.py', 'a.py'])
        self.assertAlmostEqual(fused[0]['score'], 1 / 61 + 1 / 62)
        self.assertIsNone(fused[1]['bm25_rank'])
        self.assertEqual(fused[1]['bm25_contribution'], 0)
        self.assertAlmostEqual(fused[1]['score'], 1 / 61)
        bm[0]['score'] = 0.0001
        self.assertEqual([r['file_path'] for r in fuse_files(bm, sem, 60)], ['b.py', 'a.py'])

    def test_duplicate_or_nonconsecutive_ranks_and_invalid_constants_are_rejected(self):
        sem = semantic(lexical())['ranking']
        for bad in [True, 0, float('nan'), 1.5]:
            with self.subTest(constant=bad), self.assertRaises(ValueError):
                fuse_files([], sem, bad)
        for bad in [[{'rank': 2, 'file_path': 'a.py'}],
                    [{'rank': 1, 'file_path': 'a.py'}, {'rank': 2, 'file_path': 'a.py'}]]:
            with self.assertRaises(ValueError):
                fuse_files(bad, sem, 60)

    def test_inputs_require_same_query_snapshot_labels_and_complete_semantic_ranking(self):
        bm = lexical()
        sem = semantic(bm)
        verify_case(bm, sem, bm['candidate_files'], {'body': bm['query_text']})
        changes = [{'before_fix_sha': 'b' * 40}, {'missing_gold': []}, {'ranking': sem['ranking'][:1]},
                   {'candidate_fingerprint': 'different'}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_case(bm, {**sem, **change}, bm['candidate_files'], {'body': bm['query_text']})
        with self.assertRaises(ValueError):
            verify_case(bm, sem, bm['candidate_files'], {'body': 'different report'})

    def test_truncated_bm25_is_expanded_read_only_using_original_query(self):
        bm = lexical()
        ownership = {'project': 'bug-localization', 'repository': bm['repository'],
                     'commit': bm['before_fix_sha'], 'fingerprint': bm['index_fingerprint']}
        client = SimpleNamespace(indices=SimpleNamespace(get_mapping=lambda **_: {
            bm['index']: {'mappings': {'_meta': ownership}}}), count=lambda **_: {'count': 2})
        inventory = [{'_source': v} for v in bm['candidate_files']]
        hits = [{'_source': {'file_path': 'b.py'}, '_score': 9.0},
                {'_source': {'file_path': 'a.py'}, '_score': 2.0}]
        with patch('elasticsearch.helpers.scan', side_effect=[inventory, hits]) as scan:
            ranking, audit = full_lexical_ranking(client, bm)
            self.assertEqual([r['file_path'] for r in ranking], ['b.py', 'a.py'])
            self.assertEqual(scan.call_args_list[1].kwargs['query']['query'], bm['query'])
            self.assertTrue(scan.call_args_list[1].kwargs['preserve_order'])
            self.assertTrue(audit['ownership_inventory_and_leading_ranks_verified'])
        with self.assertRaises(ValueError):
            full_lexical_ranking(None, bm)
        with patch('elasticsearch.helpers.scan', side_effect=[inventory, hits[::-1]]), self.assertRaises(ValueError):
            full_lexical_ranking(client, bm)
        changed = copy.deepcopy(inventory)
        changed[0]['_source']['sha256'] = 'tampered'
        with patch('elasticsearch.helpers.scan', return_value=changed), self.assertRaises(ValueError):
            full_lexical_ranking(client, bm)

    def test_complete_saved_bm25_and_zero_matches_need_no_index_access(self):
        bm = lexical()
        bm['total_matching_files'] = 1
        ranking, audit = full_lexical_ranking(None, bm)
        self.assertEqual(ranking, bm['results'])
        self.assertEqual(audit['source'], 'complete_saved_ranking')
        bm.update({'results': [], 'total_matching_files': 0})
        ranking, _ = full_lexical_ranking(None, bm)
        fused = fuse_files(ranking, semantic(bm)['ranking'], 60)
        self.assertEqual([r['file_path'] for r in fused], ['a.py', 'b.py'])
        self.assertTrue(all(r['bm25_contribution'] == 0 for r in fused))


if __name__ == '__main__':
    unittest.main()
