import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bug_localization.mixed_recall import mixed_candidates, original_chunks


def ranking(paths):
    return [{'file_path': str(p), 'rank': i, 'score': 1 / i} for i, p in enumerate(paths, 1)]


class MixedRecallTests(unittest.TestCase):
    def test_overlap_requires_scanning_past_secondary_top10(self):
        a = ranking(range(100))
        b = ranking(list(range(30)) + list(range(50, 60)) + list(range(30, 50)) + list(range(60, 100)))
        c = mixed_candidates(a, b, 'hypothesis')
        d = mixed_candidates(a, b, 'original')
        self.assertEqual([r['file_path'] for r in c], [str(i) for i in list(range(30)) + list(range(50, 60)) + list(range(30, 40))])
        self.assertEqual([r['file_path'] for r in d], [str(i) for i in list(range(40)) + list(range(50, 60))])
        self.assertEqual(sum(r['is_primary'] for r in c), 40)
        self.assertTrue(all(r['introduced_by'] == 'original' for r in c[40:]))
        self.assertEqual(len(set(r['file_path'] for r in c)), 50)

    def test_small_universe_and_invalid_inventory(self):
        a, b = ranking(range(7)), ranking(reversed(range(7)))
        self.assertEqual(len(mixed_candidates(a, b, 'original')), 7)
        for invalid in [b[:-1], b + [b[0]], [{**b[0], 'rank': 2}] + b[1:]]:
            with self.assertRaises(ValueError):
                mixed_candidates(a, invalid, 'original')

    def test_direction_changes_membership(self):
        a, b = ranking(range(100)), ranking(reversed(range(100)))
        c = mixed_candidates(a, b, 'hypothesis')
        d = mixed_candidates(a, b, 'original')
        self.assertEqual({r['file_path'] for r in c}, {str(i) for i in list(range(60, 100)) + list(range(10))})
        self.assertNotEqual({r['file_path'] for r in c}, {r['file_path'] for r in d})

    def test_shared_files_use_identical_original_evidence_and_tie_order(self):
        a, b = ranking(range(60)), ranking(reversed(range(60)))
        chunks = [{'chunk_id': f'{p}-{j}', 'file_path': str(p), 'embedding_text': str(j)} for p in range(60) for j in range(4)]
        scores = [{**r, 'cosine_similarity': int(r['embedding_text']) / 4} for r in chunks]
        outputs = []
        for primary in ('original', 'hypothesis'):
            candidates = mixed_candidates(a, b, primary)
            head, chosen = original_chunks(candidates, a, chunks, scores)
            self.assertEqual([int(r['file_path']) for r in head], sorted(int(r['file_path']) for r in candidates))
            outputs.append({p: [r['chunk_id'] for r in chosen if r['file_path'] == p] for p in [r['file_path'] for r in head]})
        for path in outputs[0].keys() & outputs[1].keys():
            self.assertEqual(outputs[0][path], outputs[1][path])
            self.assertEqual(outputs[0][path], [f'{path}-{j}' for j in (3, 2, 1)])


if __name__ == '__main__':
    unittest.main()
