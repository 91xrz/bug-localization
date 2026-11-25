import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
from run_direct8b500 import direct_inputs


class Direct8BTests(unittest.TestCase):
    def test_labels_and_small_model_scores_cannot_change_inputs(self):
        paths = ['a.py', 'b.py', 'c.py']
        chunks = [{'file_path': p, 'chunk_id': p, 'embedding_text': 'code ' + p} for p in paths]
        scores = [{**c, 'cosine_similarity': 0.8} for c in chunks]
        case = {'original': {'query': 'complete original report', 'ranking': [
            {'file_path': p, 'rank': i, 'score': 1 / i} for i, p in enumerate(paths, 1)], 'chunk_scores': scores},
            'hypothesis': {'ranking': [{'file_path': p, 'rank': i, 'score': 1 / i}
                                      for i, p in enumerate(reversed(paths), 1)], 'chunk_scores': scores},
            'chunks': chunks}
        clean = direct_inputs(case, 2)
        polluted = copy.deepcopy(case)
        polluted.update(gold=['secret.py'], ranking=['secret.py'], scored_chunks=[{'score': 1000000}])
        self.assertEqual(direct_inputs(polluted, 2), clean)
        self.assertNotIn('secret', json.dumps(clean))
        self.assertEqual(clean['query'], 'complete original report')
        self.assertEqual(clean['candidates'], ['a.py', 'c.py'])
        self.assertEqual(clean['tail'], ['b.py'])

    def test_full_chunk_inventory_can_exceed_fifty_file_pool(self):
        paths = [f'file_{i:03d}.py' for i in range(80)]
        ranking = [{'file_path': p, 'rank': i, 'score': 1 / i} for i, p in enumerate(paths, 1)]
        chunks = [{'file_path': p, 'chunk_id': p, 'embedding_text': 'code ' + p} for p in paths]
        scores = [{**c, 'cosine_similarity': 0.8} for c in chunks]
        result = direct_inputs({'original': {'query': 'report', 'ranking': ranking, 'chunk_scores': scores},
                                'hypothesis': {'ranking': ranking, 'chunk_scores': scores}, 'chunks': chunks})
        self.assertEqual(result['candidates'], paths[:10])
        self.assertEqual(result['tail'], paths[10:50])
        self.assertEqual(set(result['evidence']), set(paths[:10]))


if __name__ == '__main__':
    unittest.main()
