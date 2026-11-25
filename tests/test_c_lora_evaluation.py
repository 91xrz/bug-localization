import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments'))
import evaluate_c_lora as ev


def case():
    return dict(query='unexpected return', candidates=[{'file_path': p} for p in ('use.py','impl.py')],
                ranking=[dict(file_path=p, semantic_rank=i, semantic_score=1/i)
                         for i,p in enumerate(('use.py','impl.py'), 1)],
                scored_chunks=[dict(file_path=p, chunk_id=str(i), embedding_text=t)
                               for i,(p,t) in enumerate([('use.py','symptom'),('impl.py','other'),('impl.py','repair')])],
                ground_truth=['impl.py'], irrelevant='do not pass')


class FixedEvaluationTests(unittest.TestCase):
    def test_selection_invariant_to_order_and_outcomes(self):
        rows = [dict(repository=repo, dataset_row=i, issue_id=i, before_fix_sha='v', status='ok')
                for repo in ('a/one', 'b/two') for i in range(5)]
        excluded = {ev.identity(rows[0])}
        expected = list(map(ev.identity, ev.select(rows, excluded, 3)))
        self.assertEqual(len(expected), 6)
        self.assertNotIn(ev.identity(rows[0]), expected)
        altered = [{**r,'status':'failed','metrics':999} for r in reversed(rows)]
        self.assertEqual(expected, list(map(ev.identity, ev.select(altered, excluded, 3))))

    def test_inference_visibility_file_max_and_hard_pairs(self):
        item = dict(id='r#1', source_status='ok', **ev.frozen_input(case()))
        self.assertNotIn('ground_truth',item)
        seen = []

        class Model:
            def score(self, query, docs):
                seen.append((query,docs))
                return [{'score':s} for s in (2.5,-10,3)]

        r = ev.infer(item, Model())
        self.assertEqual(seen, [('unexpected return',['symptom','other','repair'])])
        self.assertEqual(r['ranking'][0]['file_path'],'impl.py')
        self.assertEqual(r['ranking'][0]['best_chunk_id'],'2')
        self.assertEqual(ev.hard_pairs(item,r['ranking'],['impl.py']),1)
        self.assertIsNone(ev.hard_pairs(item,r['ranking'],['not-in-pool.py']))
        self.assertEqual(ev.hard_pairs(item,[dict(file_path=p,score=0) for p in ('impl.py','use.py')],['impl.py']),0.5)

    def test_source_failure_does_not_call_model(self):
        failed = ev.infer({'id':'r#2','source_status':'failed'}, None)
        self.assertEqual(failed['ranking'],[])
        self.assertEqual(ev.ranking_metrics([],['new.py'],10)['average_precision_at_k'],0)

    def test_excess_duplicate_or_unknown_chunks_rejected(self):
        c = case()
        c['scored_chunks'].append(c['scored_chunks'][-1])
        with self.assertRaises(ValueError):
            ev.frozen_input(c)
        c = case()
        c['candidates'].clear()
        with self.assertRaises(ValueError):
            ev.frozen_input(c)

    def test_hash_and_cross_split_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root/'labels.json'
            bodyhash = hashlib.sha256(b'exact repeat').hexdigest()
            ev.save(labels,[dict(repository='a/r',body_sha256=bodyhash)])
            manifest = dict(labels_sha256=ev.sha256_file(labels))
            for part in ('train','validation'):
                (root/(part+'.jsonl')).write_text(json.dumps(dict(repo='other/r',query='different')))
            self.assertEqual(ev.check_disjoint(root,manifest,root)['train']['n'],1)
            for bad in (dict(repo='A/R',query='different'),dict(repo='other/r',query='exact repeat')):
                (root/'train.jsonl').write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    ev.check_disjoint(root,manifest,root)
            ev.save(labels,[])
            with self.assertRaises(ValueError):
                ev.read_checked(labels,manifest['labels_sha256'])


if __name__ == '__main__':
    unittest.main()
