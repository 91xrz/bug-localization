from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments'))
import evaluate_c_lora as original
import evaluate_c_lora_round2 as ev


class RoundTwoTests(unittest.TestCase):
    def test_disjoint_and_outcome_independent(self):
        rows = [dict(dataset_row=i,repository='r',issue_id=i,before_fix_sha='a') for i in range(40)]
        old = original.select(rows, set(), 10)
        blocked = set(map(original.identity, old))
        new = original.select(rows, blocked, 10, ev.SEED)
        reorder = [{**x, 'status': 'failed', 'metrics': 0} for x in reversed(rows)]
        self.assertEqual(set(map(original.identity,new)) & blocked, set())
        self.assertEqual(list(map(original.identity, new)),
                         list(map(original.identity, original.select(reorder, blocked, 10, ev.SEED))))

    def test_within_project_paired_differences(self):
        def m(ap, top):
            return dict(average_precision_at_k=ap,top_1=top)
        rows = [dict(repository='r1',base=m(.25,0),hard=m(1,1)),
                dict(repository='r2',base=m(1,1),hard=m(.5,0)),
                dict(repository='r3',base=m(1,1),hard=m(1,1))]
        result = ev.contrast(rows,'hard','base',repetitions=100)
        self.assertAlmostEqual(result['delta_MAP'],.25/3)
        self.assertEqual(result['project_stratified_bootstrap_CI95'], [.25/3,.25/3])
        self.assertEqual([result[k] for k in ('wins','ties','losses','top1_gained','top1_lost')], [1]*5)

    def test_refuse_adapter_bound_to_different_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for part in ('train','validation'):
                original.save(run/'data'/(part+'.jsonl'),dict(id=part))
            prot = dict(selection='fixed final epoch',epochs=2,**{
                k+'_sha256':original.sha256_file(run/'data'/(k+'.jsonl')) for k in ('train','validation')})
            original.save(run/'training/protocol.json', prot)
            original.save(run/'training/adapter/adapter_config.json',{
                'protocol_sha256':original.sha256_file(run/'training/protocol.json')})
            self.assertEqual(ev.verify_training(run), (run/'data',run/'training/adapter'))
            original.save(run/'data/train.jsonl',dict(id='wrong'))
            with self.assertRaises(ValueError):
                ev.verify_training(run)


if __name__ == '__main__':
    unittest.main()
