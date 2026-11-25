import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/experiments')]
SPEC=importlib.util.spec_from_file_location('multidata_evaluator',ROOT/'scripts/analysis/evaluate_multidata_coefficients.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
from run_multidata_coefficients import checkpoint


class MultidataEvaluationTests(unittest.TestCase):
    def experiment(self,out,validation_gold):
        protocol={'ab_alphas':[0,.5,1],'anchor_weights':[0,.5],
            'packet_weights':[0,.25,1],'prior_weights':[0,4],
            'default_coefficients':[.5,.5,.25,4], 'historical_limits':'synthetic test'}
        plan={}
        ranking=[{'rank':1,'file_path':'a','score':2.},{'rank':2,'file_path':'b','score':1.}]
        for dataset in ('one','two'):
            plan[dataset]=[]
            for i in range(1,5):
                row={'position':i,'dataset_row':i,'repository':f'{dataset}/project{i}',
                     'issue_id':i,'before_fix_sha':'a'*40,'partition':'development' if i<=2 else 'validation',
                     'gold':['b'] if i<=2 else [validation_gold]}
                plan[dataset].append(row)
                inp={'status':'ok','a':ranking,'b':ranking,'anchor':{'a':0.,'b':1.}}
                score={'status':'ok','max':{'a':1.,'b':0.},'joint':{'a':0.,'b':2.}}
                if i==4:inp=score={'status':'failed'}
                checkpoint(out/dataset/'inputs'/f'{i:04d}.json',inp,'input')
                checkpoint(out/dataset/'scores'/f'{i:04d}.json',score,'score')
        (out/'protocol.json').write_text(json.dumps(protocol))
        with patch.object(M,'load',return_value=(protocol,plan)),contextlib.redirect_stdout(io.StringIO()):
            M.run(out)
        return json.loads((out/'analysis/frozen_choice.json').read_bytes()),json.loads((out/'analysis/summary.json').read_bytes())

    def test_validation_labels_cannot_change_frozen_coefficients_and_failures_stay(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            first,first_metrics=self.experiment(a,'a')
            second,second_metrics=self.experiment(b,'b')
            self.assertEqual(first['per_dataset'],second['per_dataset'])
            self.assertEqual(first['shared'],second['shared'])
            for dataset in first_metrics:
                self.assertEqual(first_metrics[dataset]['failures'],1)
                self.assertEqual(first_metrics[dataset]['default']['n'],2)
                self.assertLessEqual(first_metrics[dataset]['default']['Top1'],.5)
                self.assertEqual(second_metrics[dataset]['failures'],1)


if __name__=='__main__':unittest.main()
