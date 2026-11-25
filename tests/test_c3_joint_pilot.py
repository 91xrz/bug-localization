import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('c3joint',ROOT/'scripts/experiments/pilot_c3_joint.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class JointTests(unittest.TestCase):
    def test_inference_boundary_keeps_original_report_and_excludes_labels(self):
        saved={'ranking':[{'file_path':p} for p in ['a.py','b.py','c.py']],
               'scored_chunks':[{'file_path':p,'chunk_id':p,'embedding_text':'code '+p} for p in ['a.py','b.py','c.py']],
               'ground_truth':['secret.py'],'metrics':{'secret':True}}
        query,paths,evidence,tail=m.inference_input(saved,{'query':'full original report','gold':'secret'},2)
        self.assertEqual(query,'full original report');self.assertEqual(paths,['a.py','b.py']);self.assertEqual(tail,['c.py'])
        self.assertNotIn('secret',json.dumps(evidence))

    def test_duplicate_output_requires_actual_model_correction(self):
        class Generator:
            def __init__(self):self.calls=0
            def count_tokens(self,messages):return 12
            def complete(self,messages,**kwargs):
                self.calls+=1
                return json.dumps({'reasoning':'why','ranked_files':['a.py','a.py'] if self.calls==1 else ['b.py','a.py']})
        g=Generator();evidence={p:[{'text':'code','chunk_id':p}] for p in ['a.py','b.py']}
        with tempfile.TemporaryDirectory() as tmp:
            ranking,_=m.infer(g,'report',['a.py','b.py'],evidence,Path(tmp))
            self.assertEqual(ranking,['b.py','a.py']);self.assertEqual(g.calls,2)

if __name__=='__main__':unittest.main()
