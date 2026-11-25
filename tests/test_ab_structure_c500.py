import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ab_c_runner',ROOT/'scripts/experiments/run_ab_structure_c500.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class PairedCChecks(unittest.TestCase):
    def test_file_score_uses_maximum_not_average(self):
        result=runner.aggregate(['a','b'],{'a':['a1','a2'],'b':['b1']},
                                {'a1':{'score':9.},'a2':{'score':-10.},'b1':{'score':5.}})
        self.assertEqual([r['file_path'] for r in result],['a','b'])
        self.assertEqual(result[0]['best_chunk_id'],'a1')

    def test_equal_c_scores_follow_each_arms_ab_order(self):
        ids={'a':['a1'],'b':['b1']};scores={'a1':{'score':2.},'b1':{'score':2.}}
        old=runner.aggregate(['a','b'],ids,scores);new=runner.aggregate(['b','a'],ids,scores)
        self.assertEqual(old[0]['file_path'],'a')
        self.assertEqual(new[0]['file_path'],'b')

    def test_evidence_preserves_A_best_and_unique_B_supplement(self):
        a=[{'file_path':'a.py','top_chunk_ids':['shared','a2','a3']}]
        b=[{'file_path':'a.py','top_chunk_ids':['b1','shared','b2']}]
        self.assertEqual(runner.mixed_ids(['a.py'],a,b),{'a.py':['shared','b1','b2']})

    def test_new_chunk_text_cannot_change_without_detection(self):
        chunk={'route':'blagent_ast','file_path':'a.py','parent_id':0,'start_byte':0,'end_byte':4,
               'embedding_text':'original'}
        chunk['chunk_id']=hashlib.sha256(json.dumps(['blagent_ast','a.py',0,0,4,'original'],ensure_ascii=False).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'chunks.json';path.write_text(json.dumps([chunk],indent=2))
            self.assertEqual(runner.extract_chunks(path,{chunk['chunk_id']})[chunk['chunk_id']]['embedding_text'],'original')
            chunk['embedding_text']='changed';path.write_text(json.dumps([chunk],indent=2))
            with self.assertRaisesRegex(ValueError,'content hash mismatch'):
                runner.extract_chunks(path,{chunk['chunk_id']})


if __name__=='__main__':unittest.main()
