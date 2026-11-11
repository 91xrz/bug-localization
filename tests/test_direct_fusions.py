import importlib.util
from pathlib import Path
import unittest
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fusion',root/'scripts/analysis/compare_direct_fusions.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def row(path,rank,score):return {'file_path':path,'rank':rank,'aggregated_similarity':score}
class FusionTests(unittest.TestCase):
 def test_dedup_and_rrf(self):
  a=[row('a',1,.9),row('b',2,.8),row('c',3,.7)];b=[row('b',1,.9),row('a',2,.8),row('c',3,.7)]
  x=m.fuse(a,b);self.assertEqual(x['interleave_original_first'],['a','b','c']);self.assertEqual(x['interleave_hypothesis_first'],['b','a','c']);self.assertEqual(x['rrf_k60_w1'][:2],['a','b'])
 def test_empty_channel_and_equal_scores(self):
  a=[row('a',1,.8),row('b',2,.8)];x=m.fuse(a,[])
  self.assertEqual(x['interleave_original_first'],['a','b']);self.assertEqual(x['rrf_k10_w1'],['a','b']);self.assertEqual(m.normalized(m.indexed(a)),{'a':0.0,'b':0.0})
 def test_duplicate_rejected(self):
  with self.assertRaises(ValueError):m.indexed([row('a',1,.5),row('a',2,.4)])
if __name__=='__main__':unittest.main()
