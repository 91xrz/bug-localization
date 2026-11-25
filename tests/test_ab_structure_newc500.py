import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
spec=importlib.util.spec_from_file_location('new_pool_new_c',ROOT/'scripts/experiments/run_ab_structure_newc500.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class NewPoolNewCChecks(unittest.TestCase):
    def test_cache_reuses_only_identical_inputs(self):
        old={'query':'report','packets':[{'file_path':'old.py','chunk_ids':['c'],'document':'old code'}]}
        new={'file_path':'new.py','chunk_ids':['n'],'document':'new code'}
        cached,missing=runner.reuse_packets('report',old['packets']+[new],old,{'old.py':{'score':2.}})
        self.assertEqual(set(cached),{'old.py'});self.assertEqual(missing,[new])
        with self.assertRaisesRegex(ValueError,'query differs'):
            runner.reuse_packets('changed report',old['packets'],old,{'old.py':{'score':2.}})
        changed=[{**old['packets'][0],'document':'changed code'}]
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            runner.reuse_packets('report',changed,old,{'old.py':{'score':2.}})

    def test_refinement_preserves_C_top10_and_tail(self):
        base=[f'f{i}' for i in range(12)]
        chunks={p:[1.] for p in base};joint={p:{'score':1.} for p in base};prior={p:0. for p in base}
        prior['f1']=1.;prior['f10']=100.
        result=runner.refine(base,chunks,prior,joint)
        self.assertEqual(result[0],'f1')
        self.assertEqual(set(result[:10]),set(base[:10]));self.assertEqual(result[10:],base[10:])

    def test_joint_component_has_frozen_quarter_weight(self):
        result=runner.refine(['a','b'],{'a':[2.],'b':[1.]},{'a':0.,'b':0.},
                             {'a':{'score':0.},'b':{'score':4.}})
        self.assertEqual(result,['b','a'])


if __name__=='__main__':unittest.main()
