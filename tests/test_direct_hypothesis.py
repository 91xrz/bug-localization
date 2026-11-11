import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('direct_hypothesis',ROOT/'scripts/experiments/run_direct_hypothesis.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class DirectHypothesisTests(unittest.TestCase):
    def test_prompt_changes_only_input_description(self):
        self.assertIn('ONLY an original bug report.',m.PROMPT_DIRECT)
        self.assertNotIn('previously extracted/enriched OB',m.PROMPT_DIRECT)
        self.assertEqual(m.PROMPT_DIRECT.split('Infer the software functionality',1)[1],m.PROMPT.split('Infer the software functionality',1)[1])

    def test_hypothesis_json_parser(self):
        value=m.parse_hypothesis('{"conflict":"c","context":"a","functionality":"b","logic":"d"}',True)
        self.assertEqual(value['functionality'],'b')
        with self.assertRaises(ValueError): m.parse_hypothesis('{"context":"a"}',True)

if __name__=='__main__': unittest.main()

class ShardingTests(unittest.TestCase):
    def test_same_report_always_goes_to_same_gpu(self):
        spec=importlib.util.spec_from_file_location('run_direct_hypothesis_multigpu',ROOT/'scripts/experiments/run_direct_hypothesis_multigpu.py')
        import sys
        sys.path.insert(0,str(ROOT/'scripts/experiments'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.shard('same report'),module.shard('same report'))
        self.assertEqual({module.shard(str(i)) for i in range(50)},{0,1})
