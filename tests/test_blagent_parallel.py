import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
PROJECT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('blagent_parallel',PROJECT/'scripts/experiments/run_blagent_parallel.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ParallelTests(unittest.TestCase):
    def selection(self):
        samples=[{'position':i+1,'dataset_row':i,'repository':'r','issue_id':i,'before_fix_sha':'a'*40} for i in range(5)]
        return {'samples':samples,'selection_sha256':m.digest(samples)}

    def test_strided_partitions_cover_once_and_merge_partial_positions(self):
        selection=self.selection();groups=m.partitions(selection)
        self.assertEqual(groups[0][0],[0,2,4]);self.assertEqual(groups[1][0],[1,3])
        self.assertEqual(sorted(groups[0][0]+groups[1][0]),list(range(5)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i,(_,part) in enumerate(groups):
                worker=root/'workers'/str(i);worker.mkdir(parents=True)
                row={**part['samples'][0],'status':'failed','error_type':'ValueError','error':'invalid','elapsed_seconds':1}
                (worker/'samples.json').write_text(json.dumps([row]))
                case=worker/'cases'/'0001';case.mkdir(parents=True);(case/'error.json').write_text('{}')
            summary=m.merge(root,selection,groups,{})
            self.assertEqual(summary['attempted_samples'],2);self.assertEqual(summary['pending_samples'],3)
            self.assertIsNone(summary['metrics'])
            rows=json.loads((root/'samples.json').read_text());self.assertEqual([r['position'] for r in rows],[1,2])

    def test_worker_different_sample_is_rejected(self):
        selection=self.selection();groups=m.partitions(selection)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);worker=root/'workers'/'0';worker.mkdir(parents=True)
            (worker/'samples.json').write_text(json.dumps([{**groups[0][1]['samples'][0],'issue_id':99}]))
            with self.assertRaisesRegex(ValueError,'identity mismatch'):m.merge(root,selection,groups,{})

    def test_settings_gate_excludes_only_selection(self):
        self.assertEqual(m.scientific({'selection_manifest':'a','max_calls':36}),m.scientific({'selection_manifest':'b','max_calls':36}))
        self.assertNotEqual(m.scientific({'max_calls':36}),m.scientific({'max_calls':20}))

if __name__=='__main__':unittest.main()
