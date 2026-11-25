"""Full-corpus denominator and resumable-result integrity checks."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
spec = importlib.util.spec_from_file_location('recall_full', ROOT / 'scripts/experiments/run_recall_full.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FullRecallTest(unittest.TestCase):
    def test_pending_not_final_failure_keeps_denominator_and_macro_is_per_project(self):
        metas = [{'dataset_row': i, 'repository': 'large' if i < 3 else 'small', 'issue_id': i}
                 for i in range(4)]
        ok = [{**m, 'status': 'ok', 'missing_gold': [],
               'metrics': module.ranking_metrics(['a.py'], ['a.py'], 10)} for m in metas]
        failed = {**metas[-1], 'status': 'failed', 'error_type': 'GenerationError', 'error': 'overflow'}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            module.save(folder / 'selection.json', {'samples': metas, 'development_rows': [0],
                                                   'project_counts': {'large': 3, 'small': 1}})
            module.save(folder / 'L0/samples.json', ok)
            module.save(folder / 'S0/samples.json', ok[:3])
            module.save(folder / 'H0/samples.json', ok[:3] + [failed])
            value = module.summarize(folder)
            self.assertEqual(value['status'], 'running')
            self.assertIsNone(value['cohorts']['all2294']['S0']['metrics'])
            self.assertEqual(value['cohorts']['all2294']['H0']['metrics']['MAP@10'], .75)
            self.assertAlmostEqual(value['cohorts']['remaining1794']['H0']['metrics']['MAP@10'], 2/3)
            module.save(folder / 'S0/samples.json', ok)
            value = module.summarize(folder)
            self.assertEqual(value['status'], 'completed_with_failures')
            self.assertEqual(value['project_macro']['H0']['MAP@10'], .5)

    def test_resume_rejects_changed_result_and_metrics(self):
        meta = {'position': 1, 'dataset_row': 0, 'repository': 'x/y', 'issue_id': 1}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            row = module.record(folder, 'S0', meta, {'ranking': [{'file_path': 'a.py'}]},
                                [{'file_path': 'a.py', 'sha256': 'abc'}], ['a.py'], [])
            checkpoint = folder / 'S0/cases/0001/record.json'
            module.save(checkpoint, row)
            self.assertEqual(module.resumed(folder, 'S0', meta), row)
            row['metrics']['average_precision_at_k'] = 0
            module.save(checkpoint, row)
            with self.assertRaisesRegex(ValueError, 'metrics differ'):
                module.resumed(folder, 'S0', meta)
            result_path = folder / row['result_file']
            result_path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum changed'):
                module.resumed(folder, 'S0', meta)

    def test_duplicate_report_bodies_always_share_generation_worker(self):
        bodies = ['report ' + str(i) for i in range(100)]
        assignments = {body: module.shard(body) for body in bodies}
        self.assertEqual(set(assignments.values()), {0, 1})
        self.assertTrue(all(module.shard(body) == assignments[body] for body in reversed(bodies)))


if __name__ == '__main__':
    unittest.main()
