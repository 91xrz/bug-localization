import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('merge_blaze_repair', PROJECT / 'scripts/experiments/merge_blaze_repair.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BlazeRepairMergeTests(unittest.TestCase):
    def test_reuses_success_and_replaces_only_failed_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original, retry, out = root / 'original', root / 'retry', root / 'merged'
            for folder in (original, retry):
                folder.mkdir()
                for route in module.ROUTES:
                    (folder / route).mkdir()
                config = {'base': {'endpoint': str(folder), 'cluster_name': str(folder), 'top_k': 10},
                          'blaze': {'source_max_tokens': 512, 'report_max_tokens': 1024,
                                    'dense_depth': 1000, 'lexical_depth': 1000, 'rrf_constant': 60}}
                module.write_json(folder / 'effective_config.json', config)
                module.write_json(folder / 'provenance.json', {'model': {'id': 'same'}, 'server_version': '9.5.5',
                                                               'implementation_sha256': {'blaze.py': 'same'}})
            def sample(position, issue):
                return {'position': position, 'dataset_row': issue, 'repository': 'o/r', 'issue_id': issue,
                        'before_fix_sha': 'a' * 40, 'query_sha256': str(issue), 'labels_sha256': str(issue)}
            samples = [sample(1, 1), sample(2, 2)]
            module.write_json(original / 'selection.json', {'dataset_metadata_sha256': 'data', 'selection_sha256': 'full', 'samples': samples})
            module.write_json(retry / 'selection.json', {'dataset_metadata_sha256': 'data', 'selection_sha256': 'retry', 'samples': [sample(1, 2)]})
            metrics = module.ranking_metrics(['a.py'], ['a.py'], 10)
            for folder, identity in [(original, samples[0]), (retry, sample(1, 2))]:
                case = folder / 'cases/0001'
                case.mkdir(parents=True)
                module.write_json(case / 'result.json', {**identity, 'gold_files': ['a.py'], 'missing_gold': [],
                                  'routes': {r: {'ranking': [{'file_path': 'a.py'}]} for r in module.ROUTES}})
            for route in module.ROUTES:
                first = {**samples[0], 'status': 'ok', 'metrics': metrics, 'missing_gold': [], 'result_file': '../cases/0001/result.json'}
                second = {**samples[1], 'status': 'failed', 'error_type': 'Error', 'error': 'failed'}
                recovered = {**sample(1, 2), 'status': 'ok', 'metrics': metrics, 'missing_gold': [], 'result_file': '../cases/0001/result.json'}
                module.write_json(original / route / 'samples.json', [first, second])
                module.write_json(retry / route / 'samples.json', [recovered])
            result = module.merge(original, retry, out)
            for route in module.ROUTES:
                self.assertEqual(result['routes'][route]['successful_samples'], 2)
                self.assertEqual(result['routes'][route]['metrics']['MAP@10'], 1)
                self.assertEqual(result['route_source_counts'][route], {'reused_success': 1, 'retry': 1})
                records = module.read(out / route / 'samples.json')
                self.assertEqual([r['position'] for r in records], [1, 2])
                self.assertEqual([r['source_position'] for r in records], [1, 1])
            tampered = module.read(retry / 'selection.json')
            tampered['samples'][0]['query_sha256'] = 'changed'
            module.write_json(retry / 'selection.json', tampered)
            with self.assertRaisesRegex(ValueError, 'query/label'):
                module.merge(original, retry, root / 'tampered')


if __name__ == '__main__':
    unittest.main()
