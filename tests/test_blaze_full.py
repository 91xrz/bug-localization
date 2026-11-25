import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
import run_blaze_full as runner


class BlazeFullTests(unittest.TestCase):
    def test_cleanup_rejects_unowned_or_wrong_index(self):
        client = Mock()
        fingerprint = 'a'*64
        index = 'bl-blaze-' + fingerprint[:32]
        owner = {'project': 'bug-localization', 'method': 'BLAZE-release', 'fingerprint': fingerprint}
        client.indices.get_mapping.return_value = {index: {'mappings': {'_meta': owner}}}
        runner.cleanup_index(client, index, fingerprint)
        client.indices.delete.assert_called_once_with(index=index)
        client.indices.delete.reset_mock()
        with self.assertRaisesRegex(ValueError, 'identity'):
            runner.cleanup_index(client, 'some-other-index', fingerprint)
        client.indices.get_mapping.return_value[index]['mappings']['_meta'] = {'method': 'other experiment'}
        with self.assertRaisesRegex(ValueError, 'unowned'):
            runner.cleanup_index(client, index, fingerprint)
        client.indices.delete.assert_not_called()

    def test_three_route_checkpoint_checks_metrics_identity_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);meta = {'position': 1, 'dataset_row': 0, 'repository': 'o/r', 'issue_id': 9}
            metrics = runner.ranking_metrics(['b','a'], ['a'], 10)
            result = {'gold_files': ['a'], 'missing_gold': [], 'routes': {
                route: {'ranking': [{'file_path': 'b'}, {'file_path': 'a'}], 'metrics': metrics}
                for route in runner.ROUTES}}
            path = root/'cases/0001/result.json';runner.save(path,result)
            bundle = {'sample': meta, 'result_file': 'cases/0001/result.json', 'result_sha256': runner.sha256_file(path),
                      'records': {route: {**meta, 'status': 'ok', 'metrics': metrics, 'missing_gold': []} for route in runner.ROUTES}}
            runner.validate_bundle(root,meta,bundle)
            bad = copy.deepcopy(bundle);bad['records'].pop('blaze_dense')
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                runner.validate_bundle(root,meta,bad)
            bad = copy.deepcopy(bundle);bad['records']['blaze_dense']['metrics']['average_precision_at_k'] = 1
            with self.assertRaisesRegex(ValueError,'metrics'):
                runner.validate_bundle(root,meta,bad)
            bad = copy.deepcopy(bundle);bad['sample']['dataset_row'] = 1
            with self.assertRaisesRegex(ValueError,'identity'):
                runner.validate_bundle(root,meta,bad)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):
                runner.validate_bundle(root,meta,bundle)


if __name__ == '__main__':
    unittest.main()
