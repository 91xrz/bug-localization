import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('current_baseline', ROOT/'scripts/retrieval/run_current_baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class FrozenBaselineChecks(unittest.TestCase):
    def register(self, folder):
        implementation = folder/'implementation.py'
        implementation.write_text('version = 1\n')
        config = folder/'version.json'
        config.write_text(json.dumps({'id':'ab_structure_v1','alpha':.5,'anchor_weight':.5,
            'top_n_files':50,'method':'score_a0.5_e0.5',
            'files_sha256':{str(implementation):baseline.sha256(implementation)}}))
        registry = folder/'current.json'
        registry.write_text(json.dumps({'initial_ranking':{'config':str(config),'sha256':baseline.sha256(config)}}))
        return registry, config, implementation

    def test_changed_parameters_rejected_by_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry, config, _ = self.register(Path(tmp))
            with patch.object(baseline,'REGISTRY',registry):
                self.assertEqual(baseline.load_current()[0]['alpha'],.5)
                changed=json.loads(config.read_text());changed['alpha']=.25
                config.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'configuration changed'):
                    baseline.load_current()

    def test_implementation_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry, _, implementation = self.register(Path(tmp))
            implementation.write_text('version = 2\n')
            with patch.object(baseline,'REGISTRY',registry):
                with self.assertRaisesRegex(ValueError,'dependency changed'):
                    baseline.load_current()

    def test_missing_shared_union_features_are_rejected(self):
        rows=[{'rank':1,'file_path':'one.py','score':1.},
              {'rank':2,'file_path':'two.py','score':.5}]
        with self.assertRaisesRegex(ValueError,'exactly A50 union B50'):
            baseline.rank_fixed({'a':rows,'b':rows,'anchor':{'one.py':0.}},
                                {'alpha':.5,'anchor_weight':.5,'method':'score_a0.5_e0.5'})


if __name__ == '__main__':
    unittest.main()
