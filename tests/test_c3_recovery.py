import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT / 'scripts/experiments'))
import finalize_c3_supplement as recovery
from bug_localization.batch import write_json
from bug_localization.bm25 import ranking_metrics


class RecoveryTests(unittest.TestCase):
    def test_reconstructs_coverage_without_touching_raw_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); target = folder / 'A3'; target.mkdir()
            record = {'repository': 'sample/repo', 'issue_id': 1, 'status': 'ok',
                'metrics': ranking_metrics(['a'], ['a', 'missing'], 10)}
            write_json(target / 'samples.json', [record]); write_json(folder / 'protocol.json', {'frozen': True})
            original = (target / 'samples.json').read_bytes()
            source = SimpleNamespace(samples=[{}], load=lambda _: {'gold': ['a', 'missing'], 'inventory': {'a'}})
            with patch.object(recovery, 'freeze', return_value={'config': {}}), \
                 patch.object(recovery, 'Sources', return_value=source), \
                 patch.object(recovery, 'audit_cells') as audit, \
                 patch.object(recovery.subprocess, 'run'):
                recovery.complete(folder, 'factorial')
                audit.assert_called_once()
            self.assertEqual(original, (target / 'samples.json').read_bytes())
            normalized = json.loads((target / 'samples_with_coverage.json').read_text())[0]
            self.assertEqual(normalized['metrics'], record['metrics'])
            self.assertEqual(normalized['missing_gold'], ['missing'])
            summary = json.loads((target / 'summary.json').read_text())
            self.assertEqual(summary['coverage']['candidate_coverage'], .5)
            self.assertEqual(summary['metrics']['MAP@10'], .5)

if __name__ == '__main__':
    unittest.main()
