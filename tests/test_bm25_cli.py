"""Unified lexical entry point must preserve cohort and failure behavior."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bm25_cli', ROOT / 'scripts/retrieval/run_bm25.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class CliTests(unittest.TestCase):
    def test_single_report_still_uses_original_config(self):
        with patch('sys.argv', ['run_bm25']), patch.object(cli, 'run') as run:
            cli.main()
        self.assertEqual(run.call_args.args[1]['repository'], 'psf/requests')

    def test_lca_batch_keeps_all_repositories_and_count(self):
        import json
        path = 'configs/retrieval/lca/py/comparison.json'
        expected = json.loads((ROOT / path).read_text())
        with patch('sys.argv', ['run_bm25', '--batch', '--config', path]), patch.object(cli, 'run_batch', return_value=(None, {'failed_samples': 0})) as run:
            cli.main()
        self.assertEqual(run.call_args.args[2:], (expected['repositories'], 50))

    def test_override_and_failure_exit(self):
        with patch('sys.argv', ['run_bm25', '--batch', '--limit', '3', '--repository', 'repo']), patch.object(cli, 'run_batch', return_value=(None, {'failed_samples': 1})) as run:
            with self.assertRaises(SystemExit) as error:
                cli.main()
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(run.call_args.args[2:], (['repo'], 3))
