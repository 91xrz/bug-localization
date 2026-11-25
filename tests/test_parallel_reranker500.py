from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/experiments'))
from parallelize_reranker500 import export_partition, ordered_records
from bug_localization.batch import write_json
from bug_localization.reranking import read_json


class ParallelRerankerTests(unittest.TestCase):
    def sample(self, position):
        return {'position': position, 'dataset_row': position + 10, 'repository': 'repo',
                'issue_id': position, 'before_fix_sha': 'a' * 40}

    def test_merge_preserves_dataset_order_and_rejects_duplicate_missing_rows(self):
        a, b = self.sample(3), self.sample(4)
        pair_a = {'baseline': [a], 'reranked': [a]}
        pair_b = {'baseline': [b], 'reranked': [b]}
        self.assertEqual(ordered_records([pair_b, pair_a], [a, b])['reranked'], [a, b])
        for pairs in ([pair_a], [pair_a, pair_a]):
            with self.assertRaises(ValueError):
                ordered_records(pairs, [a, b])

    def test_export_freezes_only_committed_prefix_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            base = folder / 'new320'
            for kind in ('semantic', 'hypothesis'):
                (base / kind / 'cases/001').mkdir(parents=True)
            a, b = self.sample(1), self.sample(2)
            write_json(base / 'semantic/selection.json', {'samples': [a, b]})
            write_json(base / 'semantic/effective_config.json', {'semantic': {}})
            write_json(base / 'semantic/samples.json', {'q0': [a]})
            write_json(base / 'hypothesis/samples.json', {'direct_hypothesis': [a]})
            write_json(base / 'semantic/cases/001/checkpoint.json', {'records': {'q0': a, 'direct_hypothesis': a}})
            write_json(base / 'hypothesis/001_queries.json', {'query': 'hypothesis'})
            export_partition(folder, 'prefix', 0, 1)
            write_json(base / 'semantic/samples.json', {'q0': [a, b]})
            frozen = folder / 'parallel/prefix/semantic'
            self.assertEqual(read_json(frozen / 'samples.json')['q0'], [a])
            self.assertEqual(read_json(frozen / 'selection.json')['samples'], [a])
            self.assertEqual((frozen / 'cases').resolve(), (base / 'semantic/cases').resolve())
            with self.assertRaises(ValueError):
                export_partition(folder, 'tail', 1, 2)


if __name__ == '__main__':
    unittest.main()
