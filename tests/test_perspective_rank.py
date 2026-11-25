import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments'))
from pilot_perspective_rank import DEV, OUT, cohort, fuse


class ConfirmationSelectionTest(unittest.TestCase):
    def setUp(self):
        self.samples = [dict(repository=f'r{i//10}',issue_id=i,dataset_row=i,position=i) for i in range(90)]

    def load(self, old):
        contents = {DEV/'confirmation90.json':{'samples':self.samples},OUT/'protocol.json':{'samples':old}}
        with patch('pilot_perspective_rank.read', side_effect=contents.__getitem__):
            return cohort('confirmation90')

    def test_frozen_order_and_separate_output(self):
        rows,archive,output = self.load([])
        self.assertEqual(rows,self.samples)
        self.assertEqual(archive,DEV/'confirm_joint10')
        self.assertNotEqual(output,OUT)

    def test_overlapping_issue_rejected_even_if_position_differs(self):
        with self.assertRaisesRegex(ValueError,'overlapping'):
            self.load([dict(self.samples[0],position=-1,dataset_row=-1)])

    def test_wrong_project_counts_rejected(self):
        self.samples[0]['repository'] = 'r1'
        with self.assertRaisesRegex(ValueError,'ten reports'):
            self.load([])


class PerspectiveFusionTest(unittest.TestCase):
    def test_shared_winner_then_prior_order_on_ties(self):
        self.assertEqual(fuse(['c','a','b'], ['c','b','a'], ['b','a','c']), ['c','b','a'])

    def test_reject_changed_membership_or_duplicates(self):
        for b in (['x','a'], ['a','a'], ['b']):
            with self.subTest(b=b), self.assertRaises(ValueError):
                fuse(['a','b'], b, ['a','b'])

    def test_fusing_does_not_depend_on_branch_order(self):
        a,b,c = ['d','c','a','b'], ['c','a','b','d'], ['b','a','c','d']
        self.assertEqual(fuse(a,b,c), fuse(b,a,c))
        self.assertEqual(set(fuse(a,b,c)), set(c))
