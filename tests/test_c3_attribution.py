from pathlib import Path
import sys
import unittest
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'scripts/analysis'))
from attribute_c3_failures import classify_file

class AttributionTests(unittest.TestCase):
    def test_unscored_tail_is_recall_failure(self):
        self.assertEqual(classify_file('gold', {'gold','other'}, {'other'}, {'other':1,'gold':51}), 'recall_gap')
    def test_missing_source_is_not_semantic_recall_failure(self):
        self.assertEqual(classify_file('new_file', {'old_file'}, {'old_file'}, {'old_file':1}), 'source_inventory_gap')
    def test_selected_gold_file_does_not_prove_relevant_chunk(self):
        self.assertEqual(classify_file('gold', {'gold'}, {'gold'}, {'gold':11}), 'selection_or_reranking_unresolved')
    def test_cutoff_inclusive(self):
        self.assertEqual(classify_file('gold', {'gold'}, {'gold'}, {'gold':10}), 'retrieved_at_k')

if __name__=='__main__': unittest.main()
