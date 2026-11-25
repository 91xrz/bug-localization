import importlib.util
from pathlib import Path
import unittest

PROJECT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pilot_c3_full_report',PROJECT/'scripts/experiments/pilot_c3_full_report.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


class FullReportTests(unittest.TestCase):
    def test_same_chunks_and_file_max_with_semantic_ties_and_tail(self):
        saved={'ground_truth':['forbidden.py'],'scored_chunks':[
            {'file_path':'b.py','chunk_id':'b1','embedding_text':'code b1','score':999},
            {'file_path':'a.py','chunk_id':'a1','embedding_text':'code a1','score':-999},
            {'file_path':'a.py','chunk_id':'a2','embedding_text':'code a2','score':-999}],
            'ranking':[{'file_path':'b.py','semantic_rank':2,'semantic_score':.7},
                       {'file_path':'a.py','semantic_rank':1,'semantic_score':.8},
                       {'file_path':'tail.py','semantic_rank':3,'semantic_score':.5}]}
        order,chunks=pilot.evidence_inputs(saved)
        self.assertEqual([c['chunk_id'] for c in chunks],['b1','a1','a2'])
        self.assertTrue(all(set(c)=={'file_path','chunk_id','embedding_text'} for c in chunks))
        class Model:
            def score(self,query,documents):
                self.query=query;self.documents=documents
                return [{'score':s,'probability':.5,'input_tokens':7} for s in [2,1,2]]
        model=Model();query='complete report\nimportant last line'
        ranking,scored=pilot.score_evidence(query,order,chunks,model)
        self.assertEqual(model.query,query)
        self.assertEqual(model.documents,['code b1','code a1','code a2'])
        self.assertEqual([r['file_path'] for r in ranking],['a.py','b.py','tail.py'])
        self.assertEqual(ranking[0]['best_chunk_id'],'a2')
        self.assertFalse(ranking[-1]['reranked'])

    def test_duplicate_evidence_is_rejected(self):
        c={'file_path':'a.py','chunk_id':'a','embedding_text':'code'}
        with self.assertRaisesRegex(ValueError,'Duplicate C3 evidence'):
            pilot.evidence_inputs({'scored_chunks':[c,c],'ranking':[]})

    def test_context_failure_propagates_without_fallback_or_truncation(self):
        class Model:
            def score(self,query,documents):
                self.query=query
                raise ValueError('Reranker input exceeds max_tokens')
        model=Model();query='x'*9000
        with self.assertRaisesRegex(ValueError,'exceeds max_tokens'):
            pilot.score_evidence(query,[],[],model)
        self.assertEqual(model.query,query)


if __name__=='__main__':unittest.main()
