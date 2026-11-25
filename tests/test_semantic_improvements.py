import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/experiments'))
from bug_localization.semantic_improvements import aggregation_scores, feedback_scores, grounded_scores, rank
from run_a_improvements import select


class SemanticImprovementTests(unittest.TestCase):
    def test_length_correction_and_normalized_lse(self):
        config={'aggregation':{'top_mean_k':[2,3],'max_top3_mix':[.25],
                               'lse_temperature':[.1]},'length_penalty':[.01]}
        rows=[{'file_path':p,'cosine_similarity':s} for p,ss in
              [('small',[.8]),('large',[.8,.8,.8])] for s in ss]
        scores=aggregation_scores(rows,config)
        self.assertAlmostEqual(scores['lse_t0.1']['small'],scores['lse_t0.1']['large'])
        self.assertGreater(scores['length_p0.01']['small'],scores['length_p0.01']['large'])
        self.assertEqual(scores['top3_mean']['small'],.8)

    def test_deeper_pool_can_recover_exact_symbol_without_labels(self):
        rows=[{'file_path':f'pkg/f{i}.py','score':1-i/100,'rank':i+1} for i in range(60)]
        symbols={r['file_path']:set() for r in rows};symbols['pkg/f55.py']={'BrokenOperation'}
        shallow,_,_=grounded_scores(rows,'BrokenOperation',symbols,1.,50)
        deep,_,_=grounded_scores(rows,'BrokenOperation',symbols,1.,200)
        self.assertNotIn('pkg/f55.py',shallow)
        self.assertIn('pkg/f55.py',rank(deep))
        noise,_,_=grounded_scores(rows,'<!-- BrokenOperation -->',symbols,1.,200)
        self.assertEqual(rank(noise,{r['file_path']:r['rank'] for r in rows}),[r['file_path'] for r in rows[:50]])

    def test_feedback_zero_weight_reproduces_cosine_and_can_drift(self):
        query=np.array([1.,0.]);vectors=np.array([[.8,.6],[.9,-np.sqrt(.19)]])
        np.testing.assert_allclose(feedback_scores(query,vectors,1,0),vectors@query)
        updated=feedback_scores(query,vectors,1,.5)
        self.assertGreater(updated[0],updated[1])
        with self.assertRaises(ValueError):feedback_scores(query,np.array([[float('nan'),0.]]),1,.2)

    def test_selection_rejects_gain_that_loses_recall(self):
        base={'MAP@10':.5,'MRR@10':.6,'Hit@10':.8,'Hit@50':.9,'MeanRecall@50':.85}
        bad={**base,'MAP@10':.7,'Hit@50':.89}
        good={**base,'MAP@10':.55}
        result=select({'A':base,'anchor_d50_w0.5':good,'anchor_d200_w1':bad})
        self.assertEqual(result['selected'],'anchor_d50_w0.5')
        self.assertFalse(result['family_eligible']['anchor'])
        self.assertEqual(select({'A':base,'anchor_d200_w1':bad})['selected'],'A')


if __name__=='__main__':unittest.main()
