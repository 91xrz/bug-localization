import itertools
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bug_localization.coefficient_search import evaluate_grid, metric, rank


class JointCoefficientChecks(unittest.TestCase):
    def test_every_grid_ranking_matches_independent_replay_with_changed_pools(self):
        rng=np.random.default_rng(41)
        paths=[f'file{i}.java' for i in range(80)]
        def ranking(order):
            return [{'rank':i+1,'file_path':p,'score':float(80-i)} for i,p in enumerate(order)]
        case={'status':'ok','a':ranking(paths),'b':ranking(paths[::-1]),
              'anchor':dict(zip(paths,rng.choice([0,.2,.8,1],len(paths)))),
              'max':dict(zip(paths,rng.integers(-3,4,len(paths)).astype(float))),
              'joint':dict(zip(paths,rng.integers(-3,4,len(paths)).astype(float)))}
        ab=list(itertools.product([0,.5,1],[0,.5,2]));c=list(itertools.product([0,.25,1],[0,4,16]))
        gold={'file0.java','file78.java','missing.java'}
        values,abvalues=evaluate_grid(case,gold,ab,c)
        for i,(alpha,gamma) in enumerate(ab):
            for j,(beta,lam) in enumerate(c):
                order,pool=rank(case,(alpha,gamma,beta,lam))
                np.testing.assert_allclose(values[i*len(c)+j],metric(order,gold),atol=1e-12,rtol=0)
                np.testing.assert_allclose(abvalues[i],metric(pool,gold),atol=1e-12,rtol=0)
                self.assertEqual(set(order),set(pool))
        self.assertNotEqual(set(rank(case,(0,0,0,0))[1]),set(rank(case,(1,0,0,0))[1]))

    def test_failed_reports_stay_zero_and_small_pools_work(self):
        v,a=evaluate_grid({'status':'failed'},['missing'],[(.5,.5)],[(.25,4)])
        self.assertFalse(v.any());self.assertFalse(a.any())
        case={'status':'ok','a':[{'rank':1,'file_path':'a','score':1.}],
              'b':[{'rank':1,'file_path':'a','score':1.}], 'anchor':{'a':0.},'max':{'a':0.},'joint':{'a':0.}}
        v,_=evaluate_grid(case,{'a','missing'},[(.5,.5)],[(.25,4)])
        np.testing.assert_allclose(v[0],metric(['a'],{'a','missing'}))

    def test_missing_reranker_score_is_rejected_not_zero_filled(self):
        case={'status':'ok','a':[{'rank':1,'file_path':'a','score':1.}],
              'b':[{'rank':1,'file_path':'a','score':1.}], 'anchor':{'a':0.},'max':{},'joint':{'a':0.}}
        with self.assertRaises(ValueError):
            evaluate_grid(case,{'a'},[(.5,.5)],[(.25,4)])


if __name__=='__main__':unittest.main()
