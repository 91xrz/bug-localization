import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/experiments'))
import resume_reranker_full_dual as r

class DualResumeTests(unittest.TestCase):
    def test_assignments_disjoint_cover_only_pending(self):
        metas=[{'position':i} for i in range(1,8)]
        done={'F':{1:None,2:None,3:None},'R0':{1:None,2:None,4:None},'FULL':{1:None,2:None}}
        plan=r.assignments(metas,done)
        self.assertEqual(plan,{'0':[3,5,7],'1':[4,6]})
        self.assertFalse(set(plan['0'])&set(plan['1']))

    def test_workers_preserve_old_record_and_each_compute_assigned_case_once(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)/'run';src=Path(temp)/'source'
            metas=[{'position':i+1,'dataset_row':i,'repository':'x/y','issue_id':i} for i in range(2)]
            proto={'files_sha256':{},'config':{'source_run':str(src),'minimum_free_mib':1},'reranker':{},'reference_model':{}}
            r.s.save(folder/'protocol.json',proto);r.s.save(folder/'selection.json',{'samples':metas})
            r.s.save(folder/'dual_resume/protocol.json',{'runner_sha256':r.old.sha256_file(Path(r.__file__)),
                'original_files_sha256':{},'assignments':{'0':[1],'1':[2]}})
            r.s.save(src/'protocol.json',{'semantic':{'query_instruction':'test'}})
            inventory=[{'file_path':p,'sha256':p} for p in ['a','b']]
            chunks=[{'chunk_id':p,'file_path':p,'embedding_text':p} for p in ['a','b']]
            scores=[{'chunk_id':p,'file_path':p,'cosine_similarity':1/i} for i,p in enumerate(['a','b'],1)]
            r.s.save(src/'assets/test/candidates.json',{'files':inventory})
            r.s.save(src/'assets/test/chunks.json',chunks)
            result={'ranking':[{'file_path':p,'rank':i,'score':1/i} for i,p in enumerate(['a','b'],1)],
                'chunk_scores':scores,'chunk_scores_sha256':r.old.digest(scores),'query':'original report',
                'query_info':{'embedding_text':'Instruct: test\nQuery: original report'}}
            for meta in metas:
                for method in ['S0','H0']:
                    row=r.s.record(src,method,meta,result,inventory,['b'],[],assets='assets/test')
                    r.s.save(src/method/'cases'/f"{meta['position']:04d}"/'record.json',row)
            # The first sample's original rerank was completed before dual continuation.
            prior=r.s.record(folder,'R0',metas[0],result,inventory,['b'],[])
            r.s.save(folder/'R0/cases/0001/record.json',prior)
            calls=[]
            class Model:
                identity={}
                def __init__(self,*args):pass
                def score(self,query,documents):
                    calls.append((os.environ['CUDA_VISIBLE_DEVICES'],query))
                    return [{'score':1.0} for _ in documents]
            with patch.object(r.old,'LocalReranker',Model),patch.object(r.s,'gpu_memory',return_value={0:100,1:100}):
                for gpu in [0,1]:
                    with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':str(gpu)}):r.worker(folder,gpu)
                self.assertEqual(calls,[('0','original report'),('1','original report'),('1','original report')])
                self.assertEqual(r.s.read(folder/'R0/cases/0001/record.json'),prior)
                for gpu in [0,1]:
                    with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':str(gpu)}):r.worker(folder,gpu)
                self.assertEqual(len(calls),3)
            for method in r.METHODS:
                for meta in metas:self.assertIsNotNone(r.s.resumed(folder,method,meta))

if __name__=='__main__':unittest.main()
