import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments"))
from train_c_lora import group_loss
from train_c_lora_round3 import encode, objective, validate


class TrainingTaskTest(unittest.TestCase):
    def test_single_positive_list_term_matches_cross_entropy(self):
        scores = torch.tensor([.1, 1., 2.],requires_grad=True)
        flags = [False, True, False]
        expected = torch.nn.functional.cross_entropy(scores[None], torch.tensor([1]))
        loss = objective(scores,flags,"head")
        self.assertTrue(torch.allclose(loss,.8*expected+.2*group_loss(scores,flags)))
        loss.backward()
        self.assertGreater(scores.grad[-1].item(), scores.grad[0].item())

    def test_marginal_with_several_positives_is_offset_invariant(self):
        s = torch.tensor([1.,2.,3.])
        y = [True,False,True]
        self.assertTrue(torch.equal(objective(s,y,"pair"),group_loss(s,y)))
        self.assertAlmostEqual(objective(s,y,"head").item(),objective(s+10,y,"head").item(),places=5)
        with self.assertRaises(ValueError):
            objective(s,[False]*3,"head")

    def test_validation_preserves_pool_misses_and_uses_retrieval_ties(self):
        rr = SimpleNamespace(score=lambda q,doc: [{"score":1} for _ in doc])
        rows = [{"id":"a","query":"q","gold":["x"],"files":[
                    {"path":p,"chunks":[p],"semantic_rank":r} for p,r in [("z",2),("x",1)]]},
                {"id":"b","query":"","gold":["y"],"files":[]}]
        self.assertEqual(validate(rr,rows)["MAP@10"],.5)

    def test_no_silent_truncation(self):
        rr = SimpleNamespace(config={"instruction":"i","max_tokens":5}, prefix=[1],suffix=[2],
            tokenizer=lambda texts, **kw: {"input_ids":[[3,4,5,6] for t in texts]})
        row={"id":"i","query":"q","gold":["a"],"files":[{"path":p,"chunks":["x"]} for p in ("a","b")]}
        with self.assertRaisesRegex(ValueError,"too long"):
            encode(rr,row)
