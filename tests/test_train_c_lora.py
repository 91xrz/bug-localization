import copy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
import train_c_lora as training


class LoRATrainingTests(unittest.TestCase):
    def test_frozen_base_zero_init_and_adapter_roundtrip(self):
        torch.manual_seed(5)
        model = nn.ModuleDict({"q_proj": nn.Linear(3, 5), "v_proj": nn.Linear(3, 5),
                               "k_proj": nn.Linear(3, 5)})
        pristine = copy.deepcopy(model)
        x = torch.rand(2, 3)
        targets = training.install(model, 2, 4)
        self.assertEqual(targets, ["q_proj", "v_proj"])
        self.assertTrue(torch.equal(model["q_proj"](x), pristine["q_proj"](x)))
        self.assertFalse(model["k_proj"].weight.requires_grad)
        model["q_proj"](x).sum().backward()
        self.assertGreater(model["q_proj"].lora_b.grad.abs().sum().item(), 0)
        with torch.no_grad():
            model["q_proj"].lora_b.add_(.1)
        def rr(m):
            return SimpleNamespace(model=m, identity={"files_sha256": {"base": "test"}},
                                   config={"instruction": "test"})
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            training.dump(root / "protocol.json", {})
            training.save_adapter(rr(model), root / "adapter", {"rank": 2, "alpha": 4}, targets)
            training.load_adapter(rr(pristine), root / "adapter")
            self.assertTrue(torch.equal(model["q_proj"](x), pristine["q_proj"](x)))
            self.assertFalse(any(p.requires_grad for p in pristine.parameters()))

    def test_group_gradient_matches_direct_file_max(self):
        p = nn.Parameter(torch.tensor([1.5, .2, 2., 8., .7, 1.]))
        group = {"ids": [[x] for x in range(6)], "sizes": [2, 2, 2],
                 "positives": [True, False, True]}
        def logits(rr, ids):
            return p[[a[0] for a in ids]]
        loss = training.group_loss(p.reshape(3, 2).max(dim=1).values, group["positives"])
        wanted, = torch.autograd.grad(loss / 4, p)
        for size in (1, 1100):
            group["ids"] = [[i] * size for i in range(6)]
            p.grad = None
            with patch.object(training, "logits", logits):
                value = training.train_report(None, group, 4)
            self.assertEqual(value, loss.item())
            self.assertTrue(torch.equal(p.grad, wanted))
            self.assertLess(p.grad[0].item(), 0)
            self.assertGreater(p.grad[3].item(), 0)

    def test_group_must_have_both_labels(self):
        with self.assertRaises(ValueError):
            training.group_loss(torch.tensor([1., 1.]), [True, True])


if __name__ == "__main__":
    unittest.main()
