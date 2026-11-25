import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
from c_lora_round2_anchor import choose, metrics, summarize


def ranked(paths, scores):
    return [{"file_path":p, "semantic_rank":i, "score":s} for i, (p,s) in enumerate(zip(paths,scores), 1)]


class TestAnchoring(unittest.TestCase):
    def test_common_input_and_interpolation(self):
        base = ranked(["wrong","right"], [3,0])
        hard = ranked(["wrong","right"], [0,3])
        g = ["right","missing"]
        self.assertEqual(metrics(base,hard,g,.5)["average_precision_at_k"], .25)
        self.assertEqual(metrics(base,hard,g,1)["average_precision_at_k"], .5)
        with self.assertRaises(ValueError):
            metrics(base,hard[:1],g,0)

    def test_retention_guard_and_tie_break(self):
        def row(w, ap, lost, saved):
            return {"weight":w, "MAP@10":ap, "lost":lost, "rescued":saved}
        self.assertEqual(choose([row(0,.5,0,0), row(.25,.6,1,1), row(.5,.6,0,1), row(1,.9,2,2)]), .25)
        self.assertEqual(choose([row(0,.5,0,0), row(.5,.7,1,0)]), 0.)

    def test_failed_report_in_denominator(self):
        base = ranked(["a"], [1])
        reports = [{"base":base,"hard":base,"gold":["a"]}, {"base":[],"hard":[],"gold":["a"]}]
        self.assertEqual(summarize(reports,1)["MAP@10"], .5)
