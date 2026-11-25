from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/experiments"))
from prepare_c_lora_round3 import evidence, ranking_metrics, select


def fixture(gold):
    paths = list("abcdefgh")
    rows = [{"file_path":p,"rank":i+1,"semantic_rank":i+1,"score":8-i} for i,p in enumerate(paths)]
    blocks = [{"file_path":p,"chunk_id":p,"embedding_text":p,"score":r["score"]}
              for p,r in zip(paths,rows)]
    return dict(ranking=rows,scored_chunks=blocks,candidates=[{"file_path":p} for p in paths],
                ground_truth=gold,metrics=ranking_metrics(paths,gold,10))


class RealPoolsTest(unittest.TestCase):
    def test_only_supplement_in_pool_positive(self):
        files,_,stats=evidence(fixture(["h","missing"]),True)
        self.assertEqual([f["path"] for f in files],list("abcdefh"))
        self.assertEqual(stats["pool_visible_gold"],1)
        files,_,stats=evidence(fixture(["missing"]),True)
        self.assertFalse(any(f["path"]=="missing" for f in files))
        self.assertIsNone(stats["supplemental_positive"])

    def test_unfiltered_validation_and_score_check(self):
        case=fixture(["missing"])
        files,order,stats=evidence(case,False)
        self.assertEqual(len(files),8)
        self.assertEqual(order,list(range(8)))
        self.assertFalse(stats["base_top1_correct"])
        case["scored_chunks"][0]["score"]-=1
        with self.assertRaises(ValueError):
            evidence(case,False)

    def test_reserved_identity_and_body_excluded_before_selection(self):
        def row(i):
            return dict(repository="r",issue_id=i,dataset_row=i,before_fix_sha=str(i))
        rows=[row(i) for i in range(5)]
        result=select(rows,[rows[0]],{"r":2},["same","same","unique","unique","another"])
        self.assertEqual({r["issue_id"] for r in result} & {0,1},set())
        self.assertEqual(select(rows[::-1],[rows[0]],{"r":2},["same","same","unique","unique","another"]),result)
