from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/experiments"))
from pilot_closed_exploration import shortlist, evidence, update


class Chars:
    def encode(self, text, **kwargs): return list(text)


class TestClosedExploration(unittest.TestCase):
    def test_disambiguate_owner(self):
        lines=["class Parent:","    def save(self):","        pass","class Child(Parent):",
               "    def save(self):","        return 42"]
        options=shortlist(lines,"Child.save fails","", "A")
        names={v["owner"]:k for k,v in options.items()}
        a=names["Child.save"]
        self.assertNotEqual(a,names["Parent.save"])
        e=evidence(lines,options[a],a,Chars())
        self.assertTrue(e["complete"])
        self.assertEqual((e["start"],e["end"],e["text"]),(5,6,"\n".join(lines[4:])))

    def test_report_truncation_of_body(self):
        e=evidence(["def save():","    return '"+("x"*500)+"'"],{"owner":"save","start":1,"end":2},"A1",Chars())
        self.assertFalse(e["complete"])
        self.assertEqual(e["text"],"def save():")

    def test_inventory_invariants(self):
        ranks=[str(i) for i in range(40)]
        for i in range(10):
            new=update(ranks,str(i),"promote")
            self.assertEqual(set(new[:10]),set(ranks[:10]))
            self.assertEqual(new[10:],ranks[10:])
            self.assertEqual(new[1:],[p for p in ranks if p!=str(i)])
        self.assertEqual(update(ranks,"2","abstain"),ranks)
        for invalid in ("20","external"):
            with self.assertRaises(ValueError): update(ranks,invalid,"promote")
