import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from bug_localization.agent_review import apply_verdict, checked_lines, excerpt, menu


class CharTokenizer:
    def encode(self,text,**kw):
        return list(text)


class ReviewTests(unittest.TestCase):
    def test_cited_challenger_only(self):
        original=["b","c","a","d"]
        ev=[{"lines":[4],"novel":[4]},{"lines":[3],"novel":[]}]
        v={"prefer":"A","line_A":4,"line_B":3}
        self.assertEqual(apply_verdict(original,["a","b"],ev,v),(["a","b","c","d"],"promoted"))
        self.assertEqual(apply_verdict(original,["a","b"],ev,{**v,"line_A":2}),(original,"invalid_citation"))
        self.assertEqual(apply_verdict(original,["a","b"],ev,{"prefer":"abstain"}),(original,"abstained"))
        ev[0]["novel"]=[]
        self.assertEqual(apply_verdict(original,["a","b"],ev,v),(original,"no_new_cited_line"))

    def test_windows_are_real_and_bounded(self):
        lines=["def save(item):","    item.save_m2m()","    return item"]
        options,fixed=menu(lines,"save_m2m fails")
        self.assertEqual(options[fixed]["action"],"reference_window")
        e=excerpt(lines,options[fixed],CharTokenizer(),"    return item",60)
        self.assertLessEqual(len(e["text"]),60)
        for row in e["text"].splitlines():
            n, line=row.split(": ",1)
            self.assertEqual(line,lines[int(n)-1])
        self.assertNotIn(3,e["novel"])
        self.assertIn(2,e["novel"])

    def test_snapshot_guard(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            (root/"a").write_bytes(b"abc")
            self.assertEqual(checked_lines(root,"a",hashlib.sha256(b"abc").hexdigest()),["abc"])
            for path,checksum in [("a","broken"),("../a","broken")]:
                with self.assertRaises(ValueError):
                    checked_lines(root,path,checksum)
