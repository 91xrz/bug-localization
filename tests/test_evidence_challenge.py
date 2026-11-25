import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/experiments'))
import pilot_evidence_challenge as mod


class ChallengeTests(unittest.TestCase):
    def test_quote_requires_single_fragment_in_the_code_body(self):
        chunks=[{'text':'Path: magic_file.py\n# skip this line\n    return owner.value'}]
        self.assertTrue(mod.anchored('return owner.value',chunks))
        for q in ['magic_file.py','skip this line','owner','return owner\n.value','return self.value']:
            self.assertFalse(mod.anchored(q,chunks),q)

    def test_promotion_keeps_membership_tail_and_other_order(self):
        old=list('abcdefghijkl')
        self.assertEqual(mod.promote(old,'d'),list('dabcefghijkl'))
        with self.assertRaises(ValueError):
            mod.promote(old,'k')

    def fixture(self,root):
        arc,prop=root/'arc',root/'prop'
        src=arc/'cases/0001'
        src.mkdir(parents=True)
        dst=prop/'cases/0001'
        dst.mkdir(parents=True)
        s=dict(position=1,repository='r',issue_id=3,dataset_row=7,before_fix_sha='x',query_sha256='0'*64)
        head=list('abcdefghij')
        mod.write_json(src/'result.json',dict(ranking=head))
        mod.write_json(src/'record.json',dict(s,status='ok',result_sha256=mod.sha256_file(src/'result.json')))
        mod.write_json(src/'inference_input.json',dict(query='bug',candidates=head,
                       evidence={p:[dict(text='return object_'+p)] for p in head}))
        mod.write_json(dst/'rankings.json',dict(heads={'dual':head[1:]+head[:1]},
                       input_sha256=mod.sha256_file(src/'inference_input.json')))
        return s,arc,prop

    def test_downstream_verifier_cannot_approve_unanchored_or_vetoed_override(self):
        for quote,expected in [('return object_b',True),('return hallucination',False)]:
            with self.subTest(quote=quote),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                s,a,p=self.fixture(root)
                replies=[({'winner':'B'},'ok'),(dict(challenger_quote=quote,incumbent_quote='return object_a',
                          repair_and_contrast='handles branch',decision='overturn'),'ok'),
                         (dict(strongest_objection='both implementations plausible',verdict='veto'),'ok')]
                with patch.object(mod,'request',side_effect=replies) as requests:
                    t=mod.work(s,None,a,p,root/'output')
                self.assertEqual(requests.call_count,3 if expected else 2)
                self.assertEqual(t['promote']['evidence'],expected)
                self.assertFalse(t['promote']['verified'])
                self.assertFalse(t['promote']['unanimous'])
