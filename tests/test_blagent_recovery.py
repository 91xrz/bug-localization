import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from bug_localization.blagent import parse_ranking,validate_scores
from bug_localization.blagent_recovery import guarded_tool,recover_scores,evidence_rerank


class Tokenizer:
    def encode(self,s,**kwargs):return list(s)
    def decode(self,ids,**kwargs):return ''.join(ids)


class Generator:
    tokenizer=Tokenizer()
    def __init__(self, replies=()):self.replies=iter(replies);self.prompts=[]
    def count_tokens(self,messages):return sum(len(m['content']) for m in messages)
    def complete(self,messages,**kwargs):
        self.prompts.append(messages)
        return next(self.replies)


class RecoveryTests(unittest.TestCase):
    def test_tool_error_does_not_read_outside_candidates_and_can_recover(self):
        seen=[];inspected=[];errors=[]
        with tempfile.TemporaryDirectory() as tmp:
            tool=guarded_tool(lambda p:seen.append(p) or 'def f()', ['a.py'],inspected,errors,Path(tmp))
            self.assertIn('No file was read',tool('../../gold.json'))
            self.assertEqual(seen,[])
            self.assertEqual(tool('"a.py"'),'def f()')
            self.assertEqual(seen,['a.py']);self.assertEqual(len(inspected),1)

    def test_scoring_repair_is_model_generated_and_validated(self):
        g=Generator(['{"outside.py":10}','{"b.py":9,"a.py":7}'])
        with tempfile.TemporaryDirectory() as tmp:
            scores,rank=recover_scores({'outside.py':10},'report',['a.py','b.py'],
                [{'file_path':'a.py','skeleton':'def f()'}],g,Path(tmp),{'score_repair_attempts':2},validate_scores)
            self.assertEqual(rank,['b.py','a.py']);self.assertEqual(len(g.prompts),2)
            self.assertEqual(scores,{'b.py':9,'a.py':7})

    def test_budget_exhaustion_finalizes_only_with_real_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'No successful'):
                recover_scores('Agent stopped due to iteration limit','r',['a.py'],[],Generator(),Path(tmp),{},validate_scores)
            g=Generator(['{"a.py":10}'])
            _,rank=recover_scores('Agent stopped due to iteration limit','r',['a.py'],
                                  [{'file_path':'a.py','skeleton':'f()'}],g,Path(tmp),{},validate_scores)
            self.assertEqual(rank,['a.py'])

    def test_repeated_invalid_outputs_are_not_silently_filtered(self):
        g=Generator(['{"bad.py":10}','{"bad.py":10}'])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'bounded correction'):
                recover_scores({'bad.py':10},'r',['a.py'],[{'file_path':'a.py','skeleton':'f()'}],g,Path(tmp),{},validate_scores)

    def test_long_ear_compares_every_file_and_preserves_all_anchors(self):
        paths=['a.py','b.py','c.py','d.py','e.py']
        class PairGenerator(Generator):
            def complete(self,messages,**kwargs):
                import re
                self.prompts.append(messages)
                present=re.findall('FILE:(.*?)\n',messages[-1]['content'])
                return json.dumps({'ranked_files':sorted(present,reverse=True)})
        g=PairGenerator()
        modules={'prompts':types.SimpleNamespace(EVIDENCE_ANCHORED_RERANKER_PROMPT='{problem_statement}\n{aggregated_code_text}'),
                 'agent':types.SimpleNamespace(extract_signature_tree=lambda s:'f()'),'ear_skeletons':dict.fromkeys(paths,'def f():')}
        # Use actual File marker from implementation in the test generator.
        def complete(messages,**kwargs):
            import re
            g.prompts.append(messages)
            self.assertLessEqual(g.count_tokens(messages)+100,2000)
            present=re.findall(r'### File: (.*?) ###',messages[-1]['content'])
            return json.dumps({'ranked_files':sorted(present,reverse=True)})
        g.complete=complete
        evidence={p:[{'code':'anchor_'+p,'start_line':1,'end_line':1}] for p in paths}
        with tempfile.TemporaryDirectory() as tmp:
            rank=evidence_rerank('report',paths,dict.fromkeys(paths,'x'*1500),evidence,modules,g,Path(tmp),
                {'max_context_tokens':2000,'max_new_tokens':100,'ear_temperature':0},parse_ranking)
            self.assertEqual(rank,paths[::-1])
            audit=json.loads((Path(tmp)/'ear_strategy.json').read_text())
            self.assertEqual(audit['mode'],'pairwise_merge')
            self.assertEqual(len(audit['compactions']),5)
            contexts=json.loads((Path(tmp)/'ear_effective_context.json').read_text())
            for p in paths:self.assertIn('anchor_'+p,contexts[p])

    def test_normal_ear_remains_one_call(self):
        g=Generator(['{"ranked_files":["b.py","a.py"]}'])
        m={'prompts':types.SimpleNamespace(EVIDENCE_ANCHORED_RERANKER_PROMPT='{problem_statement}\n{aggregated_code_text}')}
        with tempfile.TemporaryDirectory() as tmp:
            rank=evidence_rerank('r',['a.py','b.py'],{'a.py':'a','b.py':'b'},{},m,g,Path(tmp),
                                {'max_context_tokens':1000,'max_new_tokens':100,'ear_temperature':0},parse_ranking)
            self.assertEqual(rank,['b.py','a.py']);self.assertEqual(len(g.prompts),1)


if __name__=='__main__':unittest.main()
