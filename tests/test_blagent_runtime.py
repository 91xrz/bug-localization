import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from bug_localization.blagent_runtime import ServerGenerator


class FakeTokenizer:
    def apply_chat_template(self,messages,**kwargs):return messages[0]['content']
    def encode(self,text,**kwargs):return list(range(len(text)))


class RuntimeTests(unittest.TestCase):
    def generator(self,folder):
        g=ServerGenerator.__new__(ServerGenerator)
        g.config={'max_calls':3,'max_new_tokens':20,'max_context_tokens':100,'seed':42}
        g.identity={'fixed':'model'};g.endpoint='http://127.0.0.1:8099';g.tokenizer=FakeTokenizer();g.start_case(folder)
        return g

    def test_server_audits_seed_tokens_and_uses_exact_checkpoint(self):
        reply={'content':'answer','tokens_predicted':3,'truncated':False,'stop_type':'eos','timings':{'predicted_ms':1}}
        with tempfile.TemporaryDirectory() as tmp:
            g=self.generator(Path(tmp))
            with patch('bug_localization.blagent_runtime.request_json',return_value=reply) as transport:
                self.assertEqual(g.complete([{'role':'user','content':'abc'}]),'answer')
                payload=transport.call_args.args[2]
                self.assertEqual(payload['prompt'],[0,1,2]);self.assertTrue(payload['cache_prompt'])
                g.start_case(Path(tmp));self.assertEqual(g.complete([{'role':'user','content':'abc'}]),'answer')
                self.assertEqual(transport.call_count,1)
            record=json.loads((Path(tmp)/'call_001.json').read_text())
            self.assertEqual(record['seed'],payload['seed']);self.assertEqual(record['server_response'],reply)

    def test_truncated_or_limit_responses_are_failed(self):
        for truncated,stop in [(True,'eos'),(False,'limit')]:
            with tempfile.TemporaryDirectory() as tmp:
                g=self.generator(Path(tmp))
                with patch('bug_localization.blagent_runtime.request_json',return_value={'content':'x','truncated':truncated,'stop_type':stop}):
                    with self.assertRaisesRegex(ValueError,'truncated'):
                        g.complete([{'role':'user','content':'abc'}])
                self.assertEqual(json.loads((Path(tmp)/'call_001.json').read_text())['status'],'failed')

    def test_context_overflow_never_sends_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            g=self.generator(Path(tmp))
            with patch('bug_localization.blagent_runtime.request_json') as transport:
                with self.assertRaisesRegex(ValueError,'overflow'):
                    g.complete([{'role':'user','content':'a'*81}])
                transport.assert_not_called()

    def test_schema_constrains_server_and_is_part_of_checkpoint_identity(self):
        schema={'type':'object','properties':{'a.py':{'type':'number'}},'additionalProperties':False}
        reply={'content':'{"a.py":9}','tokens_predicted':8,'truncated':False,'stop_type':'eos'}
        with tempfile.TemporaryDirectory() as tmp:
            g=self.generator(Path(tmp))
            with patch('bug_localization.blagent_runtime.request_json',return_value=reply) as transport:
                g.complete([{'role':'user','content':'abc'}],stage='SAS_repair',response_schema=schema)
                self.assertEqual(transport.call_args.args[2]['json_schema'],schema)
            record=json.loads((Path(tmp)/'call_001.json').read_text())
            self.assertEqual(record['request']['response_schema'],schema)
            g.start_case(Path(tmp))
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                g.complete([{'role':'user','content':'abc'}],stage='SAS_repair')

    def test_request_seed_independent_of_folder_or_call_order(self):
        reply={'content':'ok','tokens_predicted':1,'truncated':False,'stop_type':'word'}
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            ga,gb=self.generator(Path(a)),self.generator(Path(b));gb.calls=1
            with patch('bug_localization.blagent_runtime.request_json',return_value=reply) as transport:
                for g in (ga,gb):g.complete([{'role':'user','content':'abc'}])
                self.assertEqual(transport.call_args_list[0].args[2]['seed'],transport.call_args_list[1].args[2]['seed'])

if __name__=='__main__':unittest.main()
