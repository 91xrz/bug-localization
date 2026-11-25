import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.genloc_runtime import LocalToolGenerator


TOOL = {'type': 'function', 'function': {'name': 'search_file', 'parameters': {
    'type': 'object', 'properties': {'filename': {'type': 'string'}}, 'required': ['filename']}}}
CALL = {'id': 'call_1', 'type': 'function',
        'function': {'name': 'search_file', 'arguments': '{"filename":"a.py"}'}}


class RuntimeTests(unittest.TestCase):
    def generator(self, folder):
        generator = LocalToolGenerator.__new__(LocalToolGenerator)
        generator.config = {'max_calls': 9, 'max_new_tokens': 20,
                            'max_context_tokens': 100, 'seed': 42, 'temperature': 1.0}
        generator.endpoint = 'http://127.0.0.1:8101'
        generator.model_alias = 'genloc-test'
        generator.identity = {'model': 'fixed'}
        generator.start_case(folder)
        return generator

    def reply(self, message=None, finish='tool_calls', prompt_tokens=5):
        return {'choices': [{'finish_reason': finish, 'message': message or {
            'role': 'assistant', 'content': None, 'tool_calls': [CALL]}}],
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': 7},
            'timings': {'predicted_ms': 4}}

    def test_native_tool_protocol_checkpoint_and_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            generator = self.generator(folder)
            with patch.object(generator, '_count_payload_tokens', return_value=(5, 'prompt')), \
                 patch('bug_localization.genloc_runtime.request_json', return_value=self.reply()) as transport:
                result = generator.complete([{'role': 'user', 'content': 'find a.py'}], [TOOL], 'required')
                self.assertEqual(result['tool_calls'], [CALL])
                self.assertEqual(transport.call_args.args[1], '/v1/chat/completions')
                payload = transport.call_args.args[2]
                self.assertEqual(payload['tool_choice'], 'required')
                self.assertNotIn('response_format', payload)
                generator.start_case(folder)
                self.assertEqual(generator.complete([{'role': 'user', 'content': 'find a.py'}], [TOOL], 'required'), result)
                self.assertEqual(transport.call_count, 1)
            record = json.loads((Path(folder) / 'call_001.json').read_text())
            self.assertEqual(record['seed'], payload['seed'])
            self.assertEqual(record['input_tokens'], 5)
            self.assertEqual(record['output_tokens'], 7)
            self.assertIn('elapsed_seconds', record)

    def test_overflow_never_sends_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            generator = self.generator(folder)
            with patch.object(generator, '_count_payload_tokens', return_value=(81, 'large prompt')), \
                 patch('bug_localization.genloc_runtime.request_json') as transport:
                with self.assertRaisesRegex(ValueError, 'overflow'):
                    generator.complete([], [TOOL], 'auto')
                transport.assert_not_called()
            self.assertEqual(json.loads((Path(folder) / 'call_001.json').read_text())['status'], 'failed')

    def test_schema_never_silently_disables_required_tools(self):
        with tempfile.TemporaryDirectory() as folder:
            generator = self.generator(folder)
            with patch('bug_localization.genloc_runtime.request_json') as transport:
                for choice in ['required', 'auto']:
                    with self.assertRaisesRegex(ValueError, 'schema requires'):
                        generator.complete([], [TOOL], choice, {'type': 'object'})
                transport.assert_not_called()

    def test_reject_truncation_missing_required_tool_and_token_mismatch(self):
        replies = [self.reply(finish='length'),
                   self.reply({'role': 'assistant', 'content': 'answer'}, finish='stop'),
                   self.reply(prompt_tokens=6)]
        for reply in replies:
            with self.subTest(reply=reply), tempfile.TemporaryDirectory() as folder:
                generator = self.generator(folder)
                with patch.object(generator, '_count_payload_tokens', return_value=(5, 'prompt')), \
                     patch('bug_localization.genloc_runtime.request_json', return_value=reply):
                    with self.assertRaises(ValueError):
                        generator.complete([], [TOOL], 'required')

    def test_final_schema_validated_and_preserved(self):
        schema = {'type': 'object', 'properties': {'file': {'type': 'string'}},
                  'required': ['file'], 'additionalProperties': False}
        with tempfile.TemporaryDirectory() as folder:
            generator = self.generator(folder)
            reply = self.reply({'role': 'assistant', 'content': '{"file":"a.py"}'}, finish='stop')
            with patch.object(generator, '_count_payload_tokens', return_value=(5, 'prompt')), \
                 patch('bug_localization.genloc_runtime.request_json', return_value=reply) as transport:
                result = generator.complete([], [TOOL], 'none', response_schema=schema, stage='final')
                self.assertEqual(json.loads(result['content']), {'file': 'a.py'})
                self.assertEqual(transport.call_args.args[2]['response_format']['json_schema']['schema'], schema)

    def test_ninth_call_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            generator = self.generator(folder)
            generator.calls = 9
            with self.assertRaisesRegex(ValueError, 'budget exceeded'):
                generator.complete([], [TOOL], 'none')


if __name__ == '__main__':
    unittest.main()
