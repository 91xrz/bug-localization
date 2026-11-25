import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.igsil import LocalDecoder, make_structure
from bug_localization.bm25 import ranking_metrics


class IgsilTests(unittest.TestCase):
    def test_structure_preserves_shared_candidate_paths_and_source(self):
        docs = [{'file_path': 'pkg/a.py', 'content': 'def f():\n    return 1\n'},
                {'file_path': 'tests/test_a.py', 'content': 'assert True\n'}]
        def parser(_, text):
            return [], [], text.splitlines()
        result = make_structure(docs, parser)
        self.assertEqual(result['pkg']['a.py']['text'], ['def f():', '    return 1'])
        self.assertEqual(result['tests']['test_a.py']['text'], ['assert True'])
        self.assertEqual(docs[0]['content'], 'def f():\n    return 1\n')

    def test_decoder_preserves_ordered_output_and_records_response(self):
        tokenizer = types.SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2, 3])
        result = {'choices': [{'finish_reason': 'stop', 'message': {'content': '```\na.py\nb.py\n```'}}],
                  'usage': {'completion_tokens': 5}}
        with tempfile.TemporaryDirectory() as directory:
            decoder = LocalDecoder(endpoint='http://localhost:8098/v1', alias='test', tokenizer=tokenizer,
                                   folder=Path(directory), temperature=0, max_tokens=2048)
            with patch('urllib.request.urlopen', return_value=io.StringIO(json.dumps(result))):
                response = decoder.codegen([{'role': 'user', 'content': 'report'}])
            self.assertEqual(response[0]['response'], result['choices'][0]['message']['content'])
            saved = json.loads((Path(directory) / 'call_001.json').read_text())
            self.assertEqual(saved['response'], result)
            self.assertEqual(saved['status'], 'ok')

    def test_truncated_generation_and_overflow_fail(self):
        tokenizer = types.SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1] * 4)
        reply = {'choices': [{'finish_reason': 'length', 'message': {'content': 'incomplete'}}]}
        with tempfile.TemporaryDirectory() as directory:
            decoder = LocalDecoder(endpoint='http://localhost:8098/v1', alias='test', tokenizer=tokenizer,
                                   folder=Path(directory), temperature=0, max_tokens=2048)
            with patch('urllib.request.urlopen', return_value=io.StringIO(json.dumps(reply))):
                with self.assertRaisesRegex(ValueError, 'finish normally'):
                    decoder.codegen('report')
            audit = json.loads((Path(directory) / 'call_001.json').read_text())
            self.assertEqual(audit['status'], 'failed')
            decoder.tokenizer = types.SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1] * 32768)
            with patch('urllib.request.urlopen') as urlopen:
                with self.assertRaisesRegex(ValueError, 'context'):
                    decoder.codegen('oversized report')
                urlopen.assert_not_called()

    def test_top5_does_not_invent_additional_results(self):
        metrics = ranking_metrics(['a.py', 'b.py', 'c.py', 'd.py', 'e.py'], ['z.py'], 10)
        self.assertEqual(metrics['top_10'], 0)
        self.assertEqual(metrics['average_precision_at_k'], 0)


if __name__ == '__main__':
    unittest.main()
