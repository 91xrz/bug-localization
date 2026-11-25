import io
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import types
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.rrl import BudgetExceeded, LocalChat, safe_candidate_path, validate_config


class RrlTests(unittest.TestCase):
    @unittest.skipUnless((PROJECT / '.runtime/rrl-deps/pyserini').exists(), 'RRL optional environment is not prepared')
    def test_lucene_empty_candidate_is_audited_without_changing_shared_candidates(self):
        code = '''
from pathlib import Path
import tempfile
from bug_localization.rrl_index import prepare_index, search_index
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    documents = [{'file_path': 'code.py', 'content': 'cookies request session'},
                 {'file_path': 'empty.py', 'content': ''}]
    audit = prepare_index(Path.cwd(), root, documents)
    assert audit['candidate_files'] == 2
    assert audit['indexed_files'] == 1
    assert audit['empty_files_omitted_by_lucene'] == ['empty.py']
    assert search_index(root, 'cookies', 10)[0] == ['code.py']
    assert prepare_index(Path.cwd(), root, documents)['cache_hit']
'''
        import os
        env = {**os.environ, 'PYTHONPATH': str(PROJECT / 'src')}
        completed = subprocess.run([sys.executable, '-c', code], cwd=PROJECT, env=env,
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def config(self):
        return json.loads((PROJECT / 'configs/retrieval/rrl.json').read_text())

    def test_tool_reads_require_candidate_membership_and_real_path_containment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / 'repo'
            repository.mkdir()
            (repository / 'code.py').write_text('print(1)')
            (root / 'labels.json').write_text('gold data')
            (repository / 'link.py').symlink_to(root / 'labels.json')
            allowed = {'code.py', 'link.py', '../labels.json'}
            self.assertEqual(safe_candidate_path(repository, 'code.py', allowed), str(repository / 'code.py'))
            for path in ['../labels.json', 'link.py', str(root / 'labels.json'), 'unknown.py', None]:
                self.assertIsNone(safe_candidate_path(repository, path, allowed))

    def test_local_chat_budget_truncation_and_saved_raw_response(self):
        config = {**self.config(), 'max_calls': 1}
        tokenizer = types.SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2, 3])
        result = {'choices': [{'finish_reason': 'length', 'message': {'content': '{"ranked_files":['}}]}
        with tempfile.TemporaryDirectory() as temporary:
            chat = LocalChat(config, tokenizer, Path(temporary))
            with patch('urllib.request.urlopen', return_value=io.StringIO(json.dumps(result))) as request:
                chat([{'role': 'user', 'content': 'report only'}])
                sent = json.loads(request.call_args.args[0].data)
                self.assertEqual(sent['messages'], [{'role': 'user', 'content': 'report only'}])
                self.assertEqual(sent['max_tokens'], 1024)
            self.assertEqual(chat.truncated_calls, [1])
            saved = json.loads((Path(temporary) / 'call_001.json').read_text())
            self.assertEqual(saved['response'], result)
            # Author loops catch Exception; budget abort must escape those loops.
            self.assertFalse(issubclass(BudgetExceeded, Exception))
            with self.assertRaises(BudgetExceeded):
                chat([{'role': 'user', 'content': 'again'}])

    def test_context_overflow_never_silently_truncates_report(self):
        config = self.config()
        tokenizer = types.SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1] * config['context_tokens'])
        with tempfile.TemporaryDirectory() as temporary, patch('urllib.request.urlopen') as request:
            chat = LocalChat(config, tokenizer, Path(temporary))
            with self.assertRaises(BudgetExceeded):
                chat([{'role': 'user', 'content': 'long report'}])
            request.assert_not_called()

    def test_backend_and_report_fields_are_explicit(self):
        config = self.config()
        base = json.loads((PROJECT / config['base_config']).read_text())
        validate_config(config, base)
        with self.assertRaisesRegex(ValueError, 'local model endpoint'):
            validate_config({**config, 'endpoint': 'https://api.example.com/v1'}, base)
        with self.assertRaisesRegex(ValueError, 'solution PR'):
            validate_config(config, {**base, 'query_fields': ['title']})


if __name__ == '__main__':
    unittest.main()
