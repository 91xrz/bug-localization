import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock
from elasticsearch import BadRequestError
from elastic_transport import ApiResponseMeta, NodeConfig
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bug_localization.bm25 import analyze_query


def overflow():
    return BadRequestError('index.analyze.max_token_count exceeded',
        meta=ApiResponseMeta(400, '1.1', {}, 0, NodeConfig('http', 'localhost', 9200)), body={})


class AnalyzeLimitTest(unittest.TestCase):
    def test_retry_preserves_full_text_tokens_and_restores_setting(self):
        client = MagicMock()
        result = {'tokens': [{'token': 'x'}, {'token': 'x'}, {'token': 'y'}]}
        client.indices.analyze.side_effect = [overflow(), result]
        client.indices.get_settings.return_value = {'idx': {'settings': {}}}
        text = 'x y ' * 10000
        self.assertEqual(analyze_query(client, 'idx', text), result)
        self.assertEqual(client.indices.analyze.call_args_list[0], client.indices.analyze.call_args_list[1])
        self.assertEqual(client.indices.put_settings.call_args.kwargs['settings'], {'index.analyze.max_token_count': None})

    def test_retry_failure_still_restores_existing_limit(self):
        client = MagicMock()
        client.indices.analyze.side_effect = [overflow(), RuntimeError('offline')]
        client.indices.get_settings.return_value = {'idx': {'settings': {'index.analyze.max_token_count': '12000'}}}
        with self.assertRaisesRegex(RuntimeError, 'offline'):
            analyze_query(client, 'idx', 'long report')
        self.assertEqual(client.indices.put_settings.call_args.kwargs['settings'], {'index.analyze.max_token_count': '12000'})

    def test_short_query_does_not_change_settings(self):
        client = MagicMock()
        analyze_query(client, 'idx', 'short')
        client.indices.put_settings.assert_not_called()
