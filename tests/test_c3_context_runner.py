"""Preservation, fair output protocol and recovery checks for context additions."""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'context_runner_test', ROOT / 'scripts/experiments/pilot_c3_context.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ContextRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sample = dict(position=1, dataset_row=0, repository='example/project',
                           issue_id=3, before_fix_sha='a' * 40,
                           query_sha256='frozen-query', labels_sha256='frozen-labels')
        self.query = 'Calling `dangerous` raises ValueError on this input.'
        self.paths = ['z.py', 'a.py'] + [f'm{i}.py' for i in range(9)]
        self.code = ('def compute(raw):\n'
                     '    result = raw.strip()\n'
                     '    ready = bool(result)\n'
                     '    if ready:\n'
                     '        output = dangerous(result)\n'
                     '        note = 1\n'
                     '        raise ValueError(output)\n')
        self.old_texts = ['    result = raw.strip()\n', '    ready = bool(result)\n',
                         '        raise ValueError(output)\n']
        self.saved = {
            **self.sample,
            'ranking': [{'file_path': p} for p in self.paths],
            'scored_chunks': [
                {'file_path': p, 'chunk_id': f'{p}-chunk-{i}', 'embedding_text': text,
                 'score': 'SECRET_PRIOR_SCORE'}
                for p in self.paths for i, text in enumerate(self.old_texts)],
            'ground_truth': ['SECRET_GOLD.py'], 'metrics': {'SECRET_METRIC': True},
        }
        self.base = self.root / 'repos' / self.sample['repository'] / self.sample['before_fix_sha']
        source = self.base / 'source'
        source.mkdir(parents=True)
        checksums = {}
        for path in self.paths:
            (source / path).write_text(self.code)
            checksums[path] = hashlib.sha256(self.code.encode()).hexdigest()
        archive = self.base / 'snapshot.zip'
        with zipfile.ZipFile(archive, 'w') as handle:
            for path in self.paths:
                handle.writestr('project-' + self.sample['before_fix_sha'] + '/' + path, self.code)
        self.manifest = {'repository': self.sample['repository'],
                         'commit': self.sample['before_fix_sha'],
                         'source_url': 'https://codeload.github.com/example/project/zip/' + self.sample['before_fix_sha'],
                         'archive_sha256': runner.sha256_file(archive),
                         'archive_bytes': archive.stat().st_size, 'files': checksums}
        runner.write_json(self.base / 'snapshot.json', self.manifest)

    def payloads(self):
        with patch.object(runner, 'PROJECT', self.root):
            return {v: runner.build_context_input(self.saved, self.query, self.sample, v,
                                                  lambda text: len(text.split()))
                    for v in runner.VARIANTS}

    def test_every_arm_preserves_report_candidates_tail_and_all_original_chunks(self):
        before = copy.deepcopy(self.saved)
        payloads = self.payloads()
        base = payloads['baseline']
        for variant, value in payloads.items():
            with self.subTest(variant=variant):
                self.assertEqual(value['query'], self.query)
                self.assertEqual(value['candidates'], self.paths[:10])
                self.assertEqual(value['tail'], self.paths[10:])
                self.assertEqual(value['evidence'], base['evidence'])
                self.assertEqual(value['catalog'], base['catalog'])
                self.assertTrue(value['user_content'].startswith(base['user_content']))
                self.assertNotIn('SECRET_', value['user_content'])
                for path in self.paths[:10]:
                    self.assertEqual([c['text'] for c in value['evidence'][path]], self.old_texts)
        self.assertEqual(self.saved, before)

    def test_baseline_adds_no_context_and_other_arms_use_only_same_candidate_files(self):
        payloads = self.payloads()
        self.assertEqual(payloads['baseline']['source_context'], {})
        self.assertEqual(payloads['baseline']['context_token_count'], 0)
        for variant in ('lexical', 'structural'):
            value = payloads[variant]
            self.assertEqual(set(value['source_context']), set(self.paths[:10]))
            self.assertNotIn(self.paths[-1], value['source_context'])
            for path, context in value['source_context'].items():
                self.assertEqual(context['source_sha256'], self.manifest['files'][path])
                self.assertLessEqual(context['token_count'], 256)

    def test_wrong_commit_and_changed_source_are_rejected_before_inference(self):
        changed = dict(self.manifest, commit='b' * 40)
        runner.write_json(self.base / 'snapshot.json', changed)
        with self.assertRaisesRegex(ValueError, 'pre-fix|revision|commit|snapshot'):
            self.payloads()
        runner.write_json(self.base / 'snapshot.json', self.manifest)
        (self.base / 'source' / self.paths[0]).write_text('SECRET_CHANGED_CODE\n')
        with self.assertRaisesRegex(ValueError, 'checksum|source|archive'):
            self.payloads()

    def test_archive_tampering_wrong_commit_root_and_changed_candidate_bytes_are_rejected(self):
        archive = self.base / 'snapshot.zip'
        original = archive.read_bytes()
        with patch.object(runner, 'PROJECT', self.root):
            runner.verify_source_archive(self.sample, self.saved)
            archive.write_bytes(original + b'altered archive bytes')
            with self.assertRaisesRegex(ValueError, 'archive checksum'):
                runner.verify_source_archive(self.sample, self.saved)
            for wrong_root in (True, False):
                with self.subTest(wrong_root=wrong_root):
                    commit = 'b' * 40 if wrong_root else self.sample['before_fix_sha']
                    with zipfile.ZipFile(archive, 'w') as handle:
                        for path in self.paths:
                            text = self.code
                            if not wrong_root and path == self.paths[0]:
                                text += 'ALTERED_CANDIDATE = True\n'
                            handle.writestr('project-' + commit + '/' + path, text)
                    # Even a matching archive hash cannot mask a wrong commit
                    # root or candidate bytes inconsistent with the inventory.
                    manifest = dict(self.manifest, archive_sha256=runner.sha256_file(archive))
                    runner.write_json(self.base / 'snapshot.json', manifest)
                    message = 'root does not match' if wrong_root else 'differs from pinned archive'
                    with self.assertRaisesRegex(ValueError, message):
                        runner.verify_source_archive(self.sample, self.saved)

    def test_all_arms_use_same_ordinary_prompt_and_complete_candidate_permutation(self):
        payloads = self.payloads()
        for variant, value in payloads.items():
            messages = runner.build_messages(value, 'baseline')
            self.assertEqual(messages[0]['content'], runner.BASELINE_PROMPT)
            response = {'reasoning': 'Code comparison.',
                        'ranked_ids': list(reversed(value['catalog']['files']))}
            ranking = runner.resolve_ranking(response, value)
            self.assertEqual(set(ranking[:10]), set(self.paths[:10]))
            self.assertEqual(ranking[10:], self.paths[10:])

    def test_other_original_else_block_does_not_hide_the_fault_branch_guard(self):
        code = ('def unrelated(flag):\n'
                '    if flag:\n'
                '        return 1\n'
                '    else:\n'
                '        return 2\n'
                'def process(ready):\n'
                '    if ready:\n'
                '        return 3\n'
                '    else:\n'
                '        detail = "bad"\n'
                '        raise ValueError(detail)\n')
        original = ['    else:\n        return 2\n',
                    '        raise ValueError(detail)\n',
                    '        detail = "bad"\n']
        path = 'a.py'
        (self.base / 'source' / path).write_text(code)
        self.manifest['files'][path] = hashlib.sha256(code.encode()).hexdigest()
        runner.write_json(self.base / 'snapshot.json', self.manifest)
        saved = {**self.saved, 'ranking': [{'file_path': path}],
                 'scored_chunks': [{'file_path': path, 'chunk_id': str(i), 'embedding_text': t}
                                   for i, t in enumerate(original)]}
        with patch.object(runner, 'PROJECT', self.root):
            payload = runner.build_context_input(saved, 'A `ValueError` occurs.', self.sample,
                                                  'structural', lambda t: len(t.split()))
        self.assertEqual([c['text'] for c in payload['evidence'][path]], original)
        context = payload['source_context'][path]
        selected = {n for s in context['segments'] for n in range(s['start_line'], s['end_line'] + 1)}
        self.assertTrue({7, 9} <= selected)
        self.assertNotIn(4, selected)
        self.assertIn('if ready:', context['rendered_text'])
        self.assertIn('else:', context['rendered_text'])

    def run_mock(self, output, *, source=None, fail=None):
        payloads = self.payloads()
        source_file = self.root / 'source_result.json'
        if not source_file.exists():
            runner.write_json(source_file, self.saved)
        selection = self.root / 'selection.json'
        runner.write_json(selection, {'samples': [self.sample]})
        source = source or {**self.sample, 'status': 'ok', 'missing_gold': ['SECRET_GOLD.py']}
        prepared = {1: (source, self.saved, {str(source_file): runner.sha256_file(source_file)}, payloads)}

        class Tokenizer:
            def encode(self, text, **kwargs):
                return text.split()

        class Server:
            calls_total = 0
            stages = []
            schemas = []
            users = []

            def __init__(self, project, settings):
                self.identity = {'test': True, 'settings': settings}
                self.server_properties = {'test': True}

            def count_tokens(self, messages):
                return 100

            def start_case(self, folder):
                self.calls = 0
                self.folder = folder

            def complete(self, messages, **kwargs):
                self.calls += 1
                type(self).calls_total += 1
                type(self).stages.append(kwargs['stage'])
                type(self).schemas.append(copy.deepcopy(kwargs['response_schema']))
                type(self).users.append(messages[1]['content'])
                if fail is not None:
                    raise fail
                ids = kwargs['response_schema']['properties']['ranked_ids']['items']['enum']
                return json.dumps({'reasoning': 'Mechanism comparison.', 'ranked_ids': ids})

        patches = (patch.object(runner, 'prepare', return_value=({'samples': [self.sample]}, prepared)),
                   patch.object(runner, 'ServerGenerator', Server),
                   patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()))
        return selection, Server, patches

    def test_runner_freezes_protocol_resumes_without_new_calls_and_rejects_tampering(self):
        output = self.root / 'run'
        selection, server, patches = self.run_mock(output)
        with patches[0], patches[1], patches[2]:
            runner.run(selection, output)
            self.assertEqual(server.calls_total, 3)
            self.assertEqual(set(server.stages), {'C3_baseline'})
            self.assertTrue(all(schema == server.schemas[0] for schema in server.schemas))
            self.assertEqual(set(server.schemas[0]['properties']), {'reasoning', 'ranked_ids'})
            runner.run(selection, output)
            self.assertEqual(server.calls_total, 3)
            protocols = [runner.read(output / v / 'protocol.json') for v in runner.VARIANTS]
            self.assertTrue(all(p['generator_identity'] == protocols[0]['generator_identity'] for p in protocols))
            self.assertTrue(all(p['prompt'] == runner.BASELINE_PROMPT for p in protocols))
            target = output / 'structural' / 'cases' / '0001' / 'result.json'
            result = runner.read(target)
            result['ranking'].reverse()
            runner.write_json(target, result)
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                runner.run(selection, output)
            self.assertEqual(server.calls_total, 3)

    def test_interrupted_transport_call_cannot_turn_into_algorithm_failure_on_resume(self):
        output = self.root / 'run'
        case = output / 'structural' / 'cases' / '0001'
        case.mkdir(parents=True)
        runner.write_json(case / 'call_001.json',
                          {'status': 'failed', 'error_type': 'TimeoutError', 'error': 'offline'})
        selection, server, patches = self.run_mock(output)
        with patches[0], patches[1], patches[2]:
            with self.assertRaisesRegex(RuntimeError, 'infrastructure call'):
                runner.run(selection, output)
        self.assertFalse((case / 'record.json').exists())

    def test_lock_prevents_two_writers_and_is_released_on_preparation_error(self):
        output = self.root / 'run'
        output.mkdir()
        with (output / '.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(runner, 'run_locked') as inner:
                with self.assertRaises(BlockingIOError):
                    runner.run(self.root / 'unused.json', output)
                inner.assert_not_called()
        with patch.object(runner, 'run_locked', side_effect=ValueError('preparation failed')):
            with self.assertRaisesRegex(ValueError, 'preparation failed'):
                runner.run(self.root / 'unused.json', output)
        with (output / '.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def cache_fixture(self):
        """Build a pinned preparation cache without loading a tokenizer/model."""
        for relative in ('src/bug_localization/context_evidence.py',
                         'src/bug_localization/evidence_reranking.py'):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
        model = self.root / 'models/Qwen3-8B'
        model.mkdir(parents=True)
        for name in ('tokenizer.json', 'tokenizer_config.json'):
            (model / name).write_text('{"fixture":true}\n')
        c3, reports = self.root / 'c3', self.root / 'reports'
        result_path = c3 / 'cases/001.json'
        result_path.parent.mkdir(parents=True)
        runner.write_json(result_path, self.saved)
        source = {**self.sample, 'status': 'ok', 'missing_gold': ['SECRET_GOLD.py'],
                  'result_file': 'cases/001.json', 'result_sha256': runner.sha256_file(result_path)}
        runner.write_json(c3 / 'samples.json', [source])
        report_path = reports / 'cases/0001/inputs.json'
        report_path.parent.mkdir(parents=True)
        runner.write_json(report_path, {'query': self.query, 'source_commit': self.sample['before_fix_sha']})
        selection = self.root / 'cache_selection.json'
        runner.write_json(selection, {'samples': [self.sample]})
        output = self.root / 'cache_output'
        (output / 'prepared_inputs').mkdir(parents=True)
        payloads = self.payloads()
        runner.write_json(output / 'prepared_inputs/0001.json', payloads)
        with patch.object(runner, 'PROJECT', self.root):
            runner.freeze_prepared_cache(selection, output)
        return selection, output, c3, reports, payloads

    def test_prepared_cache_loads_without_selector_calls_and_rejects_sha_tampering(self):
        selection, output, c3, reports, payloads = self.cache_fixture()
        with patch.object(runner, 'PROJECT', self.root), \
                patch.object(runner.BASE, 'C3', c3), patch.object(runner.BASE, 'REPORTS', reports), \
                patch.object(runner, 'select_context', side_effect=AssertionError('must use prepared slices')):
            manifest, prepared = runner.load_prepared_cache(selection, output, lambda t: len(t.split()))
            self.assertEqual(manifest['samples'], [self.sample])
            self.assertEqual(prepared[1][3], payloads)
            target = output / 'prepared_inputs/0001.json'
            changed = runner.read(target)
            changed['lexical']['user_content'] += '\nSECRET_EXTRA_TEXT'
            runner.write_json(target, changed)
            with self.assertRaisesRegex(ValueError, 'cache file checksum'):
                runner.load_prepared_cache(selection, output, lambda t: len(t.split()))

    def test_refrozen_self_consistent_cache_cannot_invent_source_text(self):
        selection, output, c3, reports, payloads = self.cache_fixture()
        payload = payloads['structural']
        context = next(c for c in payload['source_context'].values() if c['segments'])
        segment = context['segments'][0]
        segment['text'] += '# INVENTED_NOT_IN_SOURCE\n'
        segment['text_sha256'] = hashlib.sha256(segment['text'].encode()).hexdigest()
        context['rendered_text'] = runner.render_segments(context['file_path'], context['segments'])
        context['token_count'] = len(context['rendered_text'].split())
        payload['context_token_count'] = sum(c['token_count'] for c in payload['source_context'].values())
        sections = [payload['source_context'][p]['rendered_text'] for p in sorted(payload['source_context'])
                    if payload['source_context'][p]['rendered_text']]
        payload['user_content'] = payloads['baseline']['user_content'] + '\n\nADDITIONAL SOURCE CONTEXT:\n' + '\n\n'.join(sections)
        payload['user_content_sha256'] = runner.digest(payload['user_content'])
        runner.write_json(output / 'prepared_inputs/0001.json', payloads)
        # Simulate a newly frozen cache: source verification must still reject
        # invented text even if every cache-level checksum has been recomputed.
        (output / 'prepared_cache.json').unlink()
        with patch.object(runner, 'PROJECT', self.root), \
                patch.object(runner.BASE, 'C3', c3), patch.object(runner.BASE, 'REPORTS', reports):
            runner.freeze_prepared_cache(selection, output)
            with self.assertRaisesRegex(ValueError, 'not an exact pre-fix source slice'):
                runner.load_prepared_cache(selection, output, lambda t: len(t.split()))

    def test_prepared_cache_rejects_changed_budget_head_or_tokenizer_identity(self):
        selection, output, c3, reports, _ = self.cache_fixture()
        with patch.object(runner, 'PROJECT', self.root), \
                patch.object(runner.BASE, 'C3', c3), patch.object(runner.BASE, 'REPORTS', reports):
            for name, value in [('BUDGET', 128), ('HEAD', 9)]:
                with self.subTest(setting=name), patch.object(runner, name, value):
                    with self.assertRaisesRegex(ValueError, 'implementation/selection/tokenizer changed'):
                        runner.load_prepared_cache(selection, output, lambda t: len(t.split()))
            (self.root / 'models/Qwen3-8B/tokenizer.json').write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, 'implementation/selection/tokenizer changed'):
                runner.load_prepared_cache(selection, output, lambda t: len(t.split()))


if __name__ == '__main__':
    unittest.main()
