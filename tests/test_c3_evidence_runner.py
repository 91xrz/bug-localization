"""Exercise retries, resumability and full-denominator scoring without a model."""
import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'c3_evidence_runner', ROOT / 'scripts/experiments/pilot_c3_evidence.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def saved_result():
    return {
        'ranking': [{'file_path': p} for p in ['b.py', 'a.py', 'tail.py']],
        'scored_chunks': [
            {'file_path': p, 'chunk_id': p, 'embedding_text': 'value = compute(x)'}
            for p in ['a.py', 'b.py', 'tail.py']],
        'ground_truth': ['a.py'],
    }


class ScriptedGenerator:
    def __init__(self, replies):
        self.replies = replies
        self.calls = 0
        self.requests = []

    def count_tokens(self, messages):
        return 100

    def complete(self, messages, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        value = self.replies[self.calls]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)


@contextmanager
def interrupted_call_fixture(call_record, *, source_failed=False):
    """Build a complete fake run with a call log but no terminal case record."""
    class CheckpointServer:
        def __init__(self, project, config):
            self.identity = {'backend': 'fake-checkpoint-test'}
            self.server_properties = {'backend': 'fake-checkpoint-test'}
            self.calls = 0

        def start_case(self, folder):
            self.folder = folder
            self.calls = 0

        def count_tokens(self, messages):
            return 100

        def complete(self, messages, **kwargs):
            self.calls += 1
            checkpoint = self.folder / f'call_{self.calls:03d}.json'
            if checkpoint.exists() and runner.read(checkpoint)['status'] != 'ok':
                # This matches ServerGenerator's cached-non-ok path. The runner
                # must inspect the preserved cause before this generic ValueError
                # could turn an infrastructure outage into a localization miss.
                raise ValueError('Previous inference call failed; explicit new run required')
            return json.dumps({'reasoning': 'Mechanism comparison.',
                               'ranked_ids': ['F01', 'F02']})

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        c3, reports, output = (root / name for name in ['c3', 'reports', 'output'])
        identity = dict(position=1, dataset_row=0, repository='example/repo', issue_id=1,
                        before_fix_sha='abc', query_sha256='query', labels_sha256='labels')
        saved = {**saved_result(), **identity}
        saved['ranking'] = saved['ranking'][:2]
        saved['scored_chunks'] = saved['scored_chunks'][:2]
        result_path = c3 / 'cases' / '001.json'
        result_path.parent.mkdir(parents=True)
        (reports / 'cases' / '0001').mkdir(parents=True)
        runner.write_json(result_path, saved)
        runner.write_json(c3 / 'samples.json', [{
            **identity, 'status': 'failed' if source_failed else 'ok',
            'missing_gold': [], 'result_file': 'cases/001.json',
            'result_sha256': runner.sha256_file(result_path)}])
        runner.write_json(reports / 'cases' / '0001' / 'inputs.json',
                          {'query': 'The result is wrong.', 'source_commit': 'abc'})
        selection = root / 'selection.json'
        runner.write_json(selection, {'samples': [identity]})
        case = output / 'evidence' / 'cases' / '0001'
        case.mkdir(parents=True)
        if call_record is not None:
            runner.write_json(case / 'call_001.json', call_record)
        with patch.object(runner, 'C3', c3), patch.object(runner, 'REPORTS', reports), \
                patch.object(runner, 'ServerGenerator', CheckpointServer):
            yield selection, output, case


class RunnerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.payload = runner.build_input(saved_result(), 'The result is wrong.', head=2)
        self.good = {'reasoning': 'Implementation matches the report.',
                     'ranked_ids': ['F01', 'F02']}

    def test_retry_uses_validation_feedback_without_labels_or_previous_answer(self):
        bad = {'reasoning': 'SECRET_MODEL_TEXT', 'ranked_ids': ['F01', 'F01']}
        generator = ScriptedGenerator([bad, self.good])
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.infer(generator, self.payload, 'baseline', Path(tmp))
            self.assertEqual(result, self.good)
            self.assertEqual(generator.calls, 2)
            second = generator.requests[1][-1]['content']
            self.assertTrue(second.startswith(self.payload['user_content']))
            self.assertIn('OUTPUT VALIDATION FEEDBACK', second)
            self.assertNotIn('SECRET_MODEL_TEXT', second)
            self.assertNotIn('ground_truth', second)
            validation = runner.read(Path(tmp) / 'validation.json')
            self.assertTrue(validation[0]['errors'])
            self.assertEqual(validation[1]['errors'], [])

    def test_invalid_response_exhausts_two_call_budget_without_silent_fallback(self):
        bad = {'reasoning': 'Why', 'ranked_ids': ['F01', 'F01']}
        generator = ScriptedGenerator([bad, bad])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'two calls'):
                runner.infer(generator, self.payload, 'baseline', Path(tmp))
            self.assertEqual(generator.calls, 2)
            self.assertFalse((Path(tmp) / 'result.json').exists())

    def test_transport_error_propagates_without_an_unrecorded_retry(self):
        generator = ScriptedGenerator([ConnectionError('offline')])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConnectionError):
                runner.infer(generator, self.payload, 'baseline', Path(tmp))
            self.assertEqual(generator.calls, 1)

    def test_partial_cohort_hides_accuracy_and_failures_remain_in_denominator(self):
        metrics = runner.ranking_metrics(['a.py'], ['a.py'], 10)
        ok = {'status': 'ok', 'metrics': metrics, 'missing_gold': []}
        partial = runner.summary([ok], 2)
        self.assertEqual(partial['pending_samples'], 1)
        self.assertIsNone(partial['metrics'])
        final = runner.summary([ok, {'status': 'failed', 'missing_gold': [],
                                    'repository': 'r', 'issue_id': 1,
                                    'error_type': 'ValueError', 'error': 'invalid output'}], 2)
        self.assertEqual(final['failed_samples'], 1)
        self.assertEqual(final['metrics']['MAP@10'], 0.5)

    def test_frozen_artifact_cannot_be_replaced_with_new_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'protocol.json'
            runner.frozen_write(target, {'budget': 2048})
            runner.frozen_write(target, {'budget': 2048})
            with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                runner.frozen_write(target, {'budget': 4096})
            self.assertEqual(runner.read(target), {'budget': 2048})

    def test_resumed_infrastructure_failure_does_not_become_a_scored_failure(self):
        for error_type in ['URLError', 'TimeoutError', 'OSError', 'RuntimeError']:
            with self.subTest(error_type=error_type):
                original = {'status': 'failed', 'error_type': error_type,
                            'error': 'inference service interrupted', 'input_tokens': 100}
                with interrupted_call_fixture(original) as (selection, output, case):
                    try:
                        with self.assertRaises(RuntimeError):
                            runner.run(selection, output, head=2)
                    finally:
                        self.assertFalse((case / 'record.json').exists(),
                                         'Infrastructure interruption is not a zero-scored model failure')
                        self.assertEqual(runner.read(case / 'call_001.json'), original)

    def test_resumed_running_call_does_not_become_a_scored_failure(self):
        original = {'status': 'running', 'input_tokens': 100}
        with interrupted_call_fixture(original) as (selection, output, case):
            try:
                with self.assertRaises(RuntimeError):
                    runner.run(selection, output, head=2)
            finally:
                self.assertFalse((case / 'record.json').exists(),
                                 'Unfinished inference is not a zero-scored model failure')
                self.assertEqual(runner.read(case / 'call_001.json'), original)

    def test_resumed_output_budget_failure_stays_a_scored_model_failure(self):
        original = {'status': 'failed', 'error_type': 'ValueError',
                    'error': 'Inference truncated or exhausted the output budget',
                    'input_tokens': 100, 'output_tokens': 2048}
        with interrupted_call_fixture(original) as (selection, output, case):
            runner.run(selection, output, head=2)
            record = runner.read(case / 'record.json')
            self.assertEqual(record['status'], 'failed')
            summary = runner.read(output / 'evidence' / 'summary.json')
            self.assertEqual(summary['failed_samples'], 1)
            self.assertEqual(summary['metrics']['MAP@10'], 0.0)
            self.assertEqual(runner.read(case / 'call_001.json'), original)

    def test_failed_c3_source_remains_in_both_denominators_without_model_calls(self):
        with interrupted_call_fixture(None, source_failed=True) as (selection, output, case):
            runner.run(selection, output, head=2)
            for variant in ['baseline', 'evidence']:
                with self.subTest(variant=variant):
                    folder = output / variant
                    record = runner.read(folder / 'cases' / '0001' / 'record.json')
                    self.assertEqual(record['status'], 'failed')
                    self.assertEqual(record['model_calls'], 0)
                    self.assertEqual(record['source_sha256'], {})
                    self.assertFalse((folder / 'cases' / '0001' / 'inference_input.json').exists())
                    summary = runner.read(folder / 'summary.json')
                    self.assertEqual(summary['selected_samples'], 1)
                    self.assertEqual(summary['failed_samples'], 1)
                    self.assertEqual(summary['metrics']['MAP@10'], 0.0)

    def test_full_runner_resumes_without_calls_and_rejects_result_tampering(self):
        class FakeServer:
            total_calls = 0

            def __init__(self, project, config):
                self.identity = {'backend': 'fake-test'}
                self.server_properties = {'backend': 'fake-test'}
                self.calls = 0

            def start_case(self, folder):
                self.calls = 0

            def count_tokens(self, messages):
                return 100

            def complete(self, messages, **kwargs):
                self.calls += 1
                type(self).total_calls += 1
                value = {'reasoning': 'Mechanism comparison.', 'ranked_ids': ['F01', 'F02']}
                if kwargs['stage'] == 'C3_evidence':
                    value.update(
                        constraints=[{'id': 'R1', 'kind': 'observed', 'quote': 'wrong'}],
                        assessments=[{'file_id': f, 'state': 'unknown', 'constraint_ids': [],
                                      'evidence': [], 'reason': 'Insufficient context.'}
                                     for f in ['F01', 'F02']])
                return json.dumps(value)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c3 = root / 'c3'
            reports = root / 'reports'
            output = root / 'output'
            identity = dict(position=1, dataset_row=0, repository='example/repo', issue_id=1,
                            before_fix_sha='abc', query_sha256='query', labels_sha256='labels')
            saved = {**saved_result(), **identity}
            saved['ranking'] = saved['ranking'][:2]
            saved['scored_chunks'] = saved['scored_chunks'][:2]
            result_path = c3 / 'cases' / '001.json'
            result_path.parent.mkdir(parents=True)
            (reports / 'cases' / '0001').mkdir(parents=True)
            runner.write_json(result_path, saved)
            source = {**identity, 'status': 'ok', 'missing_gold': [],
                      'result_file': 'cases/001.json',
                      'result_sha256': runner.sha256_file(result_path)}
            runner.write_json(c3 / 'samples.json', [source])
            runner.write_json(reports / 'cases' / '0001' / 'inputs.json',
                              {'query': 'The result is wrong.', 'source_commit': 'abc'})
            selection = root / 'selection.json'
            runner.write_json(selection, {'samples': [identity]})
            with patch.object(runner, 'C3', c3), patch.object(runner, 'REPORTS', reports), \
                    patch.object(runner, 'ServerGenerator', FakeServer):
                runner.run(selection, output, head=2)
                self.assertEqual(FakeServer.total_calls, 2)
                runner.run(selection, output, head=2)
                self.assertEqual(FakeServer.total_calls, 2)
                for variant in ['baseline', 'evidence']:
                    records = runner.read(output / variant / 'samples.json')
                    self.assertEqual(len(records), 1)
                    self.assertEqual(records[0]['status'], 'ok')
                target = output / 'evidence' / 'cases' / '0001' / 'result.json'
                changed = runner.read(target)
                changed['ranking'].reverse()
                runner.write_json(target, changed)
                with self.assertRaisesRegex(ValueError, 'Checkpoint artifact checksum mismatch'):
                    runner.run(selection, output, head=2)
                self.assertEqual(FakeServer.total_calls, 2)


if __name__ == '__main__':
    unittest.main()
