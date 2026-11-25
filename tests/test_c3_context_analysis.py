import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('c3_context_analysis', PROJECT / 'scripts/analysis/analyze_c3_context_pilot.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
from bug_localization.context_evidence import select_context
from bug_localization.evidence_reranking import BASELINE_PROMPT, build_input, response_schema


class ContextAnalysisTests(unittest.TestCase):
    def fixture(self, root, position=1):
        sample = {'position': position, 'dataset_row': position - 1, 'repository': f'owner/repo{position}',
                  'issue_id': position, 'before_fix_sha': 'a' * 40, 'query_sha256': f'q{position}', 'labels_sha256': f'g{position}'}
        source = root / 'repos' / sample['repository'] / sample['before_fix_sha'] / 'source'
        source.mkdir(parents=True)
        code = b'def compute(X):\n    result = np.empty(4, dtype=X.dtype)\n    result[:] = X / 2\n    return result\n'
        for name, raw in [('a.py', code), ('b.py', code.replace(b'\n', b'\r\n')), ('c.py', b'other = 1\n')]:
            (source / name).write_bytes(raw)
        archive = source.parent / 'snapshot.zip'
        archive_root = sample['repository'].rsplit('/', 1)[-1] + '-' + sample['before_fix_sha']
        with zipfile.ZipFile(archive, 'w') as handle:
            for name in ('a.py', 'b.py', 'c.py'):
                handle.writestr(archive_root + '/' + name, (source / name).read_bytes())
        manifest_path = source.parent / 'snapshot.json'
        analysis.write_json(manifest_path, {'repository': sample['repository'], 'commit': sample['before_fix_sha'],
            'files': {name: analysis.sha256_file(source / name) for name in ('a.py', 'b.py', 'c.py')},
            'archive_sha256': analysis.sha256_file(archive)})
        inventory_path = source.parent / 'candidates.json'
        analysis.write_json(inventory_path, {'files': [{'file_path': p} for p in ('a.py', 'b.py', 'c.py')]})
        saved = {**sample, 'ranking': [{'file_path': p} for p in ('a.py', 'b.py', 'c.py')],
                 'ground_truth': ['a.py'], 'source_files_sha256': {str(inventory_path): analysis.sha256_file(inventory_path)},
                 'scored_chunks': [{'file_path': p, 'chunk_id': p, 'embedding_text': 'Compute values.'} for p in ('a.py', 'b.py')]}
        report = {'query': 'Integer input dtype loses fractional result', 'source_commit': sample['before_fix_sha']}
        snapshot = {'root': str(source), 'manifest_file': str(manifest_path),
                    'manifest_sha256': analysis.sha256_file(manifest_path), 'repository': sample['repository'],
                    'commit': sample['before_fix_sha'], 'archive_file': str(archive),
                    'archive_sha256': analysis.sha256_file(archive), 'archive_root': archive_root}
        return sample, saved, report, snapshot

    def payload(self, saved, report, snapshot, variant):
        base = build_input(saved, report['query'], 2)
        source = Path(snapshot['root'])
        manifest = analysis.read(Path(snapshot['manifest_file']))
        context = {} if variant == 'baseline' else {
            p: select_context(source, p, report['query'], [c['text'] for c in base['evidence'][p]], len,
                              variant=variant, budget_tokens=256, expected_sha256=manifest['files'][p])
            for p in base['candidates']}
        user = base['user_content']
        if variant != 'baseline':
            sections = [
                context[p]['rendered_text']
                for p in sorted(context) if context[p]['rendered_text']]
            if sections:
                user += '\n\nADDITIONAL SOURCE CONTEXT:\n' + '\n\n'.join(sections)
        return {**base, 'base_user_content_sha256': base['user_content_sha256'],
                'source_snapshot': snapshot, 'source_context': context,
                'context_budget_per_file': 256, 'context_token_count': sum(v['token_count'] for v in context.values()),
                'user_content': user, 'user_content_sha256': analysis.digest(user)}

    def test_all_variants_preserve_original_code_and_rebuild_exact_source_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, saved, report, snapshot = self.fixture(root)
            for variant in analysis.VARIANTS:
                payload = self.payload(saved, report, snapshot, variant)
                result = analysis.audit_input(payload, saved, report, sample, variant, len, project=root)
                self.assertEqual(result['before_fix_sha'], sample['before_fix_sha'])
                self.assertEqual(payload['evidence'], build_input(saved, report['query'], 2)['evidence'])
                if variant != 'baseline':
                    self.assertTrue(any('\r\n' in s['text'] for s in payload['source_context']['b.py']['segments']))
                    self.assertGreater(result['context_tokens'], 0)

    def test_fabricated_source_text_is_rejected_even_with_self_consistent_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, saved, report, snapshot = self.fixture(root)
            payload = self.payload(saved, report, snapshot, 'lexical')
            segment = payload['source_context']['a.py']['segments'][0]
            segment['text'] = 'invented repair code\n'
            segment['text_sha256'] = hashlib.sha256(segment['text'].encode()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'pre-fix file lines'):
                analysis.audit_source_context(payload, sample, 'lexical', len, project=root)

    def test_empty_additions_leave_the_actual_user_content_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, saved, report, snapshot = self.fixture(root)
            report['query'] = 'UnmatchedIdentifierZZZ'
            payload = self.payload(saved, report, snapshot, 'structural')
            self.assertEqual(payload['context_token_count'], 0)
            self.assertEqual(payload['base_user_content_sha256'], payload['user_content_sha256'])
            analysis.audit_input(payload, saved, report, sample, 'structural', len, project=root)

    def test_report_and_original_code_cannot_be_replaced_by_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, saved, report, snapshot = self.fixture(root)
            payload = self.payload(saved, report, snapshot, 'structural')
            payload['evidence']['a.py'][0]['text'] = 'changed original block'
            with self.assertRaisesRegex(ValueError, 'raw C3 evidence'):
                analysis.audit_input(payload, saved, report, sample, 'structural', len, project=root)

    def test_source_commit_and_manifest_file_hash_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, saved, report, snapshot = self.fixture(root)
            payload = self.payload(saved, report, snapshot, 'baseline')
            bad = copy.deepcopy(payload)
            bad['source_snapshot']['commit'] = 'b' * 40
            with self.assertRaisesRegex(ValueError, 'before-fix commit'):
                analysis.audit_source_context(bad, sample, 'baseline', len, project=root)
            (Path(snapshot['root']) / 'a.py').write_text('altered = True\n')
            with self.assertRaisesRegex(ValueError, 'file hash'):
                analysis.audit_source_context(payload, sample, 'baseline', len, project=root)

    def test_added_context_budget_and_selection_provenance_are_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, saved, report, snapshot = self.fixture(root)
            payload = self.payload(saved, report, snapshot, 'structural')
            bad = copy.deepcopy(payload)
            bad['source_context']['a.py']['token_count'] += 1
            with self.assertRaisesRegex(ValueError, 'token budget'):
                analysis.audit_source_context(bad, sample, 'structural', len, project=root)
            bad = copy.deepcopy(payload)
            bad['source_context']['a.py']['segments'][0]['reason'] = ['made_up_structure']
            with self.assertRaisesRegex(ValueError, 'deterministic frozen selection'):
                analysis.audit_source_context(bad, sample, 'structural', len, project=root)

    def test_source_archive_is_read_and_not_only_trusted_by_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, saved, report, snapshot = self.fixture(root)
            payload = self.payload(saved, report, snapshot, 'baseline')
            # Keep the manifest and extracted sources self-consistent while
            # deliberately giving the ZIP different original bytes.
            archive = Path(snapshot['archive_file'])
            with zipfile.ZipFile(archive, 'w') as handle:
                for name in ('a.py', 'b.py', 'c.py'):
                    data = b'altered = True\n' if name == 'a.py' else (Path(snapshot['root']) / name).read_bytes()
                    handle.writestr(snapshot['archive_root'] + '/' + name, data)
            manifest_path = Path(snapshot['manifest_file'])
            manifest = analysis.read(manifest_path)
            manifest['archive_sha256'] = analysis.sha256_file(archive)
            analysis.write_json(manifest_path, manifest)
            payload['source_snapshot'].update(archive_sha256=manifest['archive_sha256'],
                                             manifest_sha256=analysis.sha256_file(manifest_path))
            with self.assertRaisesRegex(ValueError, 'ZIP member bytes'):
                analysis.audit_source_context(payload, sample, 'baseline', len, project=root)

    def test_source_archive_root_must_identify_the_before_fix_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, saved, report, snapshot = self.fixture(root)
            payload = self.payload(saved, report, snapshot, 'baseline')
            archive = Path(snapshot['archive_file'])
            with zipfile.ZipFile(archive, 'w') as handle:
                handle.writestr('repo-WRONG/a.py', b'x = 1\n')
            manifest_path = Path(snapshot['manifest_file'])
            manifest = analysis.read(manifest_path)
            manifest['archive_sha256'] = analysis.sha256_file(archive)
            analysis.write_json(manifest_path, manifest)
            payload['source_snapshot'].update(archive_sha256=manifest['archive_sha256'],
                                             manifest_sha256=analysis.sha256_file(manifest_path))
            with self.assertRaisesRegex(ValueError, 'ZIP root'):
                analysis.audit_source_context(payload, sample, 'baseline', len, project=root)

    def test_prepared_cache_pins_renderer_tokenizer_and_every_input_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'baseline'
            run.mkdir()
            (root / 'implementation').mkdir()
            (root / 'prepared_inputs').mkdir()
            renderer = 'def build_context_input():\n    return {}\n'
            (root / 'implementation/pilot_c3_context.py').write_text(renderer)
            payloads = {v: {'variant': v} for v in analysis.VARIANTS}
            inputs_path = root / 'prepared_inputs/0001.json'
            analysis.write_json(inputs_path, payloads)
            protocol = {'selection_sha256': 'selection', 'head': 10, 'context_budget_per_file': 256,
                        'code_sha256': {'src/bug_localization/context_evidence.py': 'selector',
                                       'src/bug_localization/evidence_reranking.py': 'builder'},
                        'generator_identity': {'source_files_sha256': {'tokenizer.json': 'tokenizer'}},
                        'prepared_input_sha256': {'1': {v: analysis.digest(x) for v, x in payloads.items()}}}
            cache = {'selection_sha256': 'selection', 'head': 10, 'budget': 256,
                     'selector_sha256': 'selector', 'original_builder_sha256': 'builder',
                     'renderer_sha256': analysis.digest(renderer), 'tokenizer_sha256': {'tokenizer.json': 'tokenizer'},
                     'token_counting': 'AutoTokenizer.encode; add_special_tokens=False',
                     'files': {'0001.json': analysis.sha256_file(inputs_path)}}
            cache_path = root / 'prepared_cache.json'
            analysis.write_json(cache_path, cache)
            protocol['prepared_cache_sha256'] = analysis.sha256_file(cache_path)
            selection = {'samples': [{'position': 1}]}
            self.assertEqual(analysis.check_prepared_cache(run, protocol, selection)['files'], 1)
            payloads['lexical']['variant'] = 'altered'
            analysis.write_json(inputs_path, payloads)
            with self.assertRaisesRegex(ValueError, 'input cache file checksum'):
                analysis.check_prepared_cache(run, protocol, selection)

    def test_oracle_cannot_import_rank_eleven_and_missing_gold_stays_in_denominator(self):
        original = [f'f{i}.py' for i in range(1, 16)]
        gold = ['f8.py', 'f11.py', 'absent.py']
        ranking = analysis.oracle_head_ranking(original, gold, 10)
        self.assertEqual(ranking[0], 'f8.py')
        self.assertEqual(ranking[10:], original[10:])
        self.assertNotIn('f11.py', ranking[:10])
        self.assertEqual([p for p in ranking[:10] if p != 'f8.py'], [p for p in original[:10] if p != 'f8.py'])
        metrics = analysis.ranking_metrics(ranking, gold, 10)
        self.assertEqual(metrics['average_precision_at_k'], 1 / 3)
        self.assertEqual(metrics['recall_at_k'], 1 / 3)
        self.assertEqual(metrics['top_1'], 1)
        self.assertEqual(original[0], 'f1.py')

    def test_oracle_with_no_gold_in_head_cannot_improve_head_recall(self):
        original = [f'f{i}.py' for i in range(1, 13)]
        ranking = analysis.oracle_head_ranking(original, ['f11.py'], 10)
        self.assertEqual(ranking, original)
        self.assertEqual(analysis.ranking_metrics(ranking, ['f11.py'], 10)['top_1'], 0)

    def test_full_three_way_audit_keeps_failures_and_suppresses_pending_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = [self.fixture(root, p) for p in (1, 2)]
            selection = {'cohort': 'development', 'samples': [x[0] for x in fixtures]}
            selection_path = root / 'selection.json'
            analysis.write_json(selection_path, selection)
            identity = {'settings': {'max_calls': 2, 'max_new_tokens': 2048}}
            for variant in analysis.VARIANTS:
                run = root / variant
                run.mkdir()
                analysis.write_json(run / 'selection.json', selection)
                analysis.write_json(run / 'protocol.json', {'method': variant, 'inference_variant': 'baseline',
                    'head': 2, 'prompt': BASELINE_PROMPT, 'generator_identity': identity, 'code_sha256': {},
                    'selection_sha256': analysis.sha256_file(selection_path)})
                for sample, saved, report, snapshot in fixtures:
                    case = run / 'cases' / f'{sample["position"]:04d}'
                    case.mkdir(parents=True)
                    source_path = root / f'source{sample["position"]}.json'
                    report_path = root / f'report{sample["position"]}.json'
                    analysis.write_json(source_path, saved)
                    analysis.write_json(report_path, report)
                    payload = self.payload(saved, report, snapshot, variant)
                    analysis.write_json(case / 'inference_input.json', payload)
                    request = {'messages': [{'role': 'system', 'content': BASELINE_PROMPT},
                                            {'role': 'user', 'content': payload['user_content']}],
                               'identity': analysis.digest(identity), 'temperature': 0, 'max_new_tokens': 2048,
                               'stage': 'C3_baseline', 'response_schema': response_schema(payload, 'baseline')}
                    failed = variant == 'structural' and sample['position'] == 2
                    record = {**sample, 'source_sha256': {str(p): analysis.sha256_file(p) for p in (source_path, report_path)},
                              'model_calls': 1, 'missing_gold': [], 'elapsed_seconds': 1.,
                              'inference_input_file': str((case / 'inference_input.json').relative_to(run)),
                              'inference_input_sha256': analysis.sha256_file(case / 'inference_input.json')}
                    call = {'request': request, 'request_sha256': analysis.digest(request),
                            'input_tokens': 10, 'output_tokens': 5}
                    if failed:
                        record.update(status='failed', error_type='ValueError', error='Inference truncated or exhausted the output budget')
                        call.update(status='failed', error=record['error'], raw_response='{"reasoning":')
                    else:
                        ids = ['F02', 'F01'] if variant == 'baseline' else ['F01', 'F02']
                        response = {'reasoning': 'mechanism', 'ranked_ids': ids}
                        ranking = [payload['catalog']['files'][i] for i in ids] + payload['tail']
                        metrics = analysis.ranking_metrics(ranking, ['a.py'], 10)
                        call.update(status='ok', finished=True, response=json.dumps(response))
                        analysis.write_json(case / 'validation.json', [{'attempt': 1, 'errors': []}])
                        analysis.write_json(case / 'result.json', {'ranking': ranking, 'ground_truth': ['a.py'],
                            'metrics': metrics, 'model_calls': 1, 'structured_output': response, 'reasoning': 'mechanism'})
                        record.update(status='ok', metrics=metrics, result_file=str((case / 'result.json').relative_to(run)),
                                      result_sha256=analysis.sha256_file(case / 'result.json'))
                    analysis.write_json(case / 'call_001.json', call)
                    analysis.write_json(case / 'record.json', record)
            paths = [root / v for v in analysis.VARIANTS]
            result = analysis.analyze(*paths, selection_path, root / 'analysis', repetitions=100, token_len=len, project=root)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['runs']['structural']['summary']['metrics']['MAP@10'], .5)
            self.assertEqual(result['runs']['lexical']['summary']['metrics']['MAP@10'], 1.)
            self.assertEqual(result['paired_report_statistics']['multiplicity_family_size'], 12)
            self.assertEqual(len(result['project_cluster_statistics']), 2)
            self.assertTrue(result['candidate_head_oracle']['evaluation_only'])
            self.assertEqual(result['candidate_head_oracle']['summary']['metrics']['MAP@10'], 1.)
            self.assertEqual(result['candidate_head_oracle']['oracle_minus_method']['structural']['MAP@10'], .5)
            (root / 'structural/cases/0002/record.json').unlink()
            pending = analysis.analyze(*paths, selection_path, root / 'pending', repetitions=100, token_len=len, project=root)
            self.assertEqual(pending['status'], 'pending')
            self.assertIsNone(pending['runs']['structural']['summary']['metrics'])
            self.assertIsNone(pending['paired_report_statistics'])
            self.assertIsNone(pending['project_cluster_statistics'])
            self.assertIsNone(pending['candidate_head_oracle'])


if __name__ == '__main__':
    unittest.main()
