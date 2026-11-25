"""Boundary and citation checks for the fixed-evidence reranking experiment."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from bug_localization import evidence_reranking as er
from bug_localization.embeddings import digest


class EvidenceRerankingTests(unittest.TestCase):
    def setUp(self):
        self.query = 'Integer inputs lose fractional results; floating inputs keep them.'
        self.saved = {
            'ranking': [{'file_path': p} for p in ['b.py', 'a.py', 'tail.py']],
            'scored_chunks': [
                {'file_path': 'b.py', 'chunk_id': 'b-original',
                 'embedding_text': 'def caller(x):\n    return compute(x)\n',
                 'score': 'SECRET_SCORE'},
                {'file_path': 'a.py', 'chunk_id': 'a-original',
                 'embedding_text': 'result = empty(dtype=x.dtype)\nresult[:] = values\n',
                 'score': 'SECRET_SCORE'},
                {'file_path': 'tail.py', 'chunk_id': 'tail-original',
                 'embedding_text': 'SECRET_TAIL_CODE'},
            ],
            'query': 'SECRET_TRUNCATED_QUERY',
            'ground_truth': ['SECRET_GOLD.py'],
            'metrics': {'SECRET_METRIC': 1},
            'hypothesis': 'SECRET_HYPOTHESIS',
        }
        self.payload = er.build_input(self.saved, self.query, head=2)
        self.files = self.payload['catalog']['files']
        self.chunk = next(k for k, v in self.payload['catalog']['chunks'].items()
                          if v['file_id'] == 'F01')
        self.other_chunk = next(k for k, v in self.payload['catalog']['chunks'].items()
                                if v['file_id'] == 'F02')
        self.value = {
            'reasoning': 'Type inheritance and assignment support the reported behavior.',
            'ranked_ids': ['F01', 'F02'],
            'constraints': [{'id': 'R1', 'kind': 'observed',
                             'quote': 'Integer inputs lose fractional results'}],
            'assessments': [
                {'file_id': 'F01', 'state': 'support', 'constraint_ids': ['R1'],
                 'evidence': [{'chunk_id': self.chunk, 'start': 1, 'end': 2}],
                 'reason': 'The result inherits input type before values are assigned.'},
                {'file_id': 'F02', 'state': 'unknown', 'constraint_ids': [],
                 'evidence': [], 'reason': 'Calling this function does not establish a cause.'},
            ],
        }

    def assertInvalid(self, value, variant='evidence'):
        errors = er.validate_response(value, self.payload, variant)
        self.assertIsInstance(errors, list)
        self.assertTrue(errors, 'Invalid model output must be rejected')

    def test_input_boundary_excludes_labels_scores_and_unselected_code(self):
        text = self.payload['user_content']
        self.assertIn(self.query, text)
        self.assertNotIn('SECRET_', text)
        self.assertEqual(self.payload['query'], self.query)
        self.assertEqual(self.payload['candidates'], ['b.py', 'a.py'])
        self.assertEqual(self.payload['tail'], ['tail.py'])
        self.assertEqual(self.payload['original_ranking'], ['b.py', 'a.py', 'tail.py'])
        self.assertEqual(self.payload['user_content_sha256'], digest(text))

    def test_catalog_uses_alphabetical_ids_without_mutating_saved_result(self):
        before = copy.deepcopy(self.saved)
        payload = er.build_input(self.saved, self.query, head=2)
        self.assertEqual(self.saved, before)
        self.assertEqual(payload['catalog']['files'], {'F01': 'a.py', 'F02': 'b.py'})
        self.assertEqual(payload['catalog']['chunks'][self.chunk]['chunk_id'], 'a-original')
        self.assertEqual(payload['catalog']['chunks'][self.chunk]['line_count'], 2)

    def test_raw_evidence_is_preserved_without_truncation(self):
        original = self.saved['scored_chunks'][1]['embedding_text']
        self.assertEqual(self.payload['evidence']['a.py'],
                         [{'chunk_id': 'a-original', 'text': original}])
        self.assertEqual(self.payload['catalog']['chunks'][self.chunk]['text'], original)
        for line in original.splitlines():
            self.assertIn(line, self.payload['user_content'])

    def test_variants_receive_identical_user_data(self):
        baseline = er.build_messages(self.payload, 'baseline')
        evidence = er.build_messages(self.payload, 'evidence')
        self.assertEqual([m for m in baseline if m['role'] == 'user'],
                         [m for m in evidence if m['role'] == 'user'])
        self.assertNotEqual(baseline[0]['content'], evidence[0]['content'])

    def test_valid_evidence_and_unknown_without_citations_are_accepted(self):
        self.assertEqual(er.validate_response(self.value, self.payload, 'evidence'), [])
        baseline = {k: self.value[k] for k in ['reasoning', 'ranked_ids']}
        self.assertEqual(er.validate_response(baseline, self.payload, 'baseline'), [])

    def test_resolved_ranking_preserves_exact_head_members_and_tail(self):
        self.assertEqual(er.resolve_ranking(self.value, self.payload),
                         ['a.py', 'b.py', 'tail.py'])
        self.value['ranked_ids'].reverse()
        self.assertEqual(er.resolve_ranking(self.value, self.payload),
                         ['b.py', 'a.py', 'tail.py'])

    def test_duplicate_missing_and_unknown_ranked_ids_are_rejected(self):
        for ranked in [['F01', 'F01'], ['F01'], ['F01', 'F99']]:
            with self.subTest(ranked=ranked):
                value = copy.deepcopy(self.value)
                value['ranked_ids'] = ranked
                self.assertInvalid(value)
                self.assertInvalid({k: value[k] for k in ['reasoning', 'ranked_ids']},
                                   variant='baseline')

    def test_assessments_require_each_candidate_exactly_once(self):
        for mode in ['duplicate', 'missing', 'unknown']:
            with self.subTest(mode=mode):
                value = copy.deepcopy(self.value)
                if mode == 'duplicate':
                    value['assessments'][1]['file_id'] = 'F01'
                elif mode == 'missing':
                    value['assessments'].pop()
                else:
                    value['assessments'][1]['file_id'] = 'F99'
                self.assertInvalid(value)

    def test_report_constraints_must_quote_the_report_verbatim(self):
        for quote in ['', ' ', 'Float inputs lose fractional results', self.query + ' fabricated']:
            with self.subTest(quote=quote):
                value = copy.deepcopy(self.value)
                value['constraints'][0]['quote'] = quote
                self.assertInvalid(value)

    def test_duplicate_invalid_and_unresolved_constraint_ids_are_rejected(self):
        for mode in ['duplicate', 'invalid', 'unresolved']:
            with self.subTest(mode=mode):
                value = copy.deepcopy(self.value)
                if mode == 'duplicate':
                    value['constraints'].append(copy.deepcopy(value['constraints'][0]))
                elif mode == 'invalid':
                    value['constraints'][0]['id'] = 'R4'
                else:
                    value['assessments'][0]['constraint_ids'] = ['R2']
                self.assertInvalid(value)

    def test_support_and_contradiction_require_both_constraint_and_code_evidence(self):
        for state in ['support', 'contradiction']:
            for field in ['constraint_ids', 'evidence']:
                with self.subTest(state=state, field=field):
                    value = copy.deepcopy(self.value)
                    value['assessments'][0]['state'] = state
                    value['assessments'][0][field] = []
                    self.assertInvalid(value)

    def test_fabricated_and_cross_file_chunk_references_are_rejected(self):
        for chunk in ['C999', self.other_chunk]:
            with self.subTest(chunk=chunk):
                value = copy.deepcopy(self.value)
                value['assessments'][0]['evidence'][0]['chunk_id'] = chunk
                self.assertInvalid(value)

    def test_code_reference_lines_must_be_integer_and_inside_the_chunk(self):
        for start, end in [(0, 1), (1, 3), (2, 1), (True, 2), (1, False), (1.0, 2), ('1', 2)]:
            with self.subTest(start=start, end=end):
                value = copy.deepcopy(self.value)
                value['assessments'][0]['evidence'][0].update(start=start, end=end)
                self.assertInvalid(value)

    def test_unknown_assessment_state_and_invalid_constraint_kind_are_rejected(self):
        value = copy.deepcopy(self.value)
        value['assessments'][0]['state'] = 'definitely_correct'
        self.assertInvalid(value)
        value = copy.deepcopy(self.value)
        value['constraints'][0]['kind'] = 'imagined_root_cause'
        self.assertInvalid(value)

    def test_non_object_response_and_missing_output_fields_are_rejected(self):
        for value in [None, [], 'text', {}, {'ranked_ids': ['F01', 'F02']}]:
            with self.subTest(value=value):
                self.assertInvalid(value)

    def test_missing_saved_candidate_evidence_is_rejected_before_inference(self):
        saved = copy.deepcopy(self.saved)
        saved['scored_chunks'] = [c for c in saved['scored_chunks'] if c['file_path'] != 'a.py']
        with self.assertRaises(ValueError):
            er.build_input(saved, self.query, head=2)

    def test_schema_serializes_without_test_labels(self):
        for variant in ['baseline', 'evidence']:
            with self.subTest(variant=variant):
                schema = er.response_schema(self.payload, variant)
                self.assertEqual(schema['type'], 'object')
                self.assertNotIn('SECRET_', json.dumps(schema))

    def test_constrained_generation_emits_evidence_before_committing_ranking(self):
        # llama.cpp schema grammars use property order; ranking first would make
        # the supposedly evidence-led decision a post-hoc explanation instead.
        properties = list(er.response_schema(self.payload, 'evidence')['properties'])
        self.assertLess(properties.index('constraints'), properties.index('assessments'))
        self.assertLess(properties.index('assessments'), properties.index('ranked_ids'))


if __name__ == '__main__':
    unittest.main()
