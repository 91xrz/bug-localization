"""Exact-source, bounded-cost and structural-vs-lexical selector checks."""
import hashlib
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bug_localization.context_evidence import extract_anchors, render_segments, select_context
from bug_localization.context_evidence import _greedy_candidates


def word_tokens(text):
    return len(text.split())


class ContextEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.code = (
            'def process(raw):\n'
            '    payload = raw.strip()\n'
            '    limit = len(payload)\n'
            '    ready = limit > 1\n'
            '    if ready:\n'
            '        transformed = payload.upper()\n'
            '        note = "unchanged"\n'
            '        raise ValueError(transformed)\n'
        )
        (self.root / 'module.py').write_text(self.code)

    def select(self, variant='structural', **kwargs):
        settings = dict(source_root=self.root, file_path='module.py',
                        report='The operation raises `ValueError`.',
                        existing_texts=['        raise ValueError(transformed)\n'],
                        token_len=word_tokens, variant=variant, budget_tokens=256)
        settings.update(kwargs)
        return select_context(**settings)

    def test_structural_adds_variable_definition_and_control_beyond_lexical_window(self):
        lexical = self.select('lexical')
        structural = self.select('structural')
        self.assertIn('note = "unchanged"', lexical['rendered_text'])
        self.assertNotIn('transformed = payload.upper()', lexical['rendered_text'])
        self.assertIn('transformed = payload.upper()', structural['rendered_text'])
        self.assertIn('if ready:', structural['rendered_text'])
        self.assertNotIn('note = "unchanged"', structural['rendered_text'])
        reasons = [reason for s in structural['segments'] for reason in s['reason']]
        self.assertTrue(any(reason.startswith('one_hop_definition:transformed') for reason in reasons))
        self.assertIn('enclosing_control_condition', reasons)
        self.assertEqual(structural['effective_variant'], 'structural')
        self.assertIsNone(structural['fallback_reason'])

    def test_every_segment_is_exact_source_lines_with_recomputable_hash(self):
        output = self.select()
        lines = self.code.splitlines(keepends=True)
        for segment in output['segments']:
            expected = ''.join(lines[segment['start_line'] - 1:segment['end_line']])
            self.assertEqual(segment['text'], expected)
            self.assertEqual(segment['text_sha256'], hashlib.sha256(expected.encode()).hexdigest())
        self.assertEqual(output['source_sha256'], hashlib.sha256(self.code.encode()).hexdigest())
        self.assertEqual(output['rendered_text'], render_segments('module.py', output['segments']))

    def test_deterministic_results_and_original_file_is_unchanged(self):
        before = (self.root / 'module.py').read_bytes()
        self.assertEqual(self.select(), self.select())
        self.assertEqual((self.root / 'module.py').read_bytes(), before)

    def test_lazy_coverage_order_matches_full_rescoring_with_ties_and_skips(self):
        rng = random.Random(418)
        candidates = [{'lines': {rng.randint(1, 12)}, 'focus': rng.randint(1, 12),
                       'score': rng.randint(1, 5), 'reason': rng.choice(['anchor', 'use', 'guard']),
                       'matched_anchors': rng.sample(['a', 'b', 'c', 'd'], rng.randint(1, 3)),
                       'id': i} for i in range(100)]
        seen, expected, remaining = set(), [], list(candidates)
        while remaining:
            candidate = max(remaining, key=lambda c: (
                c['score'] + 2 * len(set(c['matched_anchors']) - seen),
                -abs(min(c['lines']) - c['focus']), -min(c['lines']), c['reason']))
            remaining.remove(candidate)
            expected.append(candidate['id'])
            if candidate['id'] % 3:
                seen.update(candidate['matched_anchors'])
        seen, actual = set(), []
        for candidate in _greedy_candidates(candidates, seen):
            actual.append(candidate['id'])
            if candidate['id'] % 3:
                seen.update(candidate['matched_anchors'])
        self.assertEqual(actual, expected)

    def test_same_named_variables_in_other_functions_do_not_create_edges(self):
        code = ('def unrelated():\n'
                '    transformed = "SECRET_OTHER_SCOPE"\n'
                '    return transformed\n\n' + self.code)
        (self.root / 'module.py').write_text(code)
        output = self.select()
        self.assertNotIn('SECRET_OTHER_SCOPE', output['rendered_text'])

    def test_nested_function_scope_does_not_supply_outer_definitions(self):
        code = ('def process(raw):\n'
                '    def inner():\n'
                '        transformed = "SECRET_INNER_SCOPE"\n'
                '        return transformed\n'
                '    transformed = raw.strip()\n'
                '    filler = 0\n'
                '    raise ValueError(transformed)\n')
        (self.root / 'module.py').write_text(code)
        output = self.select()
        self.assertIn('transformed = raw.strip()', output['rendered_text'])
        self.assertNotIn('SECRET_INNER_SCOPE', output['rendered_text'])

    def test_one_hop_does_not_expand_to_transitive_variable_dependencies(self):
        output = self.select()
        self.assertIn('transformed = payload.upper()', output['rendered_text'])
        self.assertNotIn('payload = raw.strip()', output['rendered_text'])

    def test_structural_can_follow_a_definition_to_its_use(self):
        code = ('def process(raw):\n'
                '    output = dangerous(raw)\n'
                '    unrelated = 0\n'
                '    unrelated += 1\n'
                '    return output\n')
        (self.root / 'module.py').write_text(code)
        output = self.select(report='Calling `dangerous` fails.',
                             existing_texts=['    output = dangerous(raw)\n'])
        self.assertIn('return output', output['rendered_text'])
        reasons = [reason for s in output['segments'] for reason in s['reason']]
        self.assertIn('one_hop_use:output', reasons)

    def test_else_anchor_keeps_the_real_else_header_and_parent_condition_together(self):
        code = ('def process(ready):\n'
                '    if ready:\n'
                '        return 1\n'
                '    else:\n'
                '        unused = 0\n'
                '        raise ValueError("bad")\n')
        (self.root / 'module.py').write_text(code)
        output = self.select(existing_texts=['        raise ValueError("bad")\n'])
        self.assertIn('if ready:', output['rendered_text'])
        self.assertIn('else:', output['rendered_text'])
        reasons = [reason for s in output['segments'] for reason in s['reason']]
        self.assertIn('enclosing_control_branch:else', reasons)
        self.assertNotIn('enclosing_control_condition', reasons)

    def test_else_guard_cannot_be_partially_added_under_tight_budget(self):
        code = ('def process(ready):\n'
                '    if ready:\n'
                '        return 1\n'
                '    else:\n'
                '        raise ValueError("bad")\n')
        (self.root / 'module.py').write_text(code)
        output = self.select(existing_texts=['        raise ValueError("bad")\n'],
                             token_len=len, budget_tokens=90)
        self.assertEqual('if ready:' in output['rendered_text'], 'else:' in output['rendered_text'])

    def test_repeated_else_in_unrelated_excerpt_does_not_remove_target_branch_direction(self):
        code = ('def unrelated(flag):\n'
                '    if flag:\n'
                '        return 1\n'
                '    else:\n'
                '        return 2\n'
                'def process(ready):\n'
                '    if ready:\n'
                '        return 3\n'
                '    else:\n'
                '        raise ValueError("bad")\n')
        (self.root / 'module.py').write_text(code)
        output = self.select(existing_texts=['    else:\n', '        raise ValueError("bad")\n'])
        self.assertIn('if ready:', output['rendered_text'])
        self.assertIn('else:', output['rendered_text'])
        selected_lines = {i for s in output['segments']
                          for i in range(s['start_line'], s['end_line'] + 1)}
        self.assertIn(9, selected_lines)
        self.assertNotIn(4, selected_lines)

    def test_elif_uses_its_actual_header_without_positive_prior_if(self):
        code = ('def process(first, second):\n'
                '    if first:\n'
                '        return 1\n'
                '    elif second:\n'
                '        raise ValueError("bad")\n')
        (self.root / 'module.py').write_text(code)
        output = self.select(existing_texts=['        raise ValueError("bad")\n'])
        self.assertIn('elif second:', output['rendered_text'])
        self.assertNotIn('if first:', output['rendered_text'])

    def test_except_and_finally_anchors_use_their_own_branch_headers(self):
        for branch in ['except OSError:', 'finally:']:
            with self.subTest(branch=branch):
                code = ('def process():\n'
                        '    try:\n'
                        '        operation()\n'
                        f'    {branch}\n'
                        '        raise ValueError("bad")\n')
                (self.root / 'module.py').write_text(code)
                output = self.select(report='`ValueError` fails.',
                                     existing_texts=['        raise ValueError("bad")\n'])
                self.assertIn(branch, output['rendered_text'])
                self.assertNotIn('try:', output['rendered_text'])

    def test_code_is_never_executed(self):
        marker = self.root / 'executed'
        code = f'open({str(marker)!r}, "w").write("executed")\nraise ValueError("x")\n'
        (self.root / 'module.py').write_text(code)
        self.select(existing_texts=[])
        self.assertFalse(marker.exists())

    def test_python2_syntax_falls_back_to_identical_lexical_selection(self):
        (self.root / 'module.py').write_text('print "before"\nraise ValueError, "bad"\n')
        structural = self.select(existing_texts=[])
        lexical = self.select('lexical', existing_texts=[])
        self.assertEqual(structural['effective_variant'], 'lexical')
        self.assertEqual(structural['fallback_reason'], 'python_ast_unavailable:SyntaxError')
        self.assertEqual(structural['rendered_text'], lexical['rendered_text'])

    def test_no_matching_anchor_returns_empty_context_without_header_noise(self):
        output = self.select(report='The `AbsolutelyMissingSymbol` fails.')
        self.assertEqual(output['segments'], [])
        self.assertEqual(output['rendered_text'], '')
        self.assertEqual(output['token_count'], 0)

    def test_zero_budget_emits_nothing(self):
        output = self.select(budget_tokens=0)
        self.assertEqual(output['segments'], [])
        self.assertEqual(output['token_count'], 0)

    def test_budget_counts_rendered_path_and_line_headers(self):
        for budget in [1, 10, 20, 40, 256]:
            with self.subTest(budget=budget):
                output = self.select(token_len=len, budget_tokens=budget)
                self.assertLessEqual(len(output['rendered_text']), budget)
                self.assertEqual(output['token_count'], len(output['rendered_text']))

    def test_long_source_line_is_skipped_whole_and_reported(self):
        code = 'raise ValueError("' + ('X' * 500) + '")\n'
        (self.root / 'module.py').write_text(code)
        output = self.select(existing_texts=[], token_len=len, budget_tokens=100)
        self.assertEqual(output['rendered_text'], '')
        self.assertEqual(output['oversized_lines_skipped'], [1])

    def test_existing_nonempty_lines_are_not_emitted_twice(self):
        output = self.select(existing_texts=[self.code])
        self.assertEqual(output['segments'], [])
        self.assertEqual(output['original_covered_line_count'], 8)

    def test_crlf_source_spans_preserve_original_line_endings(self):
        (self.root / 'module.py').write_bytes(self.code.replace('\n', '\r\n').encode())
        output = self.select()
        for segment in output['segments']:
            self.assertIn('\r\n', segment['text'])

    def test_declared_non_utf8_python_encoding_is_supported(self):
        raw = '# coding: latin-1\nmessage = "caf\xe9"\nraise ValueError(message)\n'.encode('latin-1')
        (self.root / 'module.py').write_bytes(raw)
        output = self.select(existing_texts=[])
        self.assertEqual(output['source_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertIn('caf\xe9', output['rendered_text'])

    def test_path_traversal_absolute_paths_and_backslashes_are_rejected(self):
        for path in ['../module.py', '/tmp/module.py', './module.py', 'a/../module.py',
                     'a\\module.py', 'a//module.py', 'C:/module.py']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.select(file_path=path)

    def test_symlink_to_outside_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            path = Path(outside) / 'secret.py'
            path.write_text('raise ValueError("outside")\n')
            (self.root / 'escape.py').symlink_to(path)
            with self.assertRaises(ValueError):
                self.select(file_path='escape.py')

    def test_pinned_hash_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.select(expected_sha256='0' * 64)
        self.select(expected_sha256=hashlib.sha256(self.code.encode()).hexdigest())

    def test_existing_blocks_are_strings_not_label_bearing_result_objects(self):
        with self.assertRaises(ValueError):
            self.select(existing_texts=[{'ground_truth': ['secret.py']}])

    def test_explicit_identifiers_exceptions_and_quoted_literals_are_extracted(self):
        anchors = extract_anchors('`dtype` loses decimals with name=None and raises ValueError for "bad mode".')
        text = {a['text'] for a in anchors}
        self.assertTrue({'dtype', 'None', 'ValueError', 'bad mode'} <= text)
        self.assertEqual(anchors, extract_anchors('`dtype` loses decimals with name=None and raises ValueError for "bad mode".'))

    def test_invalid_variant_budget_and_token_counter_are_rejected(self):
        for settings in [{'variant': 'unknown'}, {'budget_tokens': -1},
                         {'budget_tokens': True}, {'token_len': lambda text: 1.0}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.select(**settings)


if __name__ == '__main__':
    unittest.main()
