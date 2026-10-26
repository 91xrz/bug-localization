import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.ast_chunking import RoutingChunkCache as OriginalCache
from bug_localization.dependency_context import (DEPENDENCY_ROUTE, DependencyGraph, DependencyEnricher,
                                                DependencyRoutingCache, validate_dependency_config)
from test_vector import CharacterTokenizer


def document(path, text):
    return {'file_path': path, 'content': text, 'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()}


def config(**changes):
    base = json.loads((PROJECT / 'configs/experiments/ast_dependency_experiment.json').read_text(encoding='utf-8'))
    return {**base, **changes}


def function(graph, name):
    return next(f for f in graph.functions.values() if f['qualname'] == name)


def references(graph, name):
    f = function(graph, name)
    return graph.references(f['file_path'], f['start_byte'], f['end_byte'])


def chunk_for(graph, name):
    f = function(graph, name)
    raw = graph.files[f['file_path']]['raw']
    code = raw[f['start_byte']:f['end_byte']].decode('utf-8')
    prefix = 'Path: ' + f['file_path']
    return {'file_path': f['file_path'], 'route': 'blagent_ast', 'chunk_id': 'original',
            'start_byte': f['start_byte'], 'end_byte': f['end_byte'],
            'start_line': f['start_line'], 'end_line': f['end_line'],
            'code': code, 'prefix': prefix, 'embedding_text': prefix + '\n\n' + code}


class DependencyGraphTests(unittest.TestCase):
    def test_local_import_alias_receiver_single_inheritance_and_fields(self):
        docs = [document('pkg/__init__.py', ''),
            document('pkg/helpers.py', 'def helper(x):\n    return x + 1\n'),
            document('pkg/base.py', 'class Base:\n    def parent(self):\n        return 1\n'),
            document('pkg/main.py', 'from .helpers import helper as h\nfrom .base import Base\n'
                'class Child(Base):\n    def __init__(self):\n        self.mode = 1\n'
                '    def local(self):\n        return 2\n'
                '    def run(self):\n        self.local()\n        h(1)\n        h(2)\n'
                '        self.parent()\n        other.dynamic()\n        return self.mode\n')]
        graph = DependencyGraph(docs)
        dependencies, fields, unresolved, status = references(graph, 'Child.run')
        by_name = {d['target']['qualname']: d for d in dependencies}
        self.assertEqual(set(by_name), {'Child.local', 'helper', 'Base.parent'})
        self.assertEqual(by_name['helper']['count'], 2)
        self.assertEqual(by_name['helper']['target']['file_path'], 'pkg/helpers.py')
        self.assertEqual(fields[0]['name'], 'mode')
        self.assertEqual(fields[0]['start_line'], 5)
        self.assertEqual(status, 'parsed')
        self.assertEqual(unresolved, 1)

    def test_module_alias_relative_parent_import_and_reexports(self):
        docs = [document('pkg/__init__.py', ''), document('pkg/sub/__init__.py', ''),
            document('pkg/defs.py', 'def tiny():\n    return 1\n'),
            document('pkg/reexport.py', 'from .defs import tiny as exported\n'),
            document('pkg/sub/main.py', 'import pkg.defs as alias\nfrom ..reexport import exported\n'
                     'def caller():\n    alias.tiny()\n    exported()\n')]
        deps, _, unresolved, _ = references(DependencyGraph(docs), 'caller')
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0]['target']['file_path'], 'pkg/defs.py')
        self.assertEqual(deps[0]['count'], 2)
        self.assertEqual(unresolved, 0)

    def test_shadowing_ambiguous_definitions_unknown_receivers_and_wildcards_fail_closed(self):
        cases = [
            'def helper():\n    pass\ndef caller(helper):\n    helper()\n',
            'def helper():\n    pass\ndef caller():\n    helper()\n    helper = 2\n',
            'def helper():\n    pass\ndef helper():\n    pass\ndef caller():\n    helper()\n',
            'def helper():\n    pass\ndef caller():\n    obj.helper()\n',
            'from unknown import *\ndef helper():\n    pass\ndef caller():\n    helper()\n',
            'class C:\n    def helper(self):\n        pass\n    def caller(self):\n        self.helper()\n        self = 2\n',
            'class A:\n    def helper(self):\n        pass\nclass B:\n    pass\n'
            'class C(A, B):\n    def caller(self):\n        self.helper()\n']
        for text in cases:
            with self.subTest(text=text):
                graph = DependencyGraph([document('a.py', text), document('unrelated.py', 'def helper():\n    pass\n')])
                name = 'C.caller' if 'class C' in text else 'caller'
                deps, _, unresolved, _ = references(graph, name)
                self.assertEqual(deps, [])
                self.assertGreater(unresolved, 0)

    def test_recursion_does_not_expand_self_or_recurse_transitively(self):
        docs = [document('a.py', 'def a():\n    a()\n    b()\ndef b():\n    a()\n')]
        deps, _, _, _ = references(DependencyGraph(docs), 'a')
        self.assertEqual([d['target']['qualname'] for d in deps], ['b'])

    def test_decorated_unknown_targets_and_cyclic_reexports_are_not_guessed(self):
        docs = [document('a.py', 'from b import helper\n@unknown\ndef wrapped():\n    pass\n'
                          'def caller():\n    wrapped()\n    helper()\n'),
                document('b.py', 'from a import helper\n')]
        deps, _, unresolved, _ = references(DependencyGraph(docs), 'caller')
        self.assertEqual(deps, [])
        self.assertEqual(unresolved, 2)


class DependencyEnrichmentTests(unittest.TestCase):
    def test_truncated_method_preview_preserves_body_indentation(self):
        docs = [document('a.py', 'class C:\n    def dep(self):\n        value = 1\n'
                    '        long_value = "' + 'x' * 200 + '"\n        return value\n'
                    '    def caller(self):\n        self.dep()\n')]
        graph = DependencyGraph(docs)
        enricher = DependencyEnricher(None, graph, docs, CharacterTokenizer(), config(dependency_function_max_tokens=96))
        preview, truncated = enricher.previews(function(graph, 'C.dep'))[0]
        self.assertTrue(truncated)
        self.assertIn('\n        value = 1', preview)
        ast.parse(preview.split('\n', 1)[1])

    def test_enrichment_preserves_focal_bytes_and_caps_direct_functions_and_tokens(self):
        docs = [document('a.py', 'def x():\n    return 1\ndef y():\n    return 2\ndef z():\n    return 3\n'
                    'def caller():\n    x()\n    y()\n    z()\n')]
        graph = DependencyGraph(docs)
        original = chunk_for(graph, 'caller')
        cfg = config(dependency_function_max_tokens=96, dependency_max_tokens=256)
        enricher = DependencyEnricher(None, graph, docs, CharacterTokenizer(), cfg)
        result = enricher.enrich(original)
        self.assertEqual(len(result['dependency_functions']), 2)
        self.assertEqual([d['qualname'] for d in result['dependency_functions']], ['x', 'y'])
        self.assertEqual(result['code'], original['code'])
        self.assertEqual(result['prefix'], original['prefix'])
        self.assertTrue(result['embedding_text'].startswith(original['embedding_text'] + '\n\n'))
        self.assertLessEqual(result['token_count'], 512)
        self.assertLessEqual(result['dependency_context_tokens'], 256)
        self.assertEqual(result['route'], DEPENDENCY_ROUTE)
        self.assertNotEqual(result['chunk_id'], original['chunk_id'])

    def test_zero_budget_is_exact_no_context_control_and_full_focal_chunk_is_not_truncated(self):
        docs = [document('a.py', 'def helper():\n    return 1\ndef caller():\n    helper()\n')]
        graph = DependencyGraph(docs)
        original = chunk_for(graph, 'caller')
        for cfg in [config(dependency_max_tokens=0), config(max_tokens=len(original['embedding_text']) + 1)]:
            result = DependencyEnricher(None, graph, docs, CharacterTokenizer(), cfg).enrich(original)
            self.assertEqual(result['embedding_text'], original['embedding_text'])
            self.assertEqual(result['dependency_context_tokens'], 0)

    def test_fields_are_source_backed_and_nonpython_or_legacy_syntax_is_unchanged(self):
        docs = [document('a.py', 'class C:\n    def __init__(self):\n        self.mode = 1\n'
                                  '    def caller(self):\n        return self.mode\n'),
                document('b.py', 'print "legacy"\n'), document('C.java', 'class C {}\n')]
        graph = DependencyGraph(docs)
        result = DependencyEnricher(None, graph, docs, CharacterTokenizer(), config()).enrich(chunk_for(graph, 'C.caller'))
        self.assertEqual(result['dependency_fields'][0]['name'], 'mode')
        self.assertIn('self.mode = 1', result['dependency_context'])
        self.assertEqual(graph.references('b.py', 0, 10)[3], 'unsupported_python_syntax')
        self.assertEqual(graph.references('C.java', 0, 10)[3], 'language_not_supported_for_dependency_resolution')

    def test_dependency_edit_invalidates_cache_while_baseline_identity_and_chunks_stay_unchanged(self):
        main = document('pkg/main.py', 'from .helpers import helper\ndef caller():\n    return helper()\n')
        helper = document('pkg/helpers.py', 'def helper():\n    return 1\n')
        docs = [document('pkg/__init__.py', ''), main, helper]
        cfg = config(dependency_function_max_tokens=128, dependency_max_tokens=256)
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            old = OriginalCache(project, CharacterTokenizer(), cfg, 'model')
            new = DependencyRoutingCache(project, CharacterTokenizer(), cfg, 'model')
            for route in ('dynamic', 'method', 'fixed', 'blagent_ast'):
                self.assertEqual(old.get(main, route), new.get(main, route))
                self.assertEqual(old.index_identity(docs, route), new.index_identity(docs, route))
            new.prepare(docs)
            first = new.get(main, DEPENDENCY_ROUTE)
            identity = new.index_identity(docs, DEPENDENCY_ROUTE)
            self.assertEqual(first, new.get(main, DEPENDENCY_ROUTE))
            self.assertEqual(''.join(c['code'] for c in first[0]), main['content'])
            self.assertTrue(any(c['dependency_functions'] for c in first[0]))
            changed = [docs[0], main, document('pkg/helpers.py', 'def helper():\n    return 999\n')]
            new.prepare(changed)
            second = new.get(main, DEPENDENCY_ROUTE)
            self.assertNotEqual(identity, new.index_identity(changed, DEPENDENCY_ROUTE))
            self.assertNotEqual(first, second)

    def test_limits_reject_invalid_function_caps(self):
        for value in [True, 4, -1, 1.5]:
            with self.assertRaises(ValueError):
                validate_dependency_config(config(dependency_max_functions=value))


if __name__ == '__main__':
    unittest.main()
