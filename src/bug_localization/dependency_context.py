"""Conservative Python direct-call context on unchanged AST chunks.

No imports are executed. Unknown receivers, ambiguous bindings and external
symbols remain unresolved. Other languages retain their original AST text.
"""

import ast
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import platform
import warnings
from pathlib import Path, PurePosixPath

from .chunk_cache import ChunkCache
from .embeddings import digest
from .snapshot import sha256_file
from .token_ast import ExperimentalChunkCache

DEPENDENCY_ROUTE = 'ast_dependency'
DEFAULTS = {'dependency_max_functions': 2, 'dependency_max_tokens': 128,
            'dependency_function_max_tokens': 64, 'dependency_field_max_tokens': 24,
            'dependency_max_fields': 3}


def validate_dependency_config(config):
    for key, default in DEFAULTS.items():
        value = config.get(key, default)
        maximum = 3 if key in ('dependency_max_functions', 'dependency_max_fields') else 512
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError(f'{key} must be an integer in [0, {maximum}]')


def chain(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        base = chain(node.value)
        return base + (node.attr,) if base else None
    return None


def module_name(path):
    parts = list(PurePosixPath(path).with_suffix('').parts)
    package = parts[-1] == '__init__'
    return '.'.join(parts[:-1] if package else parts), package


@lru_cache(maxsize=256)
def parse_python(path, source_sha256, content):
    """Cache only source-derived symbol metadata, not whole ASTs or labels."""
    raw = content.encode('utf-8')
    module, package = module_name(path)
    result = {'path': path, 'sha256': source_sha256, 'raw': raw, 'module': module,
              'functions': {}, 'classes': {}, 'calls': [], 'field_reads': [], 'fields': defaultdict(list),
              'scopes': {'module': {'parent': None, 'kind': 'module', 'bindings': defaultdict(set),
                                   'globals': set(), 'nonlocals': set(), 'receiver': None, 'wildcard': False}}, 'status': 'parsed'}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            root = ast.parse(content, filename=path)
    except (SyntaxError, ValueError, RecursionError):
        result['status'] = 'unsupported_python_syntax'
        return result
    offsets, position = [], 0
    for line in raw.splitlines(keepends=True):
        offsets.append(position)
        position += len(line)
    offsets.append(position)

    def span(node):
        return (offsets[node.lineno - 1] + node.col_offset,
                offsets[node.end_lineno - 1] + node.end_col_offset)

    def bind(scope, name, kind, target):
        result['scopes'][scope]['bindings'][name].add((kind, target))

    def field(owner, name, node):
        start, end = span(node)
        result['fields'][(owner, name)].append({'name': name, 'class_id': owner,
            'file_path': path, 'source_sha256': source_sha256, 'start_byte': start, 'end_byte': end,
            'start_line': node.lineno, 'end_line': node.end_lineno})

    def walk(node, scope='module', owner_class=None, function=None, parent=None):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start, end = span(node)
            parent_qualname = ''
            if scope in result['functions']:
                parent_qualname = result['functions'][scope]['qualname']
            elif scope in result['classes']:
                parent_qualname = result['classes'][scope]['qualname']
            qualname = (parent_qualname + '.' if parent_qualname else '') + node.name
            identity = f'{path}:{start}:{qualname}'
            is_class = isinstance(node, ast.ClassDef)
            bind(scope, node.name, 'class' if is_class else 'function', identity)
            lexical_parent = scope
            if not is_class:
                while result['scopes'][lexical_parent]['kind'] == 'class':
                    lexical_parent = result['scopes'][lexical_parent]['parent']
            result['scopes'][identity] = {'parent': lexical_parent, 'kind': 'class' if is_class else 'function',
                'bindings': defaultdict(set), 'globals': set(), 'nonlocals': set(), 'receiver': None, 'wildcard': False}
            metadata = {'id': identity, 'file_path': path, 'source_sha256': source_sha256,
                        'qualname': qualname, 'name': node.name, 'start_byte': start, 'end_byte': end,
                        'start_line': node.lineno, 'end_line': node.end_lineno, 'definition_scope': scope}
            if is_class:
                metadata.update({'bases': [chain(b) for b in node.bases]})
                result['classes'][identity] = metadata
                for child in node.body:
                    walk(child, identity, identity, None, node)
            else:
                args = list(node.args.posonlyargs) + list(node.args.args)
                all_args = args + list(node.args.kwonlyargs) + [a for a in (node.args.vararg, node.args.kwarg) if a]
                for arg in all_args:
                    bind(identity, arg.arg, 'parameter', '')
                decorators = [chain(d) for d in node.decorator_list]
                is_static = any(d and d[-1] == 'staticmethod' for d in decorators)
                if owner_class and result['scopes'][scope]['kind'] == 'class' and args and not is_static:
                    result['scopes'][identity]['receiver'] = (args[0].arg, owner_class)
                body_start = span(node.body[0])[0]
                metadata.update({'owner_class': owner_class, 'header_end_byte': body_start,
                    'callable_target': all(d in (('staticmethod',), ('classmethod',)) for d in decorators),
                    'body_statements': [(span(child)[0], span(child)[1],
                        isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant) and isinstance(child.value.value, str))
                        for child in node.body]})
                result['functions'][identity] = metadata
                for child in node.body:
                    walk(child, identity, owner_class, identity, node)
            return
        if isinstance(node, ast.Lambda):
            return
        if isinstance(node, ast.Global):
            result['scopes'][scope]['globals'].update(node.names)
        elif isinstance(node, ast.Nonlocal):
            result['scopes'][scope]['nonlocals'].update(node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bind(scope, alias.asname or alias.name.split('.')[0], 'module', alias.name if alias.asname else alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            parent_module = module if package else module.rpartition('.')[0]
            if node.level:
                parts = parent_module.split('.') if parent_module else []
                base = '.'.join(parts[:len(parts) - node.level + 1]) if node.level <= len(parts) else ''
                target_module = '.'.join(v for v in (base, node.module) if v) if base else None
            else:
                target_module = node.module
            for alias in node.names:
                if alias.name != '*':
                    bind(scope, alias.asname or alias.name, 'import', (target_module, alias.name))
                else:
                    result['scopes'][scope]['wildcard'] = True
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bind(scope, node.id, 'shadow', '')
        elif isinstance(node, ast.Call):
            start, end = span(node)
            result['calls'].append({'start_byte': start, 'end_byte': end, 'scope': scope, 'owner': function,
                                    'expression': chain(node.func)})
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            expression = chain(node)
            if expression and len(expression) == 2 and not (isinstance(parent, ast.Call) and parent.func is node):
                result['field_reads'].append({'start_byte': span(node)[0], 'scope': scope, 'expression': expression})
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and owner_class:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if result['scopes'][scope]['kind'] == 'class' and isinstance(target, ast.Name):
                    field(owner_class, target.id, node)
                elif function and result['functions'][function]['name'] == '__init__':
                    receiver = result['scopes'][scope]['receiver']
                    target_chain = chain(target)
                    if receiver and target_chain and len(target_chain) == 2 and target_chain[0] == receiver[0]:
                        field(owner_class, target_chain[1], node)
        for child in ast.iter_child_nodes(node):
            walk(child, scope, owner_class, function, node)

    try:
        walk(root)
    except RecursionError:
        result['status'] = 'unsupported_python_syntax'
    result['calls'].sort(key=lambda v: v['start_byte'])
    return result


class DependencyGraph:
    def __init__(self, documents):
        self.inventory = [{'file_path': d['file_path'], 'sha256': d['sha256']} for d in documents]
        self.fingerprint = digest(sorted(self.inventory, key=lambda v: v['file_path']))
        self.files = {d['file_path']: parse_python(d['file_path'], d['sha256'], d['content'])
                      for d in documents if PurePosixPath(d['file_path']).suffix.lower() in ('.py', '.pyi')}
        self.modules = defaultdict(list)
        self.functions, self.classes, self.entities = {}, {}, {}
        for path, parsed in self.files.items():
            if parsed['status'] == 'parsed':
                self.modules[parsed['module']].append(path)
                self.functions.update(parsed['functions'])
                self.classes.update(parsed['classes'])
        self.entities = {**self.functions, **self.classes}

    def _name(self, path, scope, name, seen):
        key = (path, scope, name)
        if key in seen or len(seen) > 64:
            return None
        seen = seen | {key}
        info = self.files[path]['scopes'][scope]
        if info['wildcard']:
            return None
        if name in info['nonlocals']:
            return None
        if name in info['globals']:
            if info['bindings'].get(name):
                return None
            return self._name(path, 'module', name, seen)
        bindings = info['bindings'].get(name)
        if bindings:
            if len(bindings) != 1:
                return None
            kind, target = next(iter(bindings))
            if kind in ('function', 'class'):
                return kind, target
            if kind == 'module':
                return ('module', target) if len(self.modules.get(target, [])) == 1 else None
            if kind == 'import':
                module, exported = target
                if not module:
                    return None
                child_module = module + '.' + exported
                if len(self.modules.get(child_module, [])) == 1:
                    return 'module', child_module
                paths = self.modules.get(module, [])
                return self._name(paths[0], 'module', exported, seen) if len(paths) == 1 else None
            return None
        return self._name(path, info['parent'], name, seen) if info['parent'] else None

    def _receiver(self, path, scope, name):
        while scope:
            info = self.files[path]['scopes'][scope]
            if name in info['bindings']:
                receiver = info['receiver']
                return receiver[1] if receiver and receiver[0] == name and info['bindings'][name] == {('parameter', '')} else None
            scope = info['parent']
        return None

    def _member(self, class_id, name, seen=None):
        seen = set() if seen is None else seen
        if class_id in seen or len(seen) > 16:
            return None
        info = self.classes[class_id]
        local = self.files[info['file_path']]['scopes'][class_id]['bindings'].get(name)
        if local:
            if len(local) == 1 and next(iter(local))[0] == 'function':
                return next(iter(local))[1]
            return None
        bases = []
        for base in info['bases']:
            resolved = self._expression(info['file_path'], info['definition_scope'], base, set())
            if not resolved and base == ('object',):
                continue
            if not resolved or resolved[0] != 'class':
                return None
            bases.append(resolved[1])
        # Do not approximate Python's C3 MRO or unknown external base classes.
        return self._member(bases[0], name, seen | {class_id}) if len(bases) == 1 else None

    def _expression(self, path, scope, expression, seen):
        if not expression:
            return None
        if len(expression) == 2:
            receiver = self._receiver(path, scope, expression[0])
            if receiver:
                method = self._member(receiver, expression[1])
                return ('function', method) if method else None
        result = self._name(path, scope, expression[0], seen)
        for attribute in expression[1:]:
            if not result:
                return None
            kind, target = result
            if kind == 'module':
                child = target + '.' + attribute
                if len(self.modules.get(child, [])) == 1:
                    result = 'module', child
                else:
                    files = self.modules.get(target, [])
                    result = self._name(files[0], 'module', attribute, seen) if len(files) == 1 else None
            elif kind == 'class':
                method = self._member(target, attribute)
                result = ('function', method) if method else None
            else:
                return None
        return result

    def references(self, path, start, end):
        parsed = self.files.get(path)
        if not parsed or parsed['status'] != 'parsed':
            return [], [], 0, parsed['status'] if parsed else 'language_not_supported_for_dependency_resolution'
        selected, unresolved = {}, 0
        for call in parsed['calls']:
            if not start <= call['start_byte'] < end:
                continue
            resolved = self._expression(path, call['scope'], call['expression'], set())
            target = resolved[1] if resolved and resolved[0] == 'function' else None
            if resolved and resolved[0] == 'class':
                target = self._member(resolved[1], '__init__')
            if not target:
                unresolved += 1
                continue
            function = self.functions[target]
            if not function['callable_target']:
                unresolved += 1
                continue
            if target == call['owner'] or (function['file_path'] == path and function['start_byte'] < end and function['end_byte'] > start):
                continue
            priority = 0 if call['expression'] and self._receiver(path, call['scope'], call['expression'][0]) else (1 if function['file_path'] == path else 2)
            if target not in selected:
                selected[target] = {'target': function, 'count': 0, 'first_call_byte': call['start_byte'],
                                    'priority': priority, 'call_expression': '.'.join(call['expression'])}
            selected[target]['count'] += 1
        dependencies = sorted(selected.values(), key=lambda v: (v['priority'], -v['count'], v['first_call_byte'], v['target']['id']))
        fields = {}
        for ref in parsed['field_reads']:
            if not start <= ref['start_byte'] < end:
                continue
            owner = self._receiver(path, ref['scope'], ref['expression'][0])
            definitions = parsed['fields'].get((owner, ref['expression'][1]), []) if owner else []
            if definitions:
                definition = definitions[0]
                if not (definition['start_byte'] < end and definition['end_byte'] > start):
                    fields[(owner, definition['name'])] = definition
        return dependencies, list(fields.values()), unresolved, 'parsed'


class DependencyEnricher:
    def __init__(self, baseline, graph, documents, tokenizer, config):
        self.baseline, self.graph, self.tokenizer, self.config = baseline, graph, tokenizer, config
        self.documents = {d['file_path']: d for d in documents}
        self.settings = {k: config.get(k, default) for k, default in DEFAULTS.items()}

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def previews(self, target):
        raw = self.graph.files[target['file_path']]['raw']
        label = f"[Direct callee {target['file_path']}::{target['qualname']}]\n"
        full = label + raw[target['start_byte']:target['end_byte']].decode('utf-8')
        limit = self.settings['dependency_function_max_tokens']
        if self.count(full) <= limit:
            return [(full, False)]
        header = raw[target['start_byte']:target['header_end_byte']].decode('utf-8').rstrip()
        text, options = label + header, []
        if self.count(text) <= limit:
            options.append((text, True))
        for start, end, docstring in target['body_statements']:
            if docstring:
                continue  # Never truncate a long docstring into fictitious code.
            line_start = raw.rfind(b'\n', 0, start) + 1
            indentation = raw[line_start:start]
            if indentation.strip():
                indentation = b'    '  # Render a one-line function body as a preview.
            candidate = text + '\n' + indentation.decode('utf-8') + raw[start:end].decode('utf-8')
            if self.count(candidate) > limit:
                break
            text = candidate
            options.append((text, True))
        return options[::-1]

    def enrich(self, original):
        dependencies, fields, unresolved, status = self.graph.references(
            original['file_path'], original['start_byte'], original['end_byte'])
        base, context, used_functions, used_fields, skipped = original['embedding_text'], '', [], [], Counter()

        def fits(text):
            return (self.count(text) <= self.settings['dependency_max_tokens'] and
                    len(self.tokenizer.encode(base + ('\n\n' + text if text else ''), add_special_tokens=True)) <= self.config['max_tokens'])

        for definition in fields[:self.settings['dependency_max_fields']]:
            raw = self.graph.files[definition['file_path']]['raw']
            snippet = raw[definition['start_byte']:definition['end_byte']].decode('utf-8')
            candidate = (context + '\n' if context else '') + '[Field] ' + snippet
            if self.count(candidate) <= self.settings['dependency_field_max_tokens'] and fits(candidate):
                context = candidate
                used_fields.append(definition)
        for dep in dependencies:
            if len(used_functions) >= self.settings['dependency_max_functions']:
                break
            for snippet, truncated in self.previews(dep['target']):
                candidate = context + ('\n\n' if context else '') + snippet
                if fits(candidate):
                    context = candidate
                    target = dep['target']
                    used_functions.append({k: target[k] for k in ('id', 'file_path', 'source_sha256', 'qualname',
                        'start_byte', 'end_byte', 'start_line', 'end_line')} | {
                        'call_expression': dep['call_expression'], 'call_count_in_chunk': dep['count'],
                        'resolution': 'unique_static_binding', 'preview': snippet, 'preview_truncated': truncated})
                    break
            else:
                skipped['insufficient_token_budget'] += 1
        embedded = base + ('\n\n' + context if context else '')
        chunk = {**original, 'route': DEPENDENCY_ROUTE, 'original_chunk_id': original['chunk_id'],
            'embedding_text': embedded, 'token_count': len(self.tokenizer.encode(embedded, add_special_tokens=True)),
            'dependency_context': context, 'dependency_context_tokens': self.count(context),
            'dependency_functions': used_functions, 'dependency_fields': used_fields,
            'dependency_resolved_candidates': len(dependencies), 'dependency_unresolved_calls': unresolved,
            'dependency_resolution_status': status, 'dependency_skip_counts': dict(skipped),
            'original_embedding_sha256': hashlib.sha256(base.encode('utf-8')).hexdigest()}
        chunk['chunk_id'] = digest([DEPENDENCY_ROUTE, original['chunk_id'], embedded])
        if not fits(context):
            raise ValueError('Dependency context exceeds configured token limits')
        return chunk

    def chunks(self, path, content, route):
        if route != DEPENDENCY_ROUTE or self.documents[path]['content'] != content:
            raise ValueError('Dependency snapshot or route mismatch')
        chunks, parents = self.baseline.get(self.documents[path], 'blagent_ast')
        return [self.enrich(chunk) for chunk in chunks], parents


class DependencyChunkCache(ChunkCache):
    def __init__(self, project, chunker, model_fingerprint):
        self.chunker = chunker
        self.identity = {'format': 'direct-python-dependency-v1', 'model_fingerprint': model_fingerprint,
            'python': platform.python_version(),
            'snapshot_inventory': chunker.graph.fingerprint, 'settings': chunker.settings,
            'baseline': chunker.baseline.index_identity(list(chunker.documents.values()), 'blagent_ast'),
            'implementation_sha256': sha256_file(Path(__file__))}
        self.folder = project / 'cache/chunks' / digest(self.identity)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0


class DependencyRoutingCache:
    """Delegate all old routes unchanged; graph identity covers the full snapshot."""
    def __init__(self, project, tokenizer, config, model_fingerprint):
        self.baseline = ExperimentalChunkCache(project, tokenizer, config, model_fingerprint)
        self.project, self.tokenizer, self.config, self.model_fingerprint = project, tokenizer, config, model_fingerprint
        self.dependency, self.previous_hits, self.previous_misses = None, 0, 0

    def prepare(self, documents):
        if self.dependency:
            self.previous_hits += self.dependency.hits
            self.previous_misses += self.dependency.misses
        graph = DependencyGraph(documents)
        chunker = DependencyEnricher(self.baseline, graph, documents, self.tokenizer, self.config)
        self.dependency = DependencyChunkCache(self.project, chunker, self.model_fingerprint)

    @property
    def hits(self):
        return self.baseline.hits + self.previous_hits + (self.dependency.hits if self.dependency else 0)

    @property
    def misses(self):
        return self.baseline.misses + self.previous_misses + (self.dependency.misses if self.dependency else 0)

    def get(self, document, route):
        if route != DEPENDENCY_ROUTE:
            return self.baseline.get(document, route)
        if not self.dependency:
            raise ValueError('Prepare the complete dependency snapshot before chunking')
        return self.dependency.get(document, route)

    def index_identity(self, documents, route):
        if route == DEPENDENCY_ROUTE:
            return {'dependency_pipeline_fingerprint': digest(self.dependency.identity)}
        return self.baseline.index_identity(documents, route)
