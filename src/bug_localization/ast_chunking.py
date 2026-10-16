"""BLagent-style recursive AST packing with source and token-budget safeguards.

The supplied BLagent wrappers delegate to LlamaIndex CodeSplitter. This route
adapts that recursive sibling-packing strategy, independently of any vector
store, while preserving whitespace, oversized leaves and UTF-8 byte offsets.
"""

import bisect
import hashlib
import json
from pathlib import Path

from .chunking import PythonChunker, dynamic_ranges, fixed_ranges
from .chunk_cache import ChunkCache
from .embeddings import digest
from .languages import CLASS_NODES, FUNCTION_NODES, grammar_versions, language_for_path, parser_for_language
from .snapshot import sha256_file


def ast_ranges(root, raw, max_chars):
    """Recursively pack adjacent AST children; descend into oversized children.

    Gaps and trailing source belong to exactly one range. Oversized leaves are
    explicitly sliced at Unicode character boundaries instead of disappearing.
    """
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError('ast_max_chars must be a positive integer')

    def leaf(start, end, reason):
        text = raw[start:end].decode('utf-8')
        result = []
        position = start
        for offset in range(0, len(text), max_chars):
            stop = position + len(text[offset:offset + max_chars].encode('utf-8'))
            result.append((position, stop, reason))
            position = stop
        return result

    def visit(node, start, end, depth=0):
        children = [c for c in node.children if c.end_byte > c.start_byte]
        if not children or depth >= 200:
            return leaf(start, end, 'ast_oversized_leaf' if not children else 'ast_depth_fallback')
        result, current, previous = [], start, start
        for child in children:
            child_size = len(raw[child.start_byte:child.end_byte].decode('utf-8'))
            if child_size > max_chars:
                if previous > current:
                    result.append((current, previous, 'ast_recursive'))
                result.extend(visit(child, previous, child.end_byte, depth + 1))
                current = child.end_byte
            elif len(raw[current:child.end_byte].decode('utf-8')) > max_chars:
                if previous > current:
                    result.append((current, previous, 'ast_recursive'))
                current = previous
            previous = child.end_byte
        if current < end:
            if len(raw[current:end].decode('utf-8')) <= max_chars:
                result.append((current, end, 'ast_recursive'))
            else:
                result.extend(leaf(current, end, 'ast_gap_character_boundary'))
        return result

    result = visit(root, 0, len(raw)) if raw else [(0, 0, 'ast_empty')]
    if ''.join(raw[start:end].decode('utf-8') for start, end, _ in result).encode('utf-8') != raw:
        raise ValueError('AST ranges do not reconstruct the complete source')
    return result


def syntax_context(tree, raw, language):
    """Normalize language-specific class/function nodes for real prefixes."""
    scopes, methods, points, boundaries = [], [], [], {0, len(raw)}
    stack = [(tree.root_node, [], False)]
    while stack:
        node, classes, inside_function = stack.pop()
        is_class = node.type in CLASS_NODES[language]
        is_function = node.type in FUNCTION_NODES[language]
        child_classes = classes
        if is_class or is_function:
            name_node = node.child_by_field_name('name')
            if name_node is None and language == 'rust' and node.type == 'impl_item':
                name_node = node.child_by_field_name('type')
            if name_node is None:
                name_node = node.child_by_field_name('declarator')
                while name_node is not None and name_node.child_by_field_name('declarator') is not None:
                    name_node = name_node.child_by_field_name('declarator')
            name = raw[name_node.start_byte:name_node.end_byte].decode('utf-8') if name_node is not None else '<anonymous>'
            body = node.child_by_field_name('body')
            signature_end = body.start_byte if body is not None else node.end_byte
            signature = raw[node.start_byte:signature_end].decode('utf-8').rstrip()
            owner = list(classes)
            if language == 'go' and node.type == 'method_declaration':
                receiver = node.child_by_field_name('receiver')
                todo = [receiver] if receiver is not None else []
                while todo:
                    item = todo.pop()
                    if item.type == 'type_identifier':
                        owner = [raw[item.start_byte:item.end_byte].decode('utf-8')]
                        break
                    todo.extend(reversed(item.named_children))
            kind = 'class_definition' if is_class else 'function_definition'
            scopes.append({'kind': kind, 'name': name, 'classes': owner, 'signature': signature,
                           'start': node.start_byte, 'end': node.end_byte, 'node_type': node.type})
            points.append((node.start_point[0] + 1, 3 if is_class else 5))
            if is_class:
                child_classes = classes + [name]
            elif not inside_function:
                start = node.start_byte
                if node.parent is not None and node.parent.type in ('decorated_definition', 'export_statement'):
                    start = node.parent.start_byte
                methods.append((start, node.end_byte))
        if (is_class or is_function or node.type.endswith(('_statement', '_declaration', '_definition', '_item'))
                or node.type in ('comment', 'line_comment', 'block_comment', 'block', 'expression_statement')):
            boundaries.update((node.start_byte, node.end_byte))
        for child in reversed(node.named_children):
            stack.append((child, child_classes, inside_function or is_function))
    return scopes, sorted(methods), sorted(points), sorted(boundaries)


class ASTChunker(PythonChunker):
    """Multi-language source-preserving routes; reuse the shared prefix format."""
    def __init__(self, tokenizer, config):
        self.tokenizer, self.config = tokenizer, config

    def chunks(self, path, content, route):
        language = language_for_path(path, self.config.get('language_overrides', {}))
        raw = content.encode('utf-8')
        tree = parser_for_language(language).parse(raw)
        scopes, methods, points, boundaries = syntax_context(tree, raw, language)
        fixed_windows = []
        if route == 'blagent_ast':
            initial = ast_ranges(tree.root_node, raw, self.config.get('ast_max_chars', 1500))
        elif route == 'fixed':
            fixed_windows = fixed_ranges(content, self.tokenizer, self.config['fixed_chunk_tokens'])
            initial = [(w['start_byte'], w['end_byte'], 'fixed') for w in fixed_windows]
        elif route == 'dynamic':
            offsets, position = [0], 0
            lines = content.split('\n')
            for line in lines[:-1]:
                position += len(line.encode('utf-8')) + 1
                offsets.append(position)
            offsets.append(len(raw))
            initial = [(offsets[a], offsets[b], 'dynamic') for a, b in dynamic_ranges(
                points, len(lines), self.config['dynamic_max_lines'], self.config['dynamic_general_cost'])]
        elif route == 'method':
            initial, position = [], 0
            for start, end in methods:
                if start < position:
                    continue
                if position < start:
                    if raw[position:start].strip():
                        initial.append((position, start, 'outside_method'))
                    else:
                        start = position
                initial.append((start, end, 'method'))
                position = end
            if position < len(raw) or not initial:
                initial.append((position, len(raw), 'outside_method'))
        else:
            raise ValueError('Unknown chunk route: ' + route)
        chunks, parents = [], []
        for parent_id, (start, end, kind) in enumerate(initial):
            parents.append({'parent_id': parent_id, 'start_byte': start, 'end_byte': end, 'kind': kind})
            position = start
            while position < end or (start == end and not chunks):
                stop, split_reason = end, None
                prefix, _, _ = self.prefix(path, scopes, position, stop)
                text = raw[position:stop].decode('utf-8')
                if self.token_count(prefix + '\n\n' + text) > self.config['max_tokens']:
                    if route == 'fixed':
                        raise ValueError('Fixed code plus prefix exceeds the total token budget')
                    safe = boundaries[bisect.bisect_right(boundaries, position):bisect.bisect_right(boundaries, end)]
                    lo, hi, stop = 0, len(safe), position
                    reserve = self.config['prefix_max_tokens'] + 8
                    while lo < hi:
                        mid = (lo + hi) // 2
                        piece = raw[position:safe[mid]].decode('utf-8')
                        if len(self.tokenizer.encode(piece, add_special_tokens=False)) + reserve <= self.config['max_tokens']:
                            stop, lo = safe[mid], mid + 1
                        else:
                            hi = mid
                    split_reason = 'statement_boundary'
                    if stop == position:
                        offsets = self.tokenizer(text[:8192], add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
                        chars = offsets[min(len(offsets), self.config['max_tokens']) - 1][1]
                        piece = text[:8192][:chars]
                        while len(self.tokenizer.encode(piece, add_special_tokens=False)) + reserve > self.config['max_tokens']:
                            piece = piece[:-1]
                        if not piece:
                            raise ValueError('No room for one complete source character')
                        stop = position + len(piece.encode('utf-8'))
                        split_reason = 'oversized_statement_token_boundary'
                body = raw[position:stop].decode('utf-8')
                prefix, clipped, structure = self.prefix(path, scopes, position, stop)
                embedded = prefix + '\n\n' + body
                chunk = {'file_path': path, 'route': route, 'language': language, 'parent_id': parent_id,
                         'start_byte': position, 'end_byte': stop,
                         'start_line': raw[:position].count(b'\n') + 1,
                         'end_line': raw[:max(position, stop - 1)].count(b'\n') + 1,
                         'kind': kind, 'parse_has_error': tree.root_node.has_error,
                         'parse_status': 'recovered' if tree.root_node.has_error else 'parsed',
                         'split_reason': split_reason, 'prefix': prefix, 'prefix_truncated': clipped,
                         'structure': structure, 'code': body, 'embedding_text': embedded,
                         'token_count': self.token_count(embedded),
                         'prefix_tokens': len(self.tokenizer.encode(prefix, add_special_tokens=False)),
                         'body_tokens': len(self.tokenizer.encode(body, add_special_tokens=False))}
                if route == 'fixed':
                    chunk.update({'fixed_window': fixed_windows[parent_id], 'fixed_chunk_tokens': self.config['fixed_chunk_tokens']})
                if chunk['token_count'] > self.config['max_tokens'] or (route == 'fixed' and chunk['body_tokens'] > self.config['fixed_chunk_tokens']):
                    raise ValueError('Chunk exceeds the configured token budget')
                chunk['chunk_id'] = hashlib.sha256(json.dumps(
                    [route, path, parent_id, position, stop, embedded], ensure_ascii=False).encode()).hexdigest()
                chunks.append(chunk)
                position = stop
                if start == end:
                    break
        if ''.join(c['code'] for c in chunks) != content:
            raise ValueError('AST chunking changed or dropped source content: ' + path)
        return chunks, parents


class ASTChunkCache(ChunkCache):
    def __init__(self, project, chunker, model_fingerprint):
        self.chunker = chunker
        fields = ('max_tokens', 'prefix_max_tokens', 'dynamic_max_lines', 'dynamic_general_cost', 'fixed_chunk_tokens')
        self.identity = {'format': 'multilingual-ast-v1', 'model_fingerprint': model_fingerprint,
                         'settings': {**{key: chunker.config[key] for key in fields},
                                      'ast_max_chars': chunker.config.get('ast_max_chars', 1500),
                                      'language_overrides': chunker.config.get('language_overrides', {})},
                         'grammars': grammar_versions(),
                         'implementation': {name: sha256_file(Path(__file__).with_name(name))
                                            for name in ('chunking.py', 'ast_chunking.py', 'languages.py')}}
        self.folder = project / 'cache/chunks' / digest(self.identity)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0


class RoutingChunkCache:
    """Keep existing Python caches intact; use AST grammars for other inputs."""
    def __init__(self, project, tokenizer, config, model_fingerprint):
        self.config = config
        self.python = ChunkCache(project, PythonChunker(tokenizer, config), model_fingerprint)
        self.ast = ASTChunkCache(project, ASTChunker(tokenizer, config), model_fingerprint)

    @property
    def hits(self):
        return self.python.hits + self.ast.hits

    @property
    def misses(self):
        return self.python.misses + self.ast.misses

    def get(self, document, route):
        language = language_for_path(document['file_path'], self.config.get('language_overrides', {}))
        cache = self.python if language == 'python' and route != 'blagent_ast' else self.ast
        return cache.get(document, route)

    def index_identity(self, documents, route):
        identity = {'chunker_sha256': sha256_file(Path(__file__).with_name('chunking.py'))}
        if route == 'blagent_ast' or any(language_for_path(d['file_path'], self.config.get('language_overrides', {})) != 'python'
                                         for d in documents):
            identity['ast_pipeline_fingerprint'] = digest(self.ast.identity)
        return identity
