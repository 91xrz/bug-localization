"""Source-preserving dynamic, method and fixed-token Python chunkers.

The DP recurrence and node costs are adapted from the supplied processor.py.
Its line windows are retained before an additional token-budget safety pass.
No comment deletion or whitespace normalization is performed: offsets refer to
the exact UTF-8 source used by the existing BM25 candidate inventory.
"""

import bisect
import hashlib
import json


def dynamic_ranges(points, line_count, max_lines=30, general_cost=50):
    """Preserve the original recurrence, tie breaking, and line-number convention."""
    if type(max_lines) is not int or max_lines < 1:
        raise ValueError('dynamic_max_lines must be a positive integer')
    costs = dict(points)
    dp = [(0, -1)] * (line_count + 1)
    for i in range(1, line_count + 1):
        best, previous = float('inf'), -1
        for j in range(max(i - max_lines, 0), i):
            cost = dp[j][0] + costs.get(i, general_cost)
            if cost < best:
                best, previous = cost, j
        dp[i] = best, previous
    ranges, current = [], line_count
    while current:
        previous = dp[current][1]
        ranges.append((previous, current))
        current = previous
    return list(reversed(ranges))


def fixed_ranges(content, tokenizer, size=400):
    """Non-overlapping token windows, backed off only for Unicode/token safety.

    Offsets slice the original source rather than decoding arbitrary token-ID
    windows, which can corrupt a Unicode character split into byte tokens.
    Re-tokenizing a substring can change its BPE count; measure the actual body.
    """
    if type(size) is not int or size < 1:
        raise ValueError('Fixed chunk size must be a positive integer')
    encoded = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded['offset_mapping']
    if not offsets:
        return [{'start_byte': 0, 'end_byte': len(content.encode('utf-8')),
                 'token_start': 0, 'token_end': 0, 'boundary_adjusted': False}]
    ranges, token_start, char_start, byte_start = [], 0, 0, 0
    while token_start < len(offsets):
        nominal_end = min(token_start + size, len(offsets))
        token_end = nominal_end
        while token_end > token_start:
            if token_end < len(offsets) and offsets[token_end][0] < offsets[token_end - 1][1]:
                token_end -= 1
                continue
            char_end = len(content) if token_end == len(offsets) else offsets[token_end][0]
            body = content[char_start:char_end]
            if char_end > char_start and len(tokenizer.encode(body, add_special_tokens=False)) <= size:
                break
            token_end -= 1
        else:
            raise ValueError('Fixed token budget cannot fit one complete source character')
        byte_end = byte_start + len(body.encode('utf-8'))
        ranges.append({'start_byte': byte_start, 'end_byte': byte_end,
                       'token_start': token_start, 'token_end': token_end,
                       'boundary_adjusted': token_end != nominal_end})
        token_start, char_start, byte_start = token_end, char_end, byte_end
    return ranges


class PythonChunker:
    def __init__(self, tokenizer, config):
        from tree_sitter import Language, Parser
        import tree_sitter_python

        self.parser = Parser(Language(tree_sitter_python.language()))
        self.tokenizer = tokenizer
        self.config = config

    def token_count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def parse(self, content):
        raw = content.encode('utf-8')
        tree = self.parser.parse(raw)
        scopes, methods, points, boundaries = [], [], [], {0, len(raw)}
        stack = [(tree.root_node, [], False)]
        while stack:
            node, classes, inside_function = stack.pop()
            new_classes = classes
            new_inside = inside_function
            if node.type in ('class_definition', 'function_definition'):
                name_node = node.child_by_field_name('name')
                name = raw[name_node.start_byte:name_node.end_byte].decode('utf-8') if name_node else ''
                body = node.child_by_field_name('body')
                signature_end = body.start_byte if body else node.end_byte
                signature = raw[node.start_byte:signature_end].decode('utf-8').rstrip()
                scope = {'kind': node.type, 'name': name, 'classes': list(classes),
                         'signature': signature, 'start': node.start_byte, 'end': node.end_byte}
                scopes.append(scope)
                # Match the supplied extractor, including its lambda substring check.
                if b'lambda' not in raw[node.start_byte:node.end_byte]:
                    points.append((node.start_point[0] + 1, 3 if node.type == 'class_definition' else 5))
                if node.type == 'class_definition':
                    new_classes = classes + [name]
                else:
                    if not inside_function:
                        start = node.parent.start_byte if node.parent.type == 'decorated_definition' else node.start_byte
                        methods.append((start, node.end_byte))
                    new_inside = True
            if (node.type.endswith('_statement') or node.type in
                    ('class_definition', 'function_definition', 'decorated_definition', 'comment')):
                boundaries.update((node.start_byte, node.end_byte))
            for child in reversed(node.named_children):
                stack.append((child, new_classes, new_inside))
        # DP dict construction depends on source traversal order for coincident nodes.
        points.sort(key=lambda p: p[0])
        methods.sort()
        return raw, scopes, methods, points, sorted(boundaries), tree.root_node.has_error

    def prefix(self, path, scopes, start, end):
        functions = [s for s in scopes if s['kind'] == 'function_definition' and s['start'] < end and s['end'] > start]
        classes = [s for s in scopes if s['kind'] == 'class_definition' and s['start'] < end and s['end'] > start]
        class_names = list(dict.fromkeys('.'.join(s['classes'] + [s['name']]) for s in classes))
        class_names += ['.'.join(s['classes']) for s in functions if s['classes']]
        class_names = list(dict.fromkeys(class_names))
        signatures = list(dict.fromkeys(' '.join(s['signature'].split()) for s in functions))
        labels = ['Path: ', 'Class: ', 'Signature: ']
        values = [path, '; '.join(class_names) or '<module>', '; '.join(signatures) or '<none>']
        full = '\n'.join(label + value for label, value in zip(labels, values))
        ids = self.tokenizer.encode(full, add_special_tokens=False)
        if len(ids) <= self.config['prefix_max_tokens']:
            return full, False, {'classes': class_names, 'signatures': signatures, 'full_prefix': full}
        # Preserve all three field labels even when a long path or multiple
        # signatures exhaust the budget. Full values remain in metadata.
        values = [value[:256] for value in values]
        clipped = '\n'.join(label + value for label, value in zip(labels, values))
        while len(self.tokenizer.encode(clipped, add_special_tokens=False)) > self.config['prefix_max_tokens']:
            longest = max(range(3), key=lambda i: len(values[i]))
            if not values[longest]:
                raise ValueError('Prefix budget is too small for structural field labels')
            values[longest] = values[longest][:-1]
            clipped = '\n'.join(label + value for label, value in zip(labels, values))
        return clipped, True, {'classes': class_names, 'signatures': signatures, 'full_prefix': full}

    def chunks(self, path, content, route):
        if route not in ('dynamic', 'method', 'fixed'):
            raise ValueError('Unknown chunk route: ' + route)
        raw, scopes, methods, points, boundaries, parse_error = self.parse(content)
        fixed_windows = []
        if route == 'dynamic':
            lines = content.split('\n')
            offsets, position = [0], 0
            for line in lines[:-1]:
                position += len(line.encode('utf-8')) + 1
                offsets.append(position)
            offsets.append(len(raw))
            initial = [(offsets[a], offsets[b], 'dynamic') for a, b in dynamic_ranges(
                points, len(lines), self.config['dynamic_max_lines'], self.config['dynamic_general_cost'])]
        elif route == 'fixed':
            fixed_windows = fixed_ranges(content, self.tokenizer, self.config['fixed_chunk_tokens'])
            initial = [(window['start_byte'], window['end_byte'], 'fixed') for window in fixed_windows]
        else:
            initial, position = [], 0
            for start, end in methods:
                if position < start:
                    if raw[position:start].strip():
                        initial.append((position, start, 'outside_method'))
                    else:
                        start = position
                initial.append((start, end, 'method'))
                position = end
            if position < len(raw) or not initial:
                if initial and not raw[position:].strip():
                    start, _, kind = initial[-1]
                    initial[-1] = (start, len(raw), kind)
                else:
                    initial.append((position, len(raw), 'outside_method'))
        chunks, original_blocks = [], []
        for parent_id, (start, end, kind) in enumerate(initial):
            prefix, clipped, structure = self.prefix(path, scopes, start, end)
            original_blocks.append({'parent_id': parent_id, 'start_byte': start, 'end_byte': end, 'kind': kind})
            position = start
            # Also represent empty source files with a prefix-only vector.
            while position < end or (start == end and not chunks):
                text = raw[position:end].decode('utf-8')
                fallback = None
                stop = end
                if self.token_count(prefix + '\n\n' + text) > self.config['max_tokens']:
                    if route == 'fixed':
                        raise ValueError('Fixed body plus structural prefix exceeds the total token budget')
                    safe = boundaries[bisect.bisect_right(boundaries, position):bisect.bisect_right(boundaries, end)]
                    # Find a fitting statement boundary. A final token count always
                    # guards against tokenization being non-monotone under concatenation.
                    lo, hi, stop = 0, len(safe), position
                    while lo < hi:
                        mid = (lo + hi) // 2
                        candidate = safe[mid]
                        piece = raw[position:candidate].decode('utf-8')
                        reserve = self.config['prefix_max_tokens'] + 8
                        if (len(self.tokenizer.encode(piece, add_special_tokens=False)) + reserve <= self.config['max_tokens']
                                and self.token_count(prefix + '\n\n' + piece) <= self.config['max_tokens']):
                            stop, lo = candidate, mid + 1
                        else:
                            hi = mid
                    fallback = 'statement_boundary'
                    if stop == position:
                        # A single oversized statement needs an explicit token fallback.
                        sample = text[:8192]
                        offsets_map = self.tokenizer(sample, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
                        chars = offsets_map[min(len(offsets_map), self.config['max_tokens']) - 1][1]
                        piece = sample[:chars]
                        while (self.token_count(prefix + '\n\n' + piece) > self.config['max_tokens'] or
                               len(self.tokenizer.encode(piece, add_special_tokens=False)) + self.config['prefix_max_tokens'] + 8 > self.config['max_tokens']):
                            piece = piece[:-1]
                        if not piece:
                            raise ValueError('Prefix leaves no room for code')
                        stop = position + len(piece.encode('utf-8'))
                        fallback = 'oversized_statement_token_boundary'
                body = raw[position:stop].decode('utf-8')
                # Recompute the true context of each child rather than inheriting
                # every method name from the larger dynamic parent.
                child_prefix, child_clipped, child_structure = self.prefix(path, scopes, position, stop)
                used_prefix, used_clipped, used_structure = child_prefix, child_clipped, child_structure
                embedded = used_prefix + '\n\n' + body
                chunk = {'file_path': path, 'route': route, 'parent_id': parent_id,
                         'start_byte': position, 'end_byte': stop,
                         'start_line': raw[:position].count(b'\n') + 1,
                         'end_line': raw[:max(position, stop - 1)].count(b'\n') + 1,
                         'kind': kind, 'parse_has_error': parse_error,
                         'split_reason': fallback, 'prefix': used_prefix, 'prefix_truncated': used_clipped,
                         'structure': used_structure, 'code': body, 'embedding_text': embedded,
                         'token_count': self.token_count(embedded),
                         'prefix_tokens': len(self.tokenizer.encode(used_prefix, add_special_tokens=False))}
                if route == 'fixed':
                    chunk.update({'fixed_window': fixed_windows[parent_id],
                                  'body_tokens': len(self.tokenizer.encode(body, add_special_tokens=False)),
                                  'fixed_chunk_tokens': self.config['fixed_chunk_tokens']})
                    if chunk['body_tokens'] > self.config['fixed_chunk_tokens']:
                        raise ValueError('Fixed chunk body exceeds its token budget')
                chunk['chunk_id'] = hashlib.sha256(json.dumps(
                    [route, path, parent_id, position, stop, embedded], ensure_ascii=False).encode()).hexdigest()
                if chunk['token_count'] > self.config['max_tokens']:
                    raise ValueError('Chunk exceeds configured token budget')
                chunks.append(chunk)
                position = stop
                if start == end:
                    break
        if ''.join(c['code'] for c in chunks) != content:
            raise ValueError('Chunking changed or dropped source content: ' + path)
        return chunks, original_blocks
