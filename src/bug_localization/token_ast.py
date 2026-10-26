"""Experimental tokenizer-budgeted AST packing with a bounded global DP ablation.

Only source structure and the existing tokenizer are used, never bug reports or
gold labels. Baseline chunkers/cache identities live in their unchanged files.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .ast_chunking import ASTChunker, RoutingChunkCache as BaselineCache, syntax_context
from .chunk_cache import ChunkCache
from .chunking import fixed_ranges
from .embeddings import digest
from .languages import grammar_versions, language_for_path, parser_for_language
from .snapshot import sha256_file

EXPERIMENTAL_ROUTES = ('ast_token_greedy', 'ast_token_dp')
# Frozen before the exploratory 44-report comparison. Not fitted to its labels.
PACKING = {'max_atoms': 32, 'boundary_weight': 0.25, 'mixed_scope_weight': 0.5,
           'underfill_weight': 0.2, 'depth_cap': 8}


def choose_partition(atoms, measure, budget, strategy):
    """Optimize over all feasible edges of at most 32 adjacent AST atoms.

    Token counts need not be monotone: every candidate edge is checked. DP is
    global within this bounded graph, not a claim of an unrestricted optimum.
    """
    n = len(atoms)

    def candidates(start):
        for end in range(start + 1, min(n, start + PACKING['max_atoms']) + 1):
            info = measure(atoms[start][0], atoms[end - 1][1])
            if info['tokens'] <= budget:
                depth = min(atoms[end - 1][2], PACKING['depth_cap']) if end < n else 0
                mixed = max(0, len(info['structure']['signatures']) - 1)
                cost = (1 + PACKING['boundary_weight'] * depth + PACKING['mixed_scope_weight'] * mixed
                        + PACKING['underfill_weight'] * (1 - info['tokens'] / budget) ** 2)
                yield end, cost

    if strategy == 'ast_token_greedy':
        result, start = [], 0
        while start < n:
            options = list(candidates(start))
            if not options:
                raise ValueError('No feasible AST token atom')
            end, cost = options[-1]
            result.append((start, end, cost))
            start = end
        return result
    scores, previous = [float('inf')] * (n + 1), [None] * (n + 1)
    scores[0] = 0
    for start in range(n):
        if scores[start] == float('inf'):
            continue
        for end, cost in candidates(start):
            candidate = scores[start] + cost
            if candidate < scores[end]:
                scores[end], previous[end] = candidate, (start, cost)
    result, end = [], n
    while end:
        if previous[end] is None:
            raise ValueError('No feasible AST partition')
        start, cost = previous[end]
        result.append((start, end, cost))
        end = start
    return result[::-1]


class TokenASTChunker(ASTChunker):
    def chunks(self, path, content, route):
        if route not in EXPERIMENTAL_ROUTES:
            raise ValueError('Unknown experimental route: ' + route)
        raw = content.encode('utf-8')
        language = language_for_path(path, self.config.get('language_overrides', {}))
        tree = parser_for_language(language).parse(raw)
        scopes, _, _, _ = syntax_context(tree, raw, language)
        budget = self.config['max_tokens']
        prefixes = {}

        @lru_cache(maxsize=4096)
        def measure(start, end):
            # Prefix clipping can be expensive. Its output depends only on the
            # overlapping scopes, so reuse it without changing any token count.
            context = tuple(i for i, scope in enumerate(scopes)
                            if scope['start'] < end and scope['end'] > start)
            if context not in prefixes:
                prefixes[context] = self.prefix(path, [scopes[i] for i in context], start, end)
            prefix, clipped, structure = prefixes[context]
            text = prefix + '\n\n' + raw[start:end].decode('utf-8')
            return {'tokens': self.token_count(text), 'prefix': prefix, 'clipped': clipped,
                    'structure': structure}

        atoms = []

        def fallback(start, end, depth):
            text = raw[start:end].decode('utf-8')
            windows = fixed_ranges(text, self.tokenizer, budget - self.config['prefix_max_tokens'] - 8)
            for window in windows:
                a, b = start + window['start_byte'], start + window['end_byte']
                if measure(a, b)['tokens'] > budget:
                    raise ValueError('Oversized AST leaf cannot fit with the real prefix')
                atoms.append((a, b, depth, 'token_leaf_fallback'))

        def visit(node, start, end, depth):
            if measure(start, end)['tokens'] <= budget:
                atoms.append((start, end, depth, 'ast_token_unit'))
                return
            children = [c for c in node.named_children if c.end_byte > c.start_byte]
            if not children or depth >= 200:
                fallback(start, end, depth)
                return
            position = start
            for child in children:
                if child.end_byte <= position:
                    continue
                visit(child, position, child.end_byte, depth + 1)
                position = child.end_byte
            if position < end:
                if measure(position, end)['tokens'] <= budget:
                    atoms.append((position, end, depth, 'ast_trailing_source'))
                else:
                    fallback(position, end, depth)

        visit(tree.root_node, 0, len(raw), 0)
        partitions = choose_partition(atoms, measure, budget, route)
        chunks, parents = [], []
        for parent_id, (a, b, cost) in enumerate(partitions):
            start, end = atoms[a][0], atoms[b - 1][1]
            info = measure(start, end)
            body = raw[start:end].decode('utf-8')
            embedded = info['prefix'] + '\n\n' + body
            fallback_count = sum(atom[3] == 'token_leaf_fallback' for atom in atoms[a:b])
            parents.append({'parent_id': parent_id, 'start_byte': start, 'end_byte': end,
                            'kind': route, 'atom_count': b - a, 'packing_cost': cost,
                            'end_ast_depth': atoms[b - 1][2], 'token_leaf_atoms': fallback_count})
            chunk = {'file_path': path, 'route': route, 'language': language, 'parent_id': parent_id,
                     'start_byte': start, 'end_byte': end, 'start_line': raw[:start].count(b'\n') + 1,
                     'end_line': raw[:max(start, end - 1)].count(b'\n') + 1,
                     'kind': route, 'parse_has_error': tree.root_node.has_error,
                     'parse_status': 'recovered' if tree.root_node.has_error else 'parsed',
                     'split_reason': 'oversized_ast_leaf_token_boundary' if fallback_count else None,
                     'prefix': info['prefix'], 'prefix_truncated': info['clipped'], 'structure': info['structure'],
                     'code': body, 'embedding_text': embedded, 'token_count': info['tokens'],
                     'prefix_tokens': len(self.tokenizer.encode(info['prefix'], add_special_tokens=False)),
                     'body_tokens': len(self.tokenizer.encode(body, add_special_tokens=False))}
            chunk['chunk_id'] = hashlib.sha256(json.dumps(
                [route, path, parent_id, start, end, embedded], ensure_ascii=False).encode()).hexdigest()
            chunks.append(chunk)
        if ''.join(c['code'] for c in chunks) != content:
            raise ValueError('Token AST partition failed source reconstruction')
        return chunks, parents


class TokenASTCache(ChunkCache):
    def __init__(self, project, tokenizer, config, model_fingerprint):
        self.chunker = TokenASTChunker(tokenizer, config)
        self.identity = {'format': 'ast-token-dp-v1', 'model_fingerprint': model_fingerprint,
                         'settings': {key: config[key] for key in ('max_tokens', 'prefix_max_tokens')},
                         'language_overrides': config.get('language_overrides', {}), 'packing': PACKING,
                         'grammars': grammar_versions(),
                         'implementation': {name: sha256_file(Path(__file__).with_name(name))
                                            for name in ('token_ast.py', 'chunking.py', 'ast_chunking.py', 'languages.py')}}
        self.folder = project / 'cache/chunks' / digest(self.identity)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0


class ExperimentalChunkCache:
    """Keep the four baseline pipelines and cache/index identities unchanged."""
    def __init__(self, project, tokenizer, config, model_fingerprint):
        self.baseline = BaselineCache(project, tokenizer, config, model_fingerprint)
        self.experimental = TokenASTCache(project, tokenizer, config, model_fingerprint)

    @property
    def hits(self):
        return self.baseline.hits + self.experimental.hits

    @property
    def misses(self):
        return self.baseline.misses + self.experimental.misses

    def get(self, document, route):
        cache = self.experimental if route in EXPERIMENTAL_ROUTES else self.baseline
        return cache.get(document, route)

    def index_identity(self, documents, route):
        if route in EXPERIMENTAL_ROUTES:
            return {'experimental_pipeline_fingerprint': digest(self.experimental.identity)}
        return self.baseline.index_identity(documents, route)
