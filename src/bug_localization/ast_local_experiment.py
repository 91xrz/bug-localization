"""Frozen exploratory variants over the unchanged AST baseline."""
import hashlib
import json
import re
from pathlib import Path

from .ast_chunking import ASTChunker, syntax_context
from .chunk_cache import ChunkCache
from .dependency_context import DependencyRoutingCache
from .embeddings import digest
from .languages import FUNCTION_NODES, language_for_path, parser_for_language
from .snapshot import sha256_file

LOCAL_ROUTES = ('ast_control', 'ast_variables', 'ast_local_both', 'ast_scope', 'ast_overlap')


class LocalChunker(ASTChunker):
    def chunks(self, path, content, route):
        chunks, parents = super().chunks(path, content, 'blagent_ast')
        language = language_for_path(path, self.config.get('language_overrides', {}))
        raw = content.encode('utf-8')
        tree = parser_for_language(language).parse(raw)
        scopes, _, _, _ = syntax_context(tree, raw, language)
        nodes, stack = [], [tree.root_node]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(reversed(node.named_children))
        if route == 'ast_scope':
            # Split only when a baseline chunk spans multiple function starts.
            starts = sorted(s['start'] for s in scopes if s['kind'] == 'function_definition')
            pieces = []
            for c in chunks:
                if c['start_byte'] == c['end_byte']:
                    pieces.append(c)
                    continue
                contained = [s for s in starts if c['start_byte'] <= s < c['end_byte']]
                # A function already in progress also counts as the first scope.
                active = any(s['kind'] == 'function_definition' and s['start'] < c['start_byte'] < s['end'] for s in scopes)
                cuts = contained if active else contained[1:]
                boundaries = [c['start_byte']] + [x for x in cuts if c['start_byte'] < x < c['end_byte']] + [c['end_byte']]
                for a, b in zip(boundaries, boundaries[1:]):
                    part = dict(c)
                    prefix, clipped, structure = self.prefix(path, scopes, a, b)
                    body = raw[a:b].decode('utf-8')
                    embedded = prefix + '\n\n' + body
                    # If a changed prefix makes a split unsafe, retain the original chunk.
                    if self.token_count(embedded) > self.config['max_tokens']:
                        pieces = None
                        break
                    part.update(start_byte=a, end_byte=b, start_line=raw[:a].count(b'\n') + 1,
                                end_line=raw[:max(a, b-1)].count(b'\n') + 1, prefix=prefix,
                                prefix_truncated=clipped, structure=structure, code=body,
                                embedding_text=embedded, token_count=self.token_count(embedded),
                                prefix_tokens=len(self.tokenizer.encode(prefix, add_special_tokens=False)),
                                body_tokens=len(self.tokenizer.encode(body, add_special_tokens=False)))
                    pieces.append(part)
                if pieces is None:
                    return self._finish(chunks, parents, route)
            chunks = pieces
        elif language == 'python':
            functions = [n for n in nodes if n.type in FUNCTION_NODES[language]]
            statements = [n for n in nodes if n.type.endswith('_statement') or n.type in ('expression_statement',)]
            for c in chunks:
                start, end = c['start_byte'], c['end_byte']
                owners = [n for n in functions if n.start_byte < start < n.end_byte]
                if not owners:
                    continue
                owner = min(owners, key=lambda n: n.end_byte - n.start_byte)
                candidates = []
                if route in ('ast_control', 'ast_local_both'):
                    for n in nodes:
                        if n.type not in ('if_statement', 'elif_clause', 'for_statement', 'while_statement', 'try_statement', 'except_clause', 'with_statement') or not (owner.start_byte <= n.start_byte < start < n.end_byte):
                            continue
                        body = n.child_by_field_name('body')
                        if body is None:
                            body = next((x for x in n.named_children if x.type == 'block'), None)
                        if body is not None and body.start_byte <= start < body.end_byte:
                            candidates.append(('control', n.start_byte, body.start_byte))
                if route in ('ast_variables', 'ast_local_both'):
                    used = set(re.findall(r'\b[A-Za-z_]\w*\b', c['code']))
                    assignments = []
                    for n in nodes:
                        if n.type not in ('assignment', 'augmented_assignment') or not (owner.start_byte <= n.start_byte < n.end_byte <= start):
                            continue
                        ancestor = n.parent
                        while ancestor is not None and ancestor.type not in FUNCTION_NODES[language]:
                            ancestor = ancestor.parent
                        if ancestor != owner:
                            continue
                        left = n.child_by_field_name('left')
                        if left is None or left.type != 'identifier':
                            continue
                        name = raw[left.start_byte:left.end_byte].decode('utf-8')
                        if name in used:
                            assignments.append((name, n.start_byte, n.end_byte))
                    seen = set()
                    for name, a, b in reversed(assignments):
                        if name not in seen:
                            candidates.append(('definition', a, b))
                            seen.add(name)
                if route == 'ast_overlap':
                    previous = []
                    for n in statements:
                        if not (owner.start_byte <= n.start_byte < n.end_byte <= start):
                            continue
                        ancestor = n.parent
                        while ancestor is not None and ancestor.type not in FUNCTION_NODES[language]:
                            ancestor = ancestor.parent
                        if ancestor == owner:
                            previous.append(n)
                    if previous:
                        n = max(previous, key=lambda x: (x.end_byte, -x.start_byte))
                        candidates.append(('previous_statement', n.start_byte, n.end_byte))
                additions, sources = [], []
                for kind, a, b in candidates:
                    fragment = raw[a:b].decode('utf-8').strip()
                    if not fragment:
                        continue
                    addition = f'{kind}: {fragment}'
                    proposed = '\n'.join(additions + [addition])
                    embedded = c['prefix'] + '\n\n' + c['code'] + '\n\nLocal context:\n' + proposed
                    if (len(self.tokenizer.encode(proposed, add_special_tokens=False)) <= 64 and
                            self.token_count(embedded) <= self.config['max_tokens']):
                        additions.append(addition)
                        sources.append({'kind': kind, 'start_byte': a, 'end_byte': b})
                if additions:
                    c['embedding_text'] += '\n\nLocal context:\n' + '\n'.join(additions)
                    c['token_count'] = self.token_count(c['embedding_text'])
                    c['local_context_sources'] = sources
        return self._finish(chunks, parents, route)

    def _finish(self, chunks, parents, route):
        for c in chunks:
            c['route'] = route
            c['chunk_id'] = hashlib.sha256(json.dumps([route, c['file_path'], c['parent_id'], c['start_byte'], c['end_byte'], c['embedding_text']], ensure_ascii=False).encode()).hexdigest()
        return chunks, parents


class LocalRoutingCache:
    def __init__(self, project, tokenizer, config, model_fingerprint):
        self.baseline = DependencyRoutingCache(project, tokenizer, config, model_fingerprint)
        chunker = LocalChunker(tokenizer, config)
        self.local = ChunkCache(project, chunker, model_fingerprint)
        self.local.identity = {'format': 'ast-local-experiment-v1', 'baseline': self.baseline.index_identity([], 'blagent_ast'),
                               'settings': {key: config.get(key) for key in ('max_tokens', 'prefix_max_tokens', 'ast_max_chars', 'language_overrides', 'dynamic_max_lines', 'dynamic_general_cost', 'fixed_chunk_tokens')}, 'implementation': sha256_file(Path(__file__))}
        self.local.folder = project / 'cache/chunks' / digest(self.local.identity)
        self.local.folder.mkdir(parents=True, exist_ok=True)

    def prepare(self, documents):
        self.baseline.prepare(documents)

    @property
    def hits(self):
        return self.baseline.hits + self.local.hits

    @property
    def misses(self):
        return self.baseline.misses + self.local.misses

    def get(self, document, route):
        return self.local.get(document, route) if route in LOCAL_ROUTES else self.baseline.get(document, route)

    def index_identity(self, documents, route):
        return {'local_pipeline_fingerprint': digest(self.local.identity)} if route in LOCAL_ROUTES else self.baseline.index_identity(documents, route)
