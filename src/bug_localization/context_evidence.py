"""Read-only, budgeted same-file context selection from explicit report anchors.

The structural selector uses Python syntax and one-hop name def/use relations;
it is not a data-flow analysis and does not establish runtime causality. No labels,
patches, generated hypotheses, or model predictions are accepted by this module.
"""
import ast
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import heapq
import io
from pathlib import Path, PurePosixPath
import re
import tokenize


_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*")
_STOP = set('''a an and are as at be been but by can could did do does for from had
has have how i if in into is it its just may more most my no not of on or our out
please should so some than that the their them then there these they this those
to too use using was we were what when where which while will with would you your
after before also get gets got new issue bug error code example expected actual
result results return returns test tests file files function functions line lines
python import true false none self cls def class print'''.split())
_CONTROL = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith,
            ast.Try, ast.ExceptHandler)
_EXECUTABLE = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Raise,
               ast.Assert, ast.Expr, ast.Delete, ast.Import, ast.ImportFrom,
               ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def extract_anchors(report):
    """Return deterministic, bounded explicit symbols/literals and report words."""
    if not isinstance(report, str) or not report.strip():
        raise ValueError('A nonempty original report is required')
    values = {}

    def add(text, kind, weight):
        text = text.strip()
        if not text or len(text) > 100 or '\n' in text:
            return
        key = text.casefold()
        old = values.get(key)
        if old is None or weight > old['weight']:
            values[key] = {'text': text, 'kind': kind, 'weight': weight}

    for match in re.finditer(r'`([^`\n]+)`|"([^"\n]+)"|\x27([^\x27\n]+)\x27', report):
        quoted = next(x for x in match.groups() if x is not None)
        add(quoted, 'quoted', 4)
        for word in _WORD.findall(quoted):
            if word.casefold() not in _STOP or word in ('None', 'True', 'False'):
                add(word, 'identifier', 3)
    words = _WORD.findall(report)
    frequency = Counter(word.casefold() for word in words)
    for word in words:
        if word in ('None', 'True', 'False'):
            add(word, 'literal', 3)
        elif ('.' in word or '_' in word or word.endswith(('Error', 'Exception')) or
              (re.search('[a-z][A-Z]', word) is not None)):
            add(word, 'identifier', 3)
        elif word.casefold() not in _STOP and len(word) >= 3:
            add(word, 'word', 1)
    # Bounds are fixed without reference to code or localization outcomes.
    return sorted(values.values(), key=lambda a: (-a['weight'],
                  -frequency[a['text'].casefold()], a['text'].casefold(), a['text']))[:128]


def _matches(text, anchors):
    folded = text.casefold()
    matched = []
    for anchor in anchors:
        word = anchor['text'].casefold()
        if _WORD.fullmatch(anchor['text']):
            present = re.search(r'(?<![a-z_0-9])' + re.escape(word) +
                                r'(?![a-z_0-9])', folded) is not None
        else:
            present = word in folded
        if present:
            matched.append(anchor['text'])
    return matched


def _read_source(root, relative, expected_sha256):
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Expected a safe repository-relative POSIX file path')
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ('', '.', '..') for part in relative.split('/')):
        raise ValueError('Source path escapes or is not canonical')
    root = Path(root).resolve(strict=True)
    source = root.joinpath(*path.parts).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError('Source path escapes the supplied snapshot root')
    raw = source.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError('Pinned source checksum mismatch')
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        text = raw.decode(encoding)
    except (SyntaxError, UnicodeDecodeError, LookupError) as error:
        raise ValueError('Source encoding cannot be decoded safely') from error
    return text, checksum


def _covered_lines(lines, existing_texts):
    if not isinstance(existing_texts, (list, tuple)) or any(not isinstance(x, str) for x in existing_texts):
        raise ValueError('existing_texts must contain only original code-block strings')
    normalized = [line.strip() for line in lines]
    positions = defaultdict(list)
    for index, line in enumerate(normalized):
        if line:
            positions[line].append(index)
    covered = set()
    for text in existing_texts:
        old = [line.strip() for line in text.splitlines()]
        index = 0
        while index < len(old):
            if not old[index]:
                index += 1
                continue
            longest, starts = 0, []
            for source_index in positions.get(old[index], []):
                length = 0
                while (index + length < len(old) and source_index + length < len(normalized)
                       and old[index + length] == normalized[source_index + length]):
                    length += 1
                if length > longest:
                    longest, starts = length, [source_index]
                elif length == longest:
                    starts.append(source_index)
            if longest and len(starts) == 1:
                covered.update(n + 1 for n in range(starts[0], starts[0] + longest)
                               if normalized[n])
                index += longest
            else:
                # Do not mark every repeated else:/):/return line as covered
                # merely because that text occurs somewhere in an old excerpt.
                index += 1
    return covered


def _lexical_candidates(lines, anchors):
    weights = {a['text']: a['weight'] for a in anchors}
    candidates = []
    for i, line in enumerate(lines, 1):
        matches = _matches(line, anchors)
        if matches:
            candidates.append({'lines': set(range(max(1, i - 1), min(len(lines), i + 1) + 1)),
                               'focus': i, 'score': sum(weights[a] for a in matches),
                               'matched_anchors': matches, 'reason': 'lexical_anchor_window'})
    return candidates


def _names(node):
    loads, stores = set(), set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and item.id not in ('self', 'cls'):
            (stores if isinstance(item.ctx, ast.Store) else loads).add(item.id)
    return loads, stores


def _header_end(node):
    body = getattr(node, 'body', None)
    if body and hasattr(body[0], 'lineno'):
        return max(node.lineno, body[0].lineno - 1)
    return getattr(node, 'end_lineno', node.lineno)


def _branch_header(node, field, lines):
    """Recover real clause headers; never label else/except as the if/try body."""
    body = getattr(node, field, None)
    if not body:
        return None
    if field == 'body':
        return {'lines': set(range(node.lineno, _header_end(node) + 1)),
                'branch': 'except' if isinstance(node, ast.ExceptHandler) else 'body'}
    keyword = 'finally' if field == 'finalbody' else 'else'
    # An elif is represented as a nested If in orelse. That node carries its own
    # real elif source header; adding the previous positive if would mislead.
    first = body[0]
    if (field == 'orelse' and isinstance(first, ast.If) and
            lines[first.lineno - 1].lstrip().startswith('elif')):
        return None
    for number in range(first.lineno, node.lineno, -1):
        line = lines[number - 1].rstrip('\r\n')
        stripped = line.lstrip(' \t')
        if len(line) - len(stripped) != node.col_offset:
            continue
        if re.match(r'^' + keyword + r'\s*:', stripped):
            headers = {number}
            if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While)):
                headers.update(range(node.lineno, _header_end(node) + 1))
            return {'lines': headers, 'branch': keyword}
    # Unrecoverable source structure is unknown, not a positive condition.
    return None


def _statement_records(tree, lines):
    records = []

    def visit(node, scope, controls):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = ('function', node.lineno, node.name)
            controls = []
            args = {a.arg for a in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs)}
            if node.args.vararg:
                args.add(node.args.vararg.arg)
            if node.args.kwarg:
                args.add(node.args.kwarg.arg)
            records.append({'scope': scope, 'start': node.lineno, 'end': _header_end(node),
                            'loads': set(), 'stores': args - {'self', 'cls'},
                            'controls': [], 'kind': 'signature'})
        elif isinstance(node, ast.ClassDef):
            scope = ('class', node.lineno, node.name)
            controls = []
        if isinstance(node, _EXECUTABLE):
            is_doc = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            if not is_doc:
                # A compound statement's variables are its header only; child
                # bodies are visited separately and cannot fabricate broad edges.
                if isinstance(node, _CONTROL):
                    own = [value for key, value in ast.iter_fields(node)
                           if key not in ('body', 'orelse', 'finalbody', 'handlers')]
                    loads, stores = set(), set()
                    for item in own:
                        for child in (item if isinstance(item, list) else [item]):
                            if isinstance(child, ast.AST):
                                read, write = _names(child)
                                loads |= read
                                stores |= write
                    end = _header_end(node)
                else:
                    loads, stores = _names(node)
                    end = getattr(node, 'end_lineno', node.lineno)
                records.append({'scope': scope, 'start': node.lineno, 'end': end,
                                'loads': loads, 'stores': stores, 'controls': list(controls),
                                'kind': type(node).__name__})
        for field, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            next_controls = controls
            if isinstance(node, _CONTROL) and field in ('body', 'orelse', 'finalbody'):
                header = _branch_header(node, field, lines)
                if header is not None:
                    next_controls = controls + [header]
            for child in children:
                if isinstance(child, ast.AST):
                    visit(child, scope, next_controls)

    visit(tree, ('module', 0, ''), [])
    return records


def _structural_candidates(tree, lines, anchors):
    records = _statement_records(tree, lines)
    weights = {a['text']: a['weight'] for a in anchors}
    definitions, uses = defaultdict(set), defaultdict(set)
    for index, record in enumerate(records):
        for name in record['stores']:
            definitions[record['scope'], name].add(index)
        for name in record['loads']:
            uses[record['scope'], name].add(index)
    candidates = []
    for seed_index, seed in enumerate(records):
        if seed['kind'] == 'signature':
            continue
        text = ''.join(lines[seed['start'] - 1:seed['end']])
        matches = _matches(text, anchors)
        if not matches:
            continue
        score = sum(weights[a] for a in matches)
        candidates.append({'lines': set(range(seed['start'], seed['end'] + 1)),
                           'focus': seed['start'], 'score': score + 2,
                           'matched_anchors': matches, 'reason': 'anchor_executable_statement'})
        related = set()
        for name in seed['loads']:
            related.update(definitions[seed['scope'], name])
        for name in seed['stores']:
            related.update(uses[seed['scope'], name])
        # Original record order preserves deterministic tie-breaking exactly.
        for index in sorted(related - {seed_index}):
            other = records[index]
            defined = seed['loads'] & other['stores']
            used = seed['stores'] & other['loads']
            if not defined and not used:
                continue
            relation = 'one_hop_definition' if defined else 'one_hop_use'
            variables = sorted(defined or used)
            candidates.append({'lines': set(range(other['start'], other['end'] + 1)),
                               'focus': other['start'], 'score': score + 1,
                               'matched_anchors': matches,
                               'reason': relation + ':' + ','.join(variables),
                               'anchor_line': seed['start']})
        for control in seed['controls']:
            reason = ('enclosing_control_condition' if control['branch'] == 'body' else
                      'enclosing_control_branch:' + control['branch'])
            candidates.append({'lines': control['lines'], 'focus': min(control['lines']),
                               'score': score + 1, 'matched_anchors': matches,
                               'reason': reason, 'anchor_line': seed['start'],
                               'indivisible': True})
    return candidates


def _greedy_candidates(candidates, seen_anchors):
    """Lazy coverage ordering equivalent to repeatedly taking the current max.

    Coverage bonuses only decrease as anchors are selected. A stale priority is
    therefore an upper bound; refresh the heap leader until its score is current.
    """
    reason_rank = {reason: i for i, reason in enumerate(sorted({c['reason'] for c in candidates}))}

    def priority(index):
        candidate = candidates[index]
        first = min(candidate['lines'])
        return (-(candidate['score'] + 2 * len(set(candidate['matched_anchors']) - seen_anchors)),
                abs(first - candidate['focus']), first, -reason_rank[candidate['reason']], index)

    heap = [priority(i) for i in range(len(candidates))]
    heapq.heapify(heap)
    while heap:
        old = heapq.heappop(heap)
        index = old[-1]
        current = priority(index)
        if current != old:
            heapq.heappush(heap, current)
            continue
        yield candidates[index]


def _segments(selected, lines, metadata):
    segments = []
    for number in sorted(selected):
        if not segments or number != segments[-1]['end_line'] + 1:
            segments.append({'start_line': number, 'end_line': number})
        else:
            segments[-1]['end_line'] = number
    for segment in segments:
        numbers = range(segment['start_line'], segment['end_line'] + 1)
        text = ''.join(lines[segment['start_line'] - 1:segment['end_line']])
        segment.update(text=text, text_sha256=_sha(text),
                       reason=sorted({metadata[n]['reason'] for n in numbers}),
                       matched_anchors=sorted({a for n in numbers for a in metadata[n]['matched_anchors']}))
    return segments


def render_segments(file_path, segments):
    if not segments:
        return ''
    return 'Additional source context: ' + file_path + '\n' + '\n'.join(
        f"[source lines {s['start_line']}-{s['end_line']}]\n{s['text']}" for s in segments)


def select_context(source_root, file_path, report, existing_texts, token_len, *,
                   variant='lexical', budget_tokens=256, expected_sha256=None):
    """Select extra source fragments under one exact rendered-text token budget."""
    if variant not in ('lexical', 'structural'):
        raise ValueError('Unknown context selection variant')
    if type(budget_tokens) is not int or budget_tokens < 0:
        raise ValueError('budget_tokens must be a nonnegative integer')
    text, source_hash = _read_source(source_root, file_path, expected_sha256)
    lines = text.splitlines(keepends=True)
    anchors = extract_anchors(report)
    covered = _covered_lines(lines, existing_texts)
    effective, fallback = variant, None
    if variant == 'structural':
        try:
            tree = ast.parse(text, filename=file_path)
            candidates = _structural_candidates(tree, lines, anchors)
            if not candidates:
                effective, fallback = 'lexical', 'no_executable_anchor_match'
                candidates = _lexical_candidates(lines, anchors)
        except (SyntaxError, ValueError, RecursionError) as error:
            effective, fallback = 'lexical', 'python_ast_unavailable:' + type(error).__name__
            candidates = _lexical_candidates(lines, anchors)
    else:
        candidates = _lexical_candidates(lines, anchors)
    selected, metadata, seen_anchors = set(), {}, set()
    oversized_lines, budget_rejections = set(), 0

    @lru_cache(maxsize=512)
    def measure(rendered):
        count = token_len(rendered)
        if type(count) is not int or count < 0:
            raise ValueError('token_len must return a nonnegative integer')
        return count

    for candidate in _greedy_candidates(candidates, seen_anchors):
        additions = {n for n in candidate['lines'] - covered - selected
                     if 1 <= n <= len(lines) and lines[n - 1].strip()}
        if not additions:
            continue
        # Prefer a complete candidate, then its nearest lines when it cannot fit.
        # Every emitted piece is still an exact, independently identified slice.
        attempts = [additions]
        if not candidate.get('indivisible'):
            attempts += [{n} for n in sorted(additions,
                         key=lambda n: (abs(n - candidate['focus']), n))]
        for addition in attempts:
            trial = selected | addition
            trial_metadata = {**metadata, **{n: candidate for n in addition}}
            segments = _segments(trial, lines, trial_metadata)
            rendered = render_segments(file_path, segments)
            count = measure(rendered)
            if count <= budget_tokens:
                selected, metadata = trial, trial_metadata
                seen_anchors.update(candidate['matched_anchors'])
                break
            budget_rejections += 1
            if len(addition) == 1:
                alone = _segments(addition, lines, {n: candidate for n in addition})
                if measure(render_segments(file_path, alone)) > budget_tokens:
                    oversized_lines.update(addition)
    segments = _segments(selected, lines, metadata)
    rendered = render_segments(file_path, segments)
    count = measure(rendered) if rendered else 0
    if type(count) is not int or not 0 <= count <= budget_tokens:
        raise ValueError('Rendered context exceeds the declared budget')
    return {'file_path': file_path, 'source_sha256': source_hash,
            'requested_variant': variant, 'effective_variant': effective,
            'fallback_reason': fallback, 'anchors': anchors,
            'segments': segments, 'rendered_text': rendered, 'token_count': count,
            'budget_tokens': budget_tokens, 'original_covered_line_count': len(covered),
            'candidate_count': len(candidates),
            'oversized_lines_skipped': sorted(oversized_lines - selected),
            'budget_rejected_attempts': budget_rejections,
            'method_limitations': ('Static same-file syntax and one-hop name links; no alias, '
                                  'interprocedural, or runtime proof. Unrecoverable branch headers '
                                  'are omitted; branch header groups are never partially added. '
                                  'Ambiguous repeated original lines are not assumed covered.')}
