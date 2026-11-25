"""Versioned multilingual structural evidence, using the shared AST grammars.

The frozen lightweight_hypothesis implementation remains the historical v1.
Only report text and before-fix source enter these features; never labels.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import version
import hashlib
import math
from pathlib import Path, PurePosixPath
import re

from .languages import CLASS_NODES, FUNCTION_NODES, GRAMMARS, language_for_path, parser_for_language
from .lightweight_hypothesis import clean_report, normalize

ALGORITHM_VERSION = "structure_ast_v2"
NAMESPACE_NODES = {"namespace_definition", "namespace_declaration", "internal_module", "mod_item"}
NAME = re.compile(r"(?:~?[\w$]+)(?:(?:::|\.)[\w$]+)*", re.UNICODE)


@dataclass(frozen=True)
class SourceSymbols:
    path: str
    language: str
    source_sha256: str
    names: frozenset[str]
    qualified_names: frozenset[str]
    file_aliases: frozenset[str]
    parse_has_error: bool

    def to_dict(self):
        result = asdict(self)
        for key in ("names", "qualified_names", "file_aliases"):
            result[key] = sorted(result[key])
        return result


def _text(node, raw):
    return raw[node.start_byte:node.end_byte].decode("utf-8") if node is not None else ""


def _declared_name(node, raw):
    name = node.child_by_field_name("name")
    if name is None and node.type == "impl_item":
        name = node.child_by_field_name("type")
        if name is not None and name.type == "generic_type":
            name = name.child_by_field_name("type")
    if name is None:
        name = node.child_by_field_name("declarator")
        while name is not None and name.child_by_field_name("declarator") is not None:
            name = name.child_by_field_name("declarator")
    if name is None and node.type in ("arrow_function", "function_expression", "generator_function", "class"):
        parent = node.parent
        if parent is not None and parent.type in ("variable_declarator", "pair"):
            name = parent.child_by_field_name("name") or parent.child_by_field_name("key")
    value = _text(name, raw)
    return value if NAME.fullmatch(value) else ""


def _parts(name):
    return tuple(re.split(r"::|\.", name)) if name else ()


def _join(scope, name):
    parts = _parts(name)
    return parts if scope and parts[:len(scope)] == scope else scope + parts


def _valid(name):
    # No Python builtins/keywords applied to other languages. Keep case/Unicode/$.
    return len(name) >= 3 and name not in {"self", "this", "cls", "args", "kwargs"} and not (
        name.startswith("__") and name.endswith("__"))


def extract_symbols(path, content, *, language_overrides=None):
    """Extract declarations, lexical namespaces, and source-based file aliases.

    Unsupported extensions/absent grammars raise; recovery is visible in metadata.
    Calls, imports, comments, strings, and unnamed callbacks are not declarations.
    """
    if not isinstance(content, str):
        raise TypeError("Source must be decoded text")
    posix = PurePosixPath(path)
    if posix.is_absolute() or "\\" in path or any(p in ("", ".", "..") for p in path.split("/")):
        raise ValueError("Require a repository-relative POSIX file path")
    language = language_for_path(path, language_overrides)
    raw = content.encode("utf-8")
    tree = parser_for_language(language).parse(raw)
    separator = "::" if language in ("cpp", "rust") else "."
    names, qualified, aliases = set(), set(), set()
    scope = ()
    stem = str(posix.with_suffix(""))
    if "/" in stem:
        aliases.add(stem)  # extensionless repository-relative import path
    if language == "python":
        module = stem.replace("/", ".")
        if module.endswith(".__init__"):
            module = module[:-9]
        scope = _parts(module)
        if "." in module:
            aliases.add(module)
    for child in tree.root_node.named_children:
        if child.type in ("package_declaration", "package_header", "package_clause"):
            name_node = next((n for n in child.named_children if n.type in (
                "scoped_identifier", "identifier", "qualified_identifier", "package_identifier")), None)
            scope = _parts(_text(name_node, raw))
        elif child.type == "file_scoped_namespace_declaration":
            scope = _parts(_text(child.child_by_field_name("name"), raw))

    def add(name, context, local, is_type):
        if not name:
            return context
        parts = _parts(name)
        simple = parts[-1]
        full = _join(context, name)
        if _valid(simple):
            names.add(simple)
            if len(full) > 1:
                qualified.add(separator.join(full))
                # Retain class/member forms in addition to namespace-qualified forms.
                local_full = _join(local, name)
                if len(local_full) > 1:
                    qualified.add(separator.join(local_full))
                if language in ("java", "kotlin") and not is_type:
                    qualified.add(".".join(full[:-1]) + "#" + full[-1])
            # A qualified type locates a file; a namespace by itself does not.
            if is_type and len(full) > 1:
                aliases.add(separator.join(full))
        return full

    stack = [(tree.root_node, scope, ())]
    while stack:
        node, context, local = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            continue
        child_context = context
        child_local = local
        if node.type in NAMESPACE_NODES:
            name = _text(node.child_by_field_name("name"), raw)
            if NAME.fullmatch(name):
                child_context = _join(context, name)
                child_local = ()
        is_type = node.type in CLASS_NODES[language]
        if language in ("c", "cpp") and is_type and node.child_by_field_name("body") is None:
            is_type = False  # 'struct Cache *value' is a reference, not a definition.
        is_function = node.type in FUNCTION_NODES[language]
        if is_type or is_function:
            owner = context
            if language == "go" and node.type == "method_declaration":
                receiver = node.child_by_field_name("receiver")
                pending = [receiver] if receiver is not None else []
                while pending:
                    item = pending.pop()
                    if item.type == "type_identifier":
                        owner = context + (_text(item, raw),)
                        local = (_text(item, raw),)
                        break
                    pending.extend(reversed(item.named_children))
            name = _declared_name(node, raw)
            child_context = add(name, owner, local, is_type)
            child_local = _join(local, name)
        elif language in ("c", "cpp") and node.type in ("declaration", "field_declaration"):
            # Function prototypes use declarators, not function_definition nodes.
            for declarator in node.children_by_field_name("declarator"):
                if declarator.type == "function_declarator":
                    add(_declared_name(declarator, raw), context, local, False)
        for child in reversed(node.named_children):
            stack.append((child, child_context, child_local))
    return SourceSymbols(path, language, hashlib.sha256(raw).hexdigest(), frozenset(names),
                         frozenset(qualified), frozenset(aliases), tree.root_node.has_error)


def _mentioned(text, name):
    # Avoid suffix paths/names and matching a short name inside another namespace.
    # A source-location suffix :123 is allowed, but :: or a longer identifier isn't.
    boundary = r"[\w$./:#\\-]"
    return re.search(r"(?<!" + boundary + ")" + re.escape(name) +
                     r"(?:(?!" + boundary + r")|(?=:\d+(?:\D|$))|(?=[.:](?:\s|$)))", text) is not None


def anchors(report, symbols):
    """Same IDF-weighted coverage formula, with AST names and scoped matching."""
    text = clean_report(report).replace("\\", "/")
    if any(not isinstance(value, SourceSymbols) or path != value.path for path, value in symbols.items()):
        raise ValueError("Use path-keyed SourceSymbols from extract_symbols")
    vocabulary = {name for value in symbols.values() for name in value.names | value.qualified_names}
    mentioned = {name for name in sorted(vocabulary) if _mentioned(text, name)}
    matched = {path: (value.names | value.qualified_names) & mentioned for path, value in symbols.items()}
    df = Counter(name for names in matched.values() for name in names)
    weights = {name: math.log((len(symbols) + 1) / (df[name] + 1)) for name in sorted(df)}
    total = math.fsum(weights.values())
    scores, details = {}, {}
    for path, value in symbols.items():
        explicit = [name for name in sorted({path} | value.file_aliases) if _mentioned(text, name)]
        score = math.fsum(weights[name] for name in sorted(matched[path])) / total if total else 0.
        scores[path] = 1. if explicit else min(1., max(0., score))
        details[path] = {"identifiers": sorted(matched[path]), "explicit_path_or_module": bool(explicit),
                         "explicit_matches": explicit, "language": value.language,
                         "parse_has_error": value.parse_has_error, "source_sha256": value.source_sha256}
    return scores, details


def feature_identity():
    """Include implementation/grammar identities to prevent stale feature reuse."""
    files = [Path(__file__), Path(__file__).with_name("languages.py"),
             Path(__file__).with_name("lightweight_hypothesis.py")]
    return {"algorithm": ALGORITHM_VERSION, "grammar_versions": {
        module: version(module.replace("_", "-")) for module, _ in sorted(set(GRAMMARS.values()))},
        "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def score_sources(report, sources, *, language_overrides=None):
    symbols = {p: extract_symbols(p, text, language_overrides=language_overrides) for p, text in sorted(sources.items())}
    return anchors(report, symbols)


def rank_candidates(report, a, b, sources, *, alpha=.5, anchor_weight=.5, language_overrides=None):
    """Compute real structural features over A50 union B50, then keep top50.

    This is the new-data AB entry point. No historical cached E scores are used.
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1 or not math.isfinite(anchor_weight) or anchor_weight < 0:
        raise ValueError("Require alpha in [0,1] and finite nonnegative anchor_weight")
    for rows in (a, b):
        if not rows or [r["rank"] for r in rows] != list(range(1, len(rows)+1)):
            raise ValueError("Require consecutive complete rankings")
        if len({r["file_path"] for r in rows}) != len(rows) or not all(math.isfinite(r["score"]) for r in rows):
            raise ValueError("Duplicate paths or nonfinite retrieval scores")
        if any(x["score"] < y["score"] for x, y in zip(rows, rows[1:])):
            raise ValueError("Retrieval scores must be descending")
    ar = {r["file_path"]: r["rank"] for r in a}
    br = {r["file_path"]: r["rank"] for r in b}
    if ar.keys() != br.keys():
        raise ValueError("A and B must rank the same file inventory")
    pool = {r["file_path"] for rows in (a[:50], b[:50]) for r in rows}
    if set(sources) != pool:
        raise ValueError("Source inputs must cover exactly A50 union B50")
    scores, matches = score_sources(report, sources, language_overrides=language_overrides)
    an, bn = normalize(a), normalize(b)
    combined = {p: alpha*an[p] + (1-alpha)*bn[p] + anchor_weight*scores[p] for p in pool}
    ordered = sorted(pool, key=lambda p: (-combined[p], br[p], ar[p], p))[:50]
    return {"algorithm": ALGORITHM_VERSION, "alpha": alpha, "anchor_weight": anchor_weight,
            "anchor": scores, "matches": matches,
            "ranking": [{"rank": i, "file_path": p, "score": combined[p]} for i, p in enumerate(ordered, 1)]}
