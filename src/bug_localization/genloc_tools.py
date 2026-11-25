"""Pinned GenLoc parsing, exploration tools and audited file-path recovery.

The author modules loaded here contain no model clients. The shared candidate
pool deliberately includes files with no functions and test files. Other parser
semantics, including same-signature method overwrites, remain the author's.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import threading


REVISION = "92e4dd4aea0b35f31b80f160e1f59e643a81a71d"
AUTHOR_HASHES = {
    "file_parser.py": "65b4d522a48a8898908f1b5433d9c7721f1f6e735bd1643d6b4cb78586d750c8",
    "file_data_processor.py": "2cf5a0bfb8ec2dc307371b779457a2848642f7604a1a6ae4fa3172b5b0b7a481",
    "post_processor.py": "5412bea9cfd64bf18eb2c3920481de19cd91beeef4589ae8c8558827b6f86550",
    "bug_report_processor.py": "57cd1bbc3d922603d4b1ca59f477388e1ad66875c74829e6f5881e5a1680edb5",
    "file_processor.py": "58125da98353c5aafed38b959d4bce12b47e8061ad0ce890a48aedf52da067ab",
    "utils.py": "cbcb3efdab6a01c90070b996bfaeff71b253064ffdaf8139fbda9004a17e3462",
    "collection_handler.py": "e3a2b928a0cb09b4d2d559d452de7987f6654f9d798f7dc2d345e794a93669ce",
}
DEFAULT_CHECKOUT = Path(__file__).resolve().parents[2] / ".runtime/external-baselines/genloc-code"


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _safe_path(value):
    return (isinstance(value, str) and bool(value) and value == value.strip()
            and not any(ord(c) < 32 for c in value) and "\\" not in value
            and ":" not in value and not value.startswith("/")
            and all(part not in ("", ".", "..") for part in value.split("/")))


class AuthorCode:
    """Validate the author checkout and load only CPU code on demand."""

    def __init__(self, checkout: Path = DEFAULT_CHECKOUT):
        self.checkout = Path(checkout).resolve()
        self.source_root = self.checkout / "source-code/GenLoc-Python"
        actual = subprocess.check_output(
            ["git", "-C", str(self.checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != REVISION:
            raise ValueError("Unexpected GenLoc author revision")
        hashes = {}
        for name, expected in AUTHOR_HASHES.items():
            path = self.source_root / name
            if not path.resolve().is_relative_to(self.checkout):
                raise ValueError("GenLoc source file escapes checkout")
            hashes[name] = _hash(path.read_bytes())
            if hashes[name] != expected:
                raise ValueError("Modified GenLoc author source: " + name)
        self.provenance = {"commit": actual, "checkout": str(self.checkout), "source_sha256": hashes}
        self._modules = {}
        self._parser_lock = threading.Lock()
        tree = ast.parse((self.source_root / "bug_report_processor.py").read_text(encoding="utf-8"))
        self.tools = ast.literal_eval(next(
            node.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "tools" for t in node.targets)
        ))
        rank = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "rank_files")
        self.system_prompt = ast.literal_eval(next(
            node.value for node in ast.walk(rank) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "system_content" for t in node.targets)
        ))
        self.response_format = ast.literal_eval(next(
            node.value for node in ast.walk(rank)
            if isinstance(node, ast.keyword) and node.arg == "response_format"
        ))
        maximum = ast.literal_eval(next(
            node.value for node in ast.walk(rank) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "max_iterations" for t in node.targets)
        ))
        # Frozen source is verified above: max_iterations - 2 forces a final
        # response at zero-based index 8, despite the prompt's tenth-round text.
        self.iteration_policy = {
            "max_iterations": maximum,
            "first_tool_choice": "required",
            "forced_final_iteration_index": maximum - 2,
            "forced_final_tool_choice": "none",
            "other_tool_choice": "auto",
            "prompt_final_iteration": 10,
        }

    def module(self, name):
        if name not in ("file_parser", "file_data_processor", "post_processor"):
            raise ValueError("Only CPU author modules may be loaded")
        if name not in self._modules:
            path = self.source_root / (name + ".py")
            if _hash(path.read_bytes()) != AUTHOR_HASHES[name + ".py"]:
                raise ValueError("GenLoc author source changed since validation")
            spec = importlib.util.spec_from_file_location("_genloc_pinned_" + name, path)
            if spec is None or spec.loader is None:
                raise ImportError("Cannot load GenLoc author module " + name)
            module = importlib.util.module_from_spec(spec)
            # Deliberately do not register generic upstream names in sys.modules.
            spec.loader.exec_module(module)
            self._modules[name] = module
        return self._modules[name]


def build_file_data(documents, *, author=None):
    """Parse the exact common-pool contents using the pinned author parser."""
    author = author or AuthorCode()
    parser = author.module("file_parser")
    result = []
    seen = set()
    with author._parser_lock:
        parser.initialize_parser()
        for document in documents:
            path, content = document.get("file_path"), document.get("content")
            if not _safe_path(path) or not path.endswith(".py") or path in seen:
                raise ValueError("Invalid or duplicate GenLoc candidate path")
            if not isinstance(content, str):
                raise ValueError("GenLoc source content must be text")
            seen.add(path)
            functions = parser.extract_functions(content)
            if not isinstance(functions, dict):
                raise ValueError("GenLoc parser returned invalid method data")
            result.append({
                "filepath": path, "filename": PurePosixPath(path).name,
                "methods": [{"signature": signature, "body": body}
                            for signature, body in functions.items()],
            })
    return result


def build_chunks(file_data, chunk_tokens=300):
    """Author body concatenation and token-budget splitting, then path prefix."""
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError("chunk_tokens must be a positive integer")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens, chunk_overlap=0,
        length_function=lambda text: len(encoding.encode(text)),
    )
    chunks = []
    for file in file_data:
        path = file["filepath"]
        if not _safe_path(path):
            raise ValueError("Invalid GenLoc chunk path")
        text = "".join("\n" + method["body"] for method in file["methods"]).strip()
        for index, code in enumerate(splitter.split_text(text)):
            chunks.append({
                "file_path": path, "code": code,
                "embedding_text": "file: " + path + "\n" + code,
                "chunk_id": _hash((path + "\0" + str(index) + "\0" + code).encode("utf-8")),
            })
    return chunks


class Toolbox:
    """Five author tools over all shared files, not only retrieved candidates."""

    def __init__(self, file_data, candidates, *, author=None):
        self.author = author or AuthorCode()
        paths = [file["filepath"] for file in file_data]
        if len(paths) != len(set(paths)) or any(not _safe_path(path) for path in paths):
            raise ValueError("Invalid GenLoc tool file pool")
        if any(path not in set(paths) for path in candidates):
            raise ValueError("GenLoc retrieval candidate is outside the common file pool")
        if len(candidates) != len(set(candidates)):
            raise ValueError("GenLoc retrieval candidates must be unique")
        cls = self.author.module("file_data_processor").FileDataProcessor
        self.processor = cls.__new__(cls)
        self.processor.file_level_data = deepcopy(file_data)
        self.processor.suspicious_files = list(candidates[:50])
        self.schemas = {tool["function"]["name"]: tool["function"]["parameters"]
                        for tool in self.author.tools}

    def execute(self, name, args):
        if name not in self.schemas:
            return {"error": "Unknown function", "function": name}
        schema = self.schemas[name]
        if (not isinstance(args, dict) or set(args) != set(schema.get("required", []))
                or any(not isinstance(value, str) for value in args.values())):
            return {"error": "Invalid tool arguments", "function": name}
        for field in ("filepath", "filename"):
            if field in args and not _safe_path(args[field]):
                return {"error": "Unsafe repository-relative path", "field": field}
        result = getattr(self.processor, name)(**args)
        return json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))


def normalize_ranking(raw_final, file_data, *, author=None):
    """Apply author path recovery, then unique-path evaluation normalization."""
    author = author or AuthorCode()
    post = author.module("post_processor")
    value = json.loads(raw_final) if isinstance(raw_final, str) else raw_final
    if (not isinstance(value, dict)
            or set(value) != {"analysis_of_the_bug_report", "ranked_list"}
            or not isinstance(value["analysis_of_the_bug_report"], str)
            or not isinstance(value["ranked_list"], list) or not value["ranked_list"]):
        raise ValueError("Invalid or empty GenLoc final response")
    for item in value["ranked_list"]:
        if (not isinstance(item, dict) or set(item) != {"file", "justification"}
                or not isinstance(item["file"], str) or not item["file"].strip()
                or not isinstance(item["justification"], str)):
            raise ValueError("Invalid GenLoc final ranked-list entry")
    known = {file["filepath"] for file in file_data}
    if any(not _safe_path(path) for path in known):
        raise ValueError("Invalid GenLoc recovery file pool")
    paths, seen_raw, seen_corrected, entries = [], set(), set(), []
    for item in value["ranked_list"]:
        raw = item["file"]
        entry = {"raw_path": raw, "justification": item["justification"]}
        if raw in seen_raw:
            entry["action"] = "removed_duplicate_raw_path"
        elif not _safe_path(raw):
            entry["action"] = "removed_unsafe_path"
        else:
            corrected = raw if raw in known else None
            if corrected is None:
                matches = [file["filepath"] for file in file_data
                           if file["filename"] == post.extract_filename(raw)]
                corrected, similarity = post.find_most_similar_file(raw, matches)
                entry["jaccard_similarity"] = similarity
            entry["corrected_path"] = corrected
            if corrected is None:
                entry["action"] = "removed_unknown_path"
            elif corrected in seen_corrected:
                entry["action"] = "removed_duplicate_corrected_path"
            else:
                entry["action"] = "kept_exact" if corrected == raw else "kept_recovered"
                seen_corrected.add(corrected)
                paths.append(corrected)
        seen_raw.add(raw)
        entries.append(entry)
    if not paths:
        raise ValueError("GenLoc final ranking contains no valid candidate file")
    return paths[:10], {
        "analysis_of_the_bug_report": value["analysis_of_the_bug_report"],
        "entries": entries, "raw_count": len(value["ranked_list"]),
        "unique_valid_count": len(paths), "truncated_paths": paths[10:],
        "normalization": "author basename/Jaccard recovery; unsafe-path rejection; stable corrected-path deduplication; top 10",
    }
