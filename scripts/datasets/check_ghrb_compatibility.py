"""Exercise GHRB loading, pinned source, Java chunks and C packets on CPU.

This is a data/implementation smoke check, not a retrieval/AB/C accuracy run.
"""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bug_localization.ast_chunking import ASTChunker, syntax_context
from bug_localization.bm25 import collect_files, select_report, validate_config
from bug_localization.evidence_packets import file_packets
from bug_localization.languages import parser_for_language
from bug_localization.lightweight_hypothesis import definitions
from bug_localization.snapshot import get_snapshot, sha256_file
from bug_localization.vector import select_reports, validate_vector_config


def main():
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    base = json.loads((ROOT / "configs/retrieval/ghrb/bm25.json").read_bytes())
    vector = json.loads((ROOT / "configs/retrieval/ghrb/vector.json").read_bytes())
    validate_config(base)
    validate_vector_config(vector, base)
    data = load_from_disk(str(ROOT / "datasets/ghrb_formatted"))["test"]
    keys = [(r["repo_name"], r["issue_id"]) for r in data]
    assert len(keys) == len(set(keys)) == 134
    all_selected = []
    for repository in sorted(set(data["repo_name"])):
        selected, _ = select_reports(ROOT, {**vector, "repositories": [repository], "limit": 50}, base)
        all_selected.extend(selected)
    assert len(all_selected) == 134
    row, gold, query = select_report(ROOT, base)
    assert query == (row["title"] + "\n\n" + row["original_issue_body"]).strip()
    assert gold == ast.literal_eval(row["updated_files"])
    source, manifest = get_snapshot(ROOT, row["repo_name"], row["before_fix_sha"])
    docs, skipped = collect_files(source, manifest, base)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / vector["model_path"], local_files_only=True, trust_remote_code=False)
    chunker = ASTChunker(tokenizer, vector)
    counts = {"java_files": len(docs), "chunks": 0, "parse_error_files": 0,
              "files_with_java_declarations": 0, "files_with_current_python_declarations": 0}
    example = None
    for doc in docs:
        path, text = doc["file_path"], doc["content"]
        chunks, _ = chunker.chunks(path, text, "blagent_ast")
        assert "".join(c["code"] for c in chunks) == text
        assert all(c["token_count"] <= vector["max_tokens"] and c["language"] == "java" for c in chunks)
        packets = file_packets(chunks[:3])
        assert len(packets) == 1 and packets[0]["file_path"] == path
        raw = text.encode()
        tree = parser_for_language("java").parse(raw)
        scopes, _, _, _ = syntax_context(tree, raw, "java")
        java_names = sorted({s["name"] for s in scopes if s["name"]})
        python_names = sorted(definitions(text))
        counts["chunks"] += len(chunks)
        counts["parse_error_files"] += tree.root_node.has_error
        counts["files_with_java_declarations"] += bool(java_names)
        counts["files_with_current_python_declarations"] += bool(python_names)
        if path in gold and example is None:
            example = {"file_path": path, "java_names_sample": java_names[:20],
                       "current_structure_scanner_names": python_names, "packet_count": len(packets)}
    result = {"status": "passed", "kind": "CPU data and Java chunk/packet compatibility only",
              "reports_loaded": len(data), "unique_report_keys": len(set(keys)),
              "sample": {"bug_id": row["ghrb_bug_id"], "repository": row["repo_name"],
                         "commit": row["before_fix_sha"], "source": str(source.resolve())},
              **counts, "sample_gold_files": gold,
              "sample_missing_gold": sorted(set(gold) - {d["file_path"] for d in docs}),
              "structure_example": example, "skipped_files": len(skipped),
              "all_snapshots_downloaded": False, "model_inference_calls": 0,
              "full_ab_c_ready": False, "blockers": ["Java structure extraction/path matching needs a separate version",
                  "AB/C experiment runners bind historical SWE data and scores", "Remaining source snapshots and label coverage must be audited",
                  "Freeze grouped coefficient-development/evaluation protocol before scoring"],
              "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                  [Path(__file__), ROOT / "scripts/datasets/prepare_ghrb.py", ROOT / "src/bug_localization/lightweight_hypothesis.py"]}}
    out = ROOT / "datasets/GHRB_acquisition/compatibility_check.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
