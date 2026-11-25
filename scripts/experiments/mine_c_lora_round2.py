"""Mine frozen-reranker mistakes in the same lexical preselection as round one."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_c_lora_training as first


def filehash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def expand(row, snapshot_root):
    base = snapshot_root / row["repo"] / row["sha"]
    manifest = json.loads((base / "snapshot.json").read_text())
    if manifest["repository"].casefold() != row["repo"].casefold() or manifest["commit"] != row["sha"]:
        raise ValueError("Wrong pre-fix snapshot: " + row["id"])
    gold = set(row["gold"])
    paths = sorted(p for p in manifest["files"] if p.endswith(".py") and first.safe_path(p)
                   and not {".git", ".venv", "__pycache__", "node_modules"}.intersection(p.split("/")))
    if len(paths) != row["python_paths_in_snapshot"]:
        raise ValueError("Round-one Python inventory changed")
    qt = first.terms(row["query"])
    df = Counter(t for p in paths for t in first.path_terms(p).intersection(qt))
    weights = {t: min(6.0, 1 + math.log((len(paths) + 1) / (df[t] + 1))) for t in qt}

    def path_score(path):
        return sum(weights[t] for t in sorted(first.path_terms(path).intersection(qt)))

    old = {f["path"]: f for f in row["files"]}
    positives = [p for p in old if p in gold]
    negatives = sorted((p for p in paths if p not in gold), key=lambda p: (-path_score(p), p))[:18]
    expanded = {}
    for path in positives + negatives:
        sf = base / "source" / path
        if not sf.is_file() or sf.is_symlink() or sf.stat().st_size > 2097152:
            continue
        raw = sf.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != manifest["files"][path]:
            raise ValueError("Source content drift: " + path)
        chunks, relevance = first.source_chunks(path, raw.decode("utf-8", errors="replace"), weights)
        if chunks:
            expanded[path] = dict(path=path, chunks=chunks, source_sha256=sha,
                                  mining_score=round(relevance + path_score(path), 6))
    if any(expanded.get(p) != evidence for p, evidence in old.items()):
        raise ValueError("Round-one evidence not reproduced: " + row["id"])
    return {**row, "files": list(expanded.values())}


def pick(row, pool, values):
    gold = set(row["gold"])
    files = {f["path"]: f for f in pool["files"]}
    if set(files) != set(values) or len(files) != len(pool["files"]) or not all(map(math.isfinite, values.values())):
        raise ValueError("Missing, duplicate, or non-finite file score")
    old = {f["path"]: f for f in row["files"]}
    positives = set(files) & gold
    old_neg = set(old) - gold
    if (not positives or positives != set(old) & gold or len(old_neg) != 4 or
            any(files.get(p) != f for p, f in old.items())):
        raise ValueError("Gold, evidence, or negative quota differs from the paired control")
    rank = sorted(set(files) - gold, key=lambda p: (-values[p], p))
    selected = set(rank[:4])
    worst, best = min(values[p] for p in positives), max(values[p] for p in positives)
    kept = sorted(positives | selected, key=lambda p: first.digest(first.SEED + row["id"] + p))
    def exceeds(paths, threshold):
        return sum(values[p] > threshold for p in paths)
    return ({**row, "files": [files[p] for p in kept]},
            {"id": row["id"], "positive": sorted(positives), "old_negatives": sorted(old_neg),
             "selected_negatives": rank[:4], "n_replaced": len(selected-old_neg),
             "new_above_any_gold": exceeds(selected, worst), "new_above_all_gold": exceeds(selected, best),
             "old_above_any_gold": exceeds(old_neg, worst), "old_above_all_gold": exceeds(old_neg, best),
             "pool": [{"path": p, "score": values[p], "source_sha256": files[p]["source_sha256"],
                       "chunk_sha256": [first.digest(c) for c in files[p]["chunks"]]} for p in sorted(files)]})


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records))


def run(args):
    from train_c_lora import LocalReranker, encode_group, score_files
    from itertools import repeat
    from multiprocessing import get_context
    train = args.control / "data/train.jsonl"
    validation = args.control / "data/validation.jsonl"
    baseline = json.loads((args.control / "training/protocol.json").read_text())
    if filehash(train) != baseline["train_sha256"] or filehash(validation) != baseline["validation_sha256"]:
        raise ValueError("Round-one data changed after training")
    records = [json.loads(s) for s in train.read_text().splitlines()]
    if len(records) != 160 or len({r["id"] for r in records}) != 160:
        raise ValueError("Expected exact round-one training cohort")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"scope": "same restricted lexical preselection, not end-to-end C3 recall",
                "selection": "base score descending, path tie-break; highest four negatives; "
                             "strict score > minimum selected gold denotes overtaking at least one positive",
                "control": str(args.control), "train_sha256": filehash(train),
                "validation_sha256": filehash(validation), "config": baseline["original"],
                "query_and_chunk_cap": 512, "device": args.device,
                "source": {p.name: filehash(p) for p in (
                    Path(__file__), Path(first.__file__), Path(__file__).with_name("train_c_lora.py"))}}
    write_json(args.output / "protocol.json", protocol)
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as ex:
        pools = list(ex.map(expand, records, repeat(first.ROOT / "repos"), chunksize=4))
    print("Original evidence reproduced for all 160 reports", flush=True)
    rr = LocalReranker(first.ROOT, {**baseline["original"], "device": args.device})
    control_id = json.loads((args.control / "training/summary.json").read_text())["base_identity"]
    if rr.identity["files_sha256"] != control_id["files_sha256"]:
        raise ValueError("Base model mismatch")
    final, audits = [], []
    for i, (row, pool) in enumerate(zip(records, pools), 1):
        group = encode_group(rr, pool)
        scored = [s.item() for _, _, s in score_files(rr, group)]
        new, stats = pick(row, pool, dict(zip(group["paths"], scored, strict=True)))
        final.append(new)
        audits.append(stats)
        if i % 20 == 0:
            print(f"Scored and selected {i}/{len(records)} reports", flush=True)
    write_jsonl(args.output / "mining.jsonl", audits)
    shutil.copyfile(validation, args.output / "validation.jsonl")
    temp = args.output / "train.jsonl.tmp"
    write_jsonl(temp, final)
    sums = {k: sum(a[k] for a in audits) for k in (
        "n_replaced", "new_above_any_gold", "new_above_all_gold", "old_above_any_gold", "old_above_all_gold")}
    audit = {"n": len(final), "selected_positives": sum(len(a["positive"]) for a in audits),
             "selected_negatives": sum(len(a["selected_negatives"]) for a in audits),
             "scored_files": sum(len(a["pool"]) for a in audits),
             "scored_chunks": sum(len(f["chunk_sha256"]) for a in audits for f in a["pool"]),
             "reports_with_substitution": sum(a["n_replaced"] > 0 for a in audits),
             "reports_with_true_overtake": sum(a["new_above_any_gold"] > 0 for a in audits),
             "reports_with_four_true_overtakes": sum(a["new_above_any_gold"] == 4 for a in audits),
             **sums, "base_identity": rr.identity, "elapsed_seconds": time.monotonic()-start,
             "train_sha256": filehash(temp), "validation_sha256": filehash(args.output / "validation.jsonl"),
             "mining_sha256": filehash(args.output / "mining.jsonl")}
    write_json(args.output / "audit.json", audit)
    temp.replace(args.output / "train.jsonl")
    print(json.dumps({k: v for k, v in audit.items() if k != "base_identity"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=first.ROOT / "results/c_lora/round1")
    parser.add_argument("--output", type=Path, default=first.ROOT / "results/c_lora/round2/data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args())
