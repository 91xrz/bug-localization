"""Prepare sampled, weakly supervised repair-file ranking lists from BeetleBox train."""

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[2]
TRAIN_REPOS = ("apache/airflow", "PostHog/posthog", "odoo/odoo")
VALID_REPOS = ("localstack/localstack",)
SEED = "c-lora-repair-20261120-round1"
STOP = frozenset("the and for with that this from issue bug fix not you your what response airflow "
                 "localstack posthog odoo apache import def self none true false return else "
                 "other version happened anything deployment how github com https which selected".split())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def terms(text):
    split = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {t for t in re.findall(r"[a-z][a-z0-9]{2,}", split.lower()) if t not in STOP}


@lru_cache(maxsize=100000)
def path_terms(path):
    return terms(path)


def safe_path(path):
    return isinstance(path, str) and bool(path) and not PurePosixPath(path).is_absolute() and all(
        p not in ("", ".", "..") for p in path.split("/")) and "\\" not in path


def source_chunks(path, code, weights):
    """Select up to two non-overlapping windows by report evidence only, never by label."""
    rows = code.splitlines(keepends=True)
    pieces, buffer, size = [], [], 0
    for row in rows:
        if size + len(row) > 1500 and buffer:
            pieces.append("".join(buffer))
            buffer, size = [], 0
        while len(row) > 1500:
            pieces.append(row[:1500])
            row = row[1500:]
        buffer.append(row)
        size += len(row)
    if buffer:
        pieces.append("".join(buffer))
    scored = [(sum(weights[t] for t in sorted(terms(piece).intersection(weights))), index, piece)
              for index, piece in enumerate(pieces) if piece.strip()]
    ordered = sorted(scored, key=lambda x: (-x[0], x[1]))[:2]
    return [f"Path: {path}\n\n{piece}" for _, _, piece in ordered], ordered[0][0] if ordered else 0


def make_record(task):
    row, root_string = task
    repo, sha, query, gold = row["repo"], row["sha"], row["query"], set(row["gold"])
    base = Path(root_string) / repo / sha
    manifest = json.loads((base / "snapshot.json").read_text())
    if manifest["repository"].casefold() != repo.casefold() or manifest["commit"] != sha:
        raise ValueError("Source snapshot identity mismatch: " + row["id"])
    paths = sorted(p for p in manifest["files"] if p.endswith(".py") and safe_path(p)
                   and not {".git", ".venv", "__pycache__", "node_modules"}.intersection(p.split("/")))
    fixed = sorted(gold.intersection(paths))
    if not fixed:
        return None, "no old Python repair path"
    qt = terms(query)
    df = Counter(t for p in paths for t in path_terms(p).intersection(qt))
    weights = {t: min(6.0, 1 + math.log((len(paths) + 1) / (df[t] + 1))) for t in qt}

    def path_score(p):
        return sum(weights[t] for t in sorted(path_terms(p).intersection(qt)))

    non_gold = sorted((p for p in paths if p not in gold), key=lambda p: (-path_score(p), p))[:18]
    fixed = sorted(fixed, key=lambda p: (-path_score(p), p))
    candidates = []
    positive_count = 0
    for path in [*non_gold, *fixed]:
        if path in gold and positive_count >= 3:
            break
        sf = base / "source" / path
        if not sf.is_file() or sf.is_symlink() or sf.stat().st_size > 2097152:
            continue
        raw = sf.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["files"][path]:
            raise ValueError("Pre-fix source hash mismatch: " + str(sf))
        chunks, relevance = source_chunks(path, raw.decode("utf-8", errors="replace"), weights)
        if not chunks:
            continue
        positive_count += int(path in gold)
        candidates.append({"path": path, "chunks": chunks, "source_sha256": manifest["files"][path],
                           "mining_score": round(relevance + path_score(path), 6)})
    positives = [c for c in candidates if c["path"] in gold]
    negatives = sorted((c for c in candidates if c["path"] not in gold),
                       key=lambda c: (-c["mining_score"], c["path"]))[:4]
    if not positives or len(negatives) < 4:
        return None, "insufficient nonempty positives or negatives"
    selected = sorted(positives + negatives, key=lambda c: digest(SEED + row["id"] + c["path"]))
    return {**row, "files": selected, "available_old_python_gold": fixed,
            "python_paths_in_snapshot": len(paths)}, None


def pools(rows, test_repos, test_bodies, test_root=None):
    out, seen, bodies, fixes, excluded = defaultdict(list), set(), set(), set(), Counter()
    if {r.casefold() for r in TRAIN_REPOS + VALID_REPOS}.intersection(test_repos):
        raise ValueError("SWE test repositories overlap the BeetleBox protocol")
    for i, x in enumerate(rows):
        repo = x["repo_name"]
        if repo not in TRAIN_REPOS + VALID_REPOS or x["language"].lower() != "python":
            continue
        report_id = f'{repo}#{x["issue_id"]}'
        labels = ast.literal_eval(x["updated_files"])
        if not isinstance(labels, list) or not all(safe_path(p) for p in labels):
            raise ValueError("Unsafe label: " + report_id)
        sha, query = x["before_fix_sha"], x["body"] or ""
        if not query.strip() or not re.fullmatch(r"[a-f0-9]{40}", sha or "") or not any(p.endswith(".py") for p in labels):
            excluded["missing report, before-fix SHA or Python labels"] += 1
            continue
        body_key = digest(" ".join(query.casefold().split()))
        fix_key = (repo, x.get("after_fix_sha") or x.get("pull_url") or report_id)
        if report_id in seen or body_key in bodies or fix_key in fixes or body_key in test_bodies:
            excluded["repeated issue, normalized body or fix identity, or SWE body overlap"] += 1
            continue
        if test_root is not None and not (test_root / repo / sha / "snapshot.json").is_file():
            excluded["missing pre-fix manifest"] += 1
            continue
        seen.add(report_id)
        bodies.add(body_key)
        fixes.add(fix_key)
        out[repo].append({"id": report_id, "repo": repo, "sha": sha, "dataset_row": i,
                          "query": query, "body_sha256": digest(query), "gold": sorted(set(labels))})
    for pool in out.values():
        pool.sort(key=lambda item: digest(SEED + item["id"]))
    return out, excluded


def balanced(pool, repos):
    for j in range(max(len(pool[r]) for r in repos)):
        for r in repos:
            if j < len(pool[r]):
                yield pool[r][j]


def run(out, n_train, n_valid, workers):
    from datasets import load_from_disk
    from itertools import islice
    from datetime import datetime, timezone
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use an empty output directory")
    ds = load_from_disk(str(ROOT / "datasets/BeetleBox/train"))
    swe = load_from_disk(str(ROOT / "datasets/swe_bench_formatted/test"))
    test_repos = {v.casefold() for v in swe["repo_name"]}
    test_bodies = {digest(" ".join(v.casefold().split())) for v in swe["body"] if v}
    pool, excluded = pools(ds, test_repos, test_bodies, ROOT / "repos")
    audit = {"seed": SEED, "created_utc": datetime.now(timezone.utc).isoformat(),
             "dataset": "BeetleBox/train", "metadata_sha256": hashlib.sha256(
                 (ROOT / "datasets/BeetleBox/train/dataset_info.json").read_bytes()).hexdigest(),
             "preparation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "sampling": "repo-round-robin then seeded identity hash, skip only missing supervision",
             "preselection_counts": {r: len(rows) for r, rows in pool.items()},
             "preselection_excluded": excluded, "swe_repositories_excluded": sorted(test_repos),
             "negative_rule": "18 highest lexical path matches, then best report/code-window plus path score; exclude all changed paths",
             "positive_rule": "up to 3 existing Python changed files by report/path match, not every chunk is positive",
             "evidence_rule": "pre-fix only, source SHA verified; 2 non-overlapping lexical code windows, body<=1500 chars, with path",
             "validation_scope": "Sampled candidate ranking with gold injection, not an end-to-end benchmark",
             "splits": {}}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for split, repos, n in [("train", TRAIN_REPOS, n_train), ("validation", VALID_REPOS, n_valid)]:
            stream = balanced(pool, repos)
            selected, skipped = [], []
            while len(selected) < n:
                batch = list(islice(stream, min(32, n-len(selected))))
                if not batch:
                    raise ValueError("Not enough eligible rows: " + split)
                for source, (record, reason) in zip(batch, ex.map(make_record, (
                        (item, str(ROOT / "repos")) for item in batch))):
                    if reason:
                        skipped.append({"id": source["id"], "reason": reason})
                    else:
                        selected.append(record)
                print(f"{split}: {len(selected)}/{n}, skipped={len(skipped)}", flush=True)
            data = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected)
            (out / (split + ".jsonl")).write_text(data, encoding="utf-8")
            audit["splits"][split] = {"sha256": digest(data), "n": n, "skipped": skipped,
                                      "projects": Counter(item["repo"] for item in selected),
                                      "files": sum(len(item["files"]) for item in selected),
                                      "chunks": sum(len(f["chunks"]) for item in selected for f in item["files"]),
                                      "ids": [item["id"] for item in selected]}
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/c_lora/round1/data")
    parser.add_argument("--train", type=int, default=160)
    parser.add_argument("--validation", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.train, args.validation, args.workers) < 1:
        parser.error("Counts and workers must be positive")
    run(args.output, args.train, args.validation, args.workers)
