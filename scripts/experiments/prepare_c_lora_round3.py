"""Freeze a task-specific SWE split from real, pre-fix FULL retrieval groups."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bug_localization.bm25 import prepare_report, ranking_metrics
from bug_localization.snapshot import sha256_file

ARCHIVE = ROOT / "results/reranking_full/20261119_swe2294_gpu0"
HISTORY = ROOT / "results/chunk_ablation_reranking/20261112_C2_C3_500/C3/samples.json"
RESERVED = ROOT / "results/c_lora/round3/reserved_lite_ids.json"
TRAIN = {r:20 for r in ("django/django","sympy/sympy","matplotlib/matplotlib","sphinx-doc/sphinx","astropy/astropy")}
SEED = "c-r3-head-objective-project-holdout-v1"
IDENT = ("dataset_row", "repository", "issue_id", "before_fix_sha", "query_sha256", "labels_sha256")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def hash_text(text):
    return sha256(text.encode()).hexdigest()


def select(samples, historical, counts, bodies):
    banned_issues = {(r["repository"], r["issue_id"]) for r in historical}
    banned_bodies = {bodies[r["dataset_row"]] for r in historical}
    seen_issues, seen_bodies = set(), set()
    chosen = []
    for repo, n in sorted(counts.items()):
        rows = [r for r in samples if r["repository"] == repo]
        rows.sort(key=lambda r: hash_text(f"{SEED}|{repo}|{r['issue_id']}|{r['before_fix_sha']}"))
        nr = 0
        for r in rows:
            ix = (repo,r["issue_id"])
            body_sha = bodies[r["dataset_row"]]
            if ix in banned_issues or body_sha in banned_bodies or ix in seen_issues or body_sha in seen_bodies:
                continue
            chosen.append(r)
            seen_issues.add(ix)
            seen_bodies.add(body_sha)
            nr += 1
            if nr == n:
                break
        if nr != n:
            raise ValueError(f"Not enough unique eligible reports in {repo}: {nr}/{n}")
    return chosen


def evidence(case, training):
    ranking = case["ranking"]
    paths = [r["file_path"] for r in ranking]
    if (not paths or len(paths) > 50 or len(set(paths)) != len(paths) or
            [r["rank"] for r in ranking] != list(range(1, len(paths)+1)) or
            {r["file_path"] for r in case["candidates"]} != set(paths)):
        raise ValueError("Invalid real candidate pool")
    g = set(case["ground_truth"])
    visible = [p for p in paths if p in g]
    chosen = set(paths[:6] if training else paths)
    supplemental = None
    if training and visible and chosen.isdisjoint(g):
        supplemental = visible[0]
        chosen.add(supplemental)
    stats = {"pool_files":len(paths), "pool_visible_gold":len(visible), "gold_files":len(g),
             "top6_visible_gold":len(g.intersection(paths[:6])), "supplemental_positive":supplemental,
             "base_top1_correct":paths[0] in g, "base_metrics":ranking_metrics(paths, g, 10)}
    if stats["base_metrics"] != case["metrics"]:
        raise ValueError("Archive metrics do not reproduce")

    rows = {r["file_path"]:r for r in ranking}
    grouped = {}
    chunk_ids, archive_ids = set(), []
    for c in case["scored_chunks"]:
        p, idx = c["file_path"], c["chunk_id"]
        if p not in rows or idx in chunk_ids or not c["embedding_text"].strip() or not math.isfinite(c["score"]):
            raise ValueError("Invalid archived evidence")
        chunk_ids.add(idx)
        f = grouped.setdefault(p, {"path":p, "chunks":[], "chunk_ids":[], "chunk_scores":[],
                                   "base_score":rows[p]["score"], "base_rank":rows[p]["rank"],
                                   "semantic_rank":rows[p]["semantic_rank"]})
        f["chunks"].append(c["embedding_text"])
        f["chunk_ids"].append(idx)
        f["chunk_scores"].append(c["score"])
        if p in chosen:
            archive_ids.append(idx)
    if (set(grouped) != set(paths) or
            any(not 1 <= len(f["chunks"]) <= 3 or f["base_score"] != max(f["chunk_scores"])
                for f in grouped.values()) or
            sorted(paths, key=lambda p:(-grouped[p]["base_score"], grouped[p]["semantic_rank"])) != paths):
        raise ValueError("File-Max or historical tie order does not reproduce")
    files = [f for p,f in grouped.items() if p in chosen]
    flat = [i for f in files for i in f["chunk_ids"]]
    index = {v:i for i,v in enumerate(flat)}
    return files, [index[i] for i in archive_ids], stats


def run(out, reserved_file):
    from datasets import load_from_disk
    if out.exists():
        raise FileExistsError("Do not overwrite experiment data")
    source = ARCHIVE/"FULL/samples.json"
    samples, old = read(source), read(HISTORY)
    if len(samples) != 2294 or len(old) != 500 or read(ARCHIVE/"dual_resume/verification.json")["status"] != "passed":
        raise ValueError("Different archive or historical cohort")
    ds = load_from_disk(str(ROOT/"datasets/swe_bench_formatted/test"))
    bodies = [hash_text(" ".join((v or "").casefold().split())) for v in ds["body"]]
    reserved = set(read(reserved_file)["dataset_rows"])
    if not reserved or not reserved.issubset(range(len(ds))):
        raise ValueError("Failed to map the reserved Lite reports")
    excluded = old + [r for r in samples if r["dataset_row"] in reserved]
    banned = {bodies[r["dataset_row"]] for r in excluded}
    banned_issues = {(r["repository"], r["issue_id"]) for r in excluded}
    n_pylint = min(8, len({bodies[r["dataset_row"]] for r in samples
                          if r["repository"]=="pylint-dev/pylint" and bodies[r["dataset_row"]] not in banned
                          and (r["repository"],r["issue_id"]) not in banned_issues}))
    val_counts = {"pydata/xarray":16-n_pylint}
    if n_pylint:
        val_counts["pylint-dev/pylint"] = n_pylint
    tr, val = select(samples, excluded, TRAIN, bodies), select(samples, excluded, val_counts, bodies)
    if ({r["repository"] for r in tr} & {r["repository"] for r in val} or
            {bodies[r["dataset_row"]] for r in tr} & {bodies[r["dataset_row"]] for r in val}):
        raise ValueError("Development split overlaps training")
    out.mkdir(parents=True)
    write(out/"protocol.json", {"seed":SEED, "training_projects":TRAIN, "validation_projects":val_counts,
          "source":"SWE formatted test repurposed for project-disjoint training; not official benchmark training",
          "archive_variant":"FULL: H0 top40 plus S0 to 50, mixed S0/H0 blocks, file-Max plus RRF ties",
          "training_group":"base top 6; if none positive add first existing positive from same pool; never add gold from outside pool",
          "validation":"entire real pool including retrieval misses", "historical500_sha256":sha256_file(HISTORY),
          "lite_exclusion_sha256":sha256_file(reserved_file),"local_rows_reserved_for_lite":len(reserved),
          "source_sha256":sha256_file(source), "program_sha256":sha256_file(Path(__file__))})
    audit = {}
    listing = []
    for split, rows in (("train",tr),("validation",val)):
        summaries = []
        with (out/(split+".jsonl")).open("x",encoding="utf-8") as stream:
            for r in rows:
                raw = ds[r["dataset_row"]]
                _, gold, body = prepare_report(raw, {"repository":r["repository"],"issue_id":r["issue_id"],"query_fields":["body"]})
                if (raw["before_fix_sha"] != r["before_fix_sha"] or
                        hash_text(json.dumps({"body":raw["body"]},sort_keys=True)) != r["query_sha256"] or
                        hash_text(str(raw["updated_files"])) != r["labels_sha256"]):
                    raise ValueError("Dataset identity changed")
                info = {**{k:r[k] for k in IDENT}, "id":f"{r['repository']}#{r['issue_id']}@{r['before_fix_sha']}",
                        "repo":r["repository"], "body_sha256":hash_text(raw["body"]),
                        "normalized_body_sha256":bodies[r["dataset_row"]], "status":r["status"]}
                case_path = None
                if r["status"] == "ok":
                    case_path = ARCHIVE/r["result_file"]
                    if sha256_file(case_path) != r["result_sha256"]:
                        raise ValueError("Archived result checksum mismatch")
                    case = read(case_path)
                    if (any(case[k] != r[k] for k in IDENT) or set(gold) != set(case["ground_truth"])
                            or not body.startswith(case["query"])):
                        raise ValueError("Archived report/label identity mismatch")
                    files, order, stat = evidence(case, split=="train")
                    query = case["query"]
                else:
                    files, order, stat, query = [], [], {"pool_files":0, "pool_visible_gold":0,
                        "top6_visible_gold":0,"gold_files":len(set(gold)),"supplemental_positive":None,
                        "base_top1_correct":False,"base_metrics":ranking_metrics([],gold,10)}, ""
                pos = sum(f["path"] in gold for f in files)
                eligible = 0 < pos < len(files)
                record = {**info, "query":query, "gold":gold, "files":files, "chunk_order":order}
                if split=="validation" or eligible:
                    stream.write(json.dumps(record,ensure_ascii=False)+"\n")
                summaries.append({**stat, "id":info["id"], "eligible_pair":eligible, "status":r["status"],
                                  "files":len(files),"chunks":len(order),"pos":pos})
                listing.append({**info,"split":split,"source_file":str(case_path) if case_path else None,
                                "source_sha256":r.get("result_sha256")})
        written = [v for v in summaries if split=="validation" or v["eligible_pair"]]
        audit[split] = {"selected":len(rows), "written":len(written), "projects":dict(Counter(r["repository"] for r in rows)),
              "failed":sum(v["status"]!="ok" for v in summaries),
              "pool_misses":sum(v["pool_visible_gold"]==0 and v["status"]=="ok" for v in summaries),
              "supplemented":sum(v["supplemental_positive"] is not None for v in summaries),
              "positive_files":sum(v["pos"] for v in written),"files":sum(v["files"] for v in written),
              "chunks":sum(v["chunks"] for v in written),"base_top1":sum(v["base_top1_correct"] for v in summaries),
              "sha256":sha256_file(out/(split+".jsonl")), "cases":summaries}
    write(out/"selection.json",listing)
    write(out/"audit.json",audit)
    print(json.dumps({k:{a:b for a,b in v.items() if a!="cases"} for k,v in audit.items()},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results/c_lora/round3/data")
    parser.add_argument("--reserved", type=Path, default=RESERVED)
    args = parser.parse_args()
    run(args.output, args.reserved)
