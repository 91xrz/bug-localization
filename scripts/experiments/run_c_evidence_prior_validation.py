"""Validate frozen packet/prior/guard components outside the historical500.

No new hyperparameter search. Every proposed combination is declared before
scoring this cohort. Primary method: combined_head10, with guarded as a separate
predeclared risk-control variant. Archived data are not a new blind benchmark.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bug_localization.evidence_packets import file_packets
from bug_localization.reranking import LocalReranker
from bug_localization.conservative_reranking import constrain_proposal
from run_conservative_reranking import FULL, read, save, sha, interval, metric, summarize

CPU = ROOT / "results/conservative_reranking/20261123_v3"
PACKETS = ROOT / "results/evidence_packets/20261123_v2"


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    pc, ec = read(CPU / "choices.json"), read(PACKETS / "choice.json")
    pool = [r for r in read(CPU / "full.json") if not r["development"]]
    by_repo = defaultdict(list)
    def key(row):
        identity = f"20261123-c-validation:{row['repository']}:{row['issue_id']}:{row['before_fix_sha']}"
        return hashlib.sha256(identity.encode()).hexdigest()
    for row in pool:
        by_repo[row["repository"]].append(row)
    selected = [r for rows in by_repo.values() for r in sorted(rows, key=key)[:15]]
    selected_ids = {r["dataset_row"] for r in selected}
    selected += sorted([r for r in pool if r["dataset_row"] not in selected_ids], key=key)[:180-len(selected)]
    selected.sort(key=lambda r: r["dataset_row"])
    if len(selected) != 180 or len({r["dataset_row"] for r in selected}) != 180:
        raise ValueError("Unexpected validation selection")
    metadata = {r["dataset_row"]: r for r in read(FULL / "FULL/samples.json")}
    labels, plan, hashes = [], [], {}
    for i, row in enumerate(selected):
        rowid = row["dataset_row"]
        record = metadata[rowid]
        if row["base"]:
            path = FULL / record["result_file"]
            digest = sha(path)
            if digest != record["result_sha256"]:
                raise ValueError("Source changed")
            case = read(path)
            packets = file_packets(case["scored_chunks"])
            hashes[str(path)] = digest
            inp = {"dataset_row": rowid, "query": case["query"], "packets": packets}
        else:
            inp = {"dataset_row": rowid, "query": "", "packets": []}
        relative = f"inputs/{rowid:04d}.json"
        save(out / relative, inp)
        plan.append({"dataset_row": rowid, "repository": row["repository"], "issue_id": row["issue_id"],
                     "before_fix_sha": row["before_fix_sha"], "worker": i % 2,
                     "input_file": relative, "input_sha256": sha(out / relative)})
        labels.append(row)
    source_protocol = read(PACKETS / "protocol.json")
    protocol = {"selection": "Outside development500; first 15 identity hashes per project, fill to 180 by hash; seed string 20261123-c-validation",
                "not_blind": "Historical reports; selected without labels, scores or method outcomes. Not a fresh independent dataset.",
                "packet_weight": ec["weight"], "prior_weight": pc["residual"]["prior_weight"],
                "guard": pc["guard"], "config": source_protocol["config"], "model": source_protocol["source_model"],
                "primary": "combined_head10", "variants": ["FULL_max", "prior_head10", "packet_head10", "combined_head10", "combined_guarded", "combined_full50", "BLAgent"],
                "primary_AP_contrasts": [["combined_head10", "FULL_max"], ["combined_head10", "BLAgent"],
                                         ["combined_guarded", "FULL_max"], ["combined_guarded", "BLAgent"]],
                "choice_sha256": {str(CPU / "choices.json"): sha(CPU / "choices.json"),
                                  str(PACKETS / "choice.json"): sha(PACKETS / "choice.json")},
                "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__),
                    ROOT / "src/bug_localization/evidence_packets.py", ROOT / "src/bug_localization/conservative_reranking.py",
                    ROOT / "src/bug_localization/reranking.py", ROOT / "scripts/experiments/run_conservative_reranking.py")},
                "sample_plan": plan, "source_sha256": hashes}
    save(out / "labels.json", labels)
    protocol["labels_sha256"] = sha(out / "labels.json")
    save(out / "protocol.json", protocol)
    print(json.dumps({"selected": len(selected), "counts": {k:sum(r['repository']==k for r in selected) for k in by_repo},
                      "packet_weight": protocol["packet_weight"], "prior_weight": protocol["prior_weight"]}), flush=True)


def load(out):
    p = read(out / "protocol.json")
    for name, digest in p["code_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Implementation changed")
    for name, digest in p["choice_sha256"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Previously frozen choice changed")
    return p


def run(out, worker, device):
    p = load(out)
    with (out / f"worker{worker}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = LocalReranker(ROOT, {**p["config"], "device": device})
        if model.identity["files_sha256"] != p["model"]:
            raise ValueError("Model changed")
        save(out / f"model{worker}.json", model.identity)
        rows = [r for r in p["sample_plan"] if r["worker"] == worker]
        for i, row in enumerate(rows, 1):
            path = out / row["input_file"]
            if sha(path) != row["input_sha256"]:
                raise ValueError("Input changed")
            result = out / "cases" / f"{row['dataset_row']:04d}.json"
            if result.exists():
                if read(result)["input_sha256"] != row["input_sha256"]:
                    raise ValueError("Invalid resume")
                continue
            inp = read(path)
            started = time.perf_counter()
            scores = model.score(inp["query"], [d["document"] for d in inp["packets"]]) if inp["packets"] else []
            save(result, {"dataset_row": row["dataset_row"], "input_sha256": row["input_sha256"],
                          "protocol_sha256": sha(out / "protocol.json"),
                          "scores": [{"file_path": d["file_path"], **s} for d,s in zip(inp["packets"],scores)],
                          "seconds": time.perf_counter()-started})
            print(f"worker{worker} {i}/{len(rows)}", flush=True)


def rank(row, scores, packet_weight, prior_weight, head=10):
    base = row["base"]
    if set(scores) != set(base):
        raise ValueError("Scored candidate pool changed")
    values = {f: (1-packet_weight)*max(row["chunks"][f]) + packet_weight*scores[f] + prior_weight*row["prior"][f] for f in base}
    ties = {p:i for i,p in enumerate(base)}
    return sorted(base[:head], key=lambda p:(-values[p],ties[p])) + base[head:]


def evaluate(out):
    p = load(out)
    if (out / "summary.json").exists():
        raise FileExistsError("Evaluation already complete")
    if sha(out / "labels.json") != p["labels_sha256"]:
        raise ValueError("Labels changed")
    rows = read(out / "labels.json")
    orders = {n: [] for n in p["variants"]}
    tokens, seconds, hashes = 0, 0., {}
    for row, meta in zip(rows, p["sample_plan"]):
        path = out / "cases" / f"{row['dataset_row']:04d}.json"
        record = read(path)
        if (record["dataset_row"] != row["dataset_row"] or record["input_sha256"] != meta["input_sha256"]
                or record["protocol_sha256"] != sha(out / "protocol.json")):
            raise ValueError("Scoring provenance mismatch")
        hashes[str(path.relative_to(out))] = sha(path)
        scores = {s["file_path"]: s["score"] for s in record["scores"]}
        combined = rank(row, scores, p["packet_weight"], p["prior_weight"])
        guarded = constrain_proposal(row["base"], combined[:10], row["chunks"], row["prior"], **p["guard"])[0]
        case_orders = {"FULL_max": row["base"], "BLAgent": row["bl"],
                       "prior_head10": rank(row, scores, 0., p["prior_weight"]),
                       "packet_head10": rank(row, scores, p["packet_weight"], 0.),
                       "combined_head10": combined, "combined_guarded": guarded,
                       "combined_full50": rank(row, scores, p["packet_weight"], p["prior_weight"], 50)}
        for name, order in case_orders.items():
            if name not in ("FULL_max", "BLAgent", "combined_full50"):
                if set(order[:10]) != set(row["base"][:10]) or order[10:] != row["base"][10:]:
                    raise ValueError("Membership invariant failed")
            orders[name].append(order)
        tokens += sum(s["input_tokens"] for s in record["scores"])
        seconds += record["seconds"]
    summary = {n:summarize(rows,o,orders["FULL_max"]) for n,o in orders.items()}
    vals = {n:[metric(o,r["gold"]) for o,r in zip(os_,rows)] for n,os_ in orders.items()}
    stats = {}
    for a,b in p["primary_AP_contrasts"]:
        stats[a+"_minus_"+b] = interval(rows,[x["average_precision_at_k"]-y["average_precision_at_k"] for x,y in zip(vals[a],vals[b])])
    previous = 0.
    for i,k in enumerate(sorted(stats, key=lambda k:stats[k]["paired_randomization_p"])):
        previous = max(previous,min(1.,(len(stats)-i)*stats[k]["paired_randomization_p"]))
        stats[k]["holm_p"] = previous
    save(out / "summary.json", summary)
    save(out / "statistics.json", stats)
    save(out / "rankings.json", {"dataset_rows":[r["dataset_row"] for r in rows],"orders":orders})
    save(out / "verification.json", {"status":"passed","reports":len(rows),"protocol_sha256":sha(out/"protocol.json"),
        "result_sha256":hashes,"input_tokens":tokens,"sum_scoring_seconds":seconds,
        "timing_scope":"Additional packet scoring only; not end-to-end","new_8B_calls":0})
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","run","evaluate"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker",type=int,choices=(0,1),default=0)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.mode=="run":run(args.output.resolve(),args.worker,args.device)
    else:globals()[args.mode](args.output.resolve())
