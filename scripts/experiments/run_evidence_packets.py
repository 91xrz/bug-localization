"""Paired C1/C3 experiment: score the same three chunks jointly with 0.6B.

GPU workers only read model inputs. Metrics/labels remain in a separate artifact.
This is a file-document test, not a reproduction of trained PARADE aggregation.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bug_localization.evidence_packets import file_packets
from bug_localization.reranking import LocalReranker
from run_conservative_reranking import C1, C3, checked, compact, read, save, sha, metric, summarize, interval

PILOT = ROOT / "results/c3_improvement_pilot/20261119"


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    cohorts = {n: read(PILOT / (n + ".json"))["samples"] for n in ("development45", "confirmation90")}
    ids = [s["dataset_row"] for group in cohorts.values() for s in group]
    if len(ids) != 135 or len(set(ids)) != 135:
        raise ValueError("Expected disjoint archived 45/90 cohorts")
    hashes, labels = {}, []
    indexes = {n: {r["dataset_row"]: r for r in read(p / "samples.json")} for n, p in (("C1", C1), ("C3", C3))}
    for cohort, samples in cohorts.items():
        for sample in samples:
            rowid = sample["dataset_row"]
            lab = {"dataset_row": rowid, "repository": sample["repository"], "cohort": cohort}
            previous = None
            for name, root in (("C1", C1), ("C3", C3)):
                record = indexes[name][rowid]
                case = checked(root, record, hashes)
                if case is None:
                    raise ValueError("Archived pilot expected successful sources")
                feature = compact(case)
                packet = file_packets(case["scored_chunks"])
                if previous and (case["query"] != previous["query"] or set(feature["base"]) != set(previous["base"])
                                 or set(case["ground_truth"]) != set(previous["gold"])):
                    raise ValueError("Unpaired query/pool/labels")
                previous = {"query": case["query"], "base": feature["base"], "gold": case["ground_truth"]}
                inp = {"dataset_row": rowid, "query": case["query"], "packets": packet}
                save(out / "inputs" / name / f"{rowid:04d}.json", inp)
                lab.update(gold=case["ground_truth"])
                lab[name] = {"base": feature["base"], "max_scores": {p: max(v) for p, v in feature["chunks"].items()},
                             "input_sha256": sha(out / "inputs" / name / f"{rowid:04d}.json")}
            labels.append(lab)
    config = {**read(ROOT / "configs/retrieval/reranker.json"), "batch_size": 8, "max_tokens": 4096}
    protocol = {"config": config, "source_model": read(C3 / "effective_config.json")["model"]["files_sha256"],
                "grid": [0., .25, .5, 1.],
                "selection": "maximize C3 development45 MAP for head10 interpolation; <=1 previously correct Top1 lost; smaller weight wins ties",
                "scope": "same 0.6B, same query and 50 candidates, same selected chunks, file-level joint input; maximum token budget increased to fit all evidence without truncation",
                "not_blind": True, "sources": hashes,
                "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                    (Path(__file__), ROOT / "scripts/experiments/run_conservative_reranking.py",
                     ROOT / "src/bug_localization/evidence_packets.py", ROOT / "src/bug_localization/reranking.py")},
                "cohorts": {k: [s["dataset_row"] for s in v] for k, v in cohorts.items()}}
    save(out / "labels.json", labels)
    # Freeze the model-visible inputs independently from the evaluation labels.
    protocol["input_sha256"] = {str(p.relative_to(out)): sha(p) for p in sorted((out / "inputs").glob("*/*.json"))}
    protocol["labels_sha256"] = sha(out / "labels.json")
    save(out / "protocol.json", protocol)
    print("prepared paired 45+90 file-packet inputs", flush=True)


def load_protocol(out):
    protocol = read(out / "protocol.json")
    for name, value in protocol["code_sha256"].items():
        if sha(ROOT / name) != value:
            raise ValueError("Frozen implementation changed")
    return protocol


def run(out, method, cohort, device):
    protocol = load_protocol(out)
    if cohort == "confirmation90" and not (out / "choice.json").exists():
        raise ValueError("Freeze the development choice before confirmation inference")
    dest = out / method
    dest.mkdir(exist_ok=True)
    with (dest / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = LocalReranker(ROOT, {**protocol["config"], "device": device})
        if model.identity["files_sha256"] != protocol["source_model"]:
            raise ValueError("Model differs from archived C1/C3")
        save(dest / "model.json", model.identity)
        for i, rowid in enumerate(protocol["cohorts"][cohort], 1):
            path = out / "inputs" / method / f"{rowid:04d}.json"
            digest = sha(path)
            if digest != protocol["input_sha256"][str(path.relative_to(out))]:
                raise ValueError("Prepared input changed")
            result = dest / "cases" / f"{rowid:04d}.json"
            if result.exists():
                if read(result)["input_sha256"] != digest:
                    raise ValueError("Invalid resume")
                continue
            inp = read(path)
            started = time.perf_counter()
            scores = model.score(inp["query"], [p["document"] for p in inp["packets"]])
            record = {"dataset_row": rowid, "input_sha256": digest,
                      "scores": [{"file_path": p["file_path"], **s} for p, s in zip(inp["packets"], scores)],
                      "seconds": time.perf_counter()-started}
            save(result, record)
            print(f"{method} {cohort} {i}/{len(protocol['cohorts'][cohort])}: {record['seconds']:.2f}s", flush=True)


def ranking(row, scores, method, weight, head=10):
    base = row[method]["base"]
    if set(scores) != set(base):
        raise ValueError("Incomplete file scoring")
    values = {p: (1-weight)*row[method]["max_scores"][p] + weight*scores[p] for p in base}
    tie = {p: i for i, p in enumerate(base)}
    return sorted(base[:head], key=lambda p: (-values[p], tie[p])) + base[head:]


def analyze(out, mode):
    protocol = load_protocol(out)
    if sha(out / "labels.json") != protocol["labels_sha256"]:
        raise ValueError("Labels changed")
    cohort = "development45" if mode == "calibrate" else "confirmation90"
    rows = [r for r in read(out / "labels.json") if r["cohort"] == cohort]
    values, sources = {}, {}
    for row in rows:
        rowid = row["dataset_row"]
        values[rowid] = {}
        for method in ("C1", "C3"):
            path = out / method / "cases" / f"{rowid:04d}.json"
            record = read(path)
            if record["input_sha256"] != row[method]["input_sha256"] or record["dataset_row"] != rowid:
                raise ValueError("Model result input mismatch")
            values[rowid][method] = {r["file_path"]: r["score"] for r in record["scores"]}
            sources[str(path.relative_to(out))] = sha(path)
    if mode == "calibrate":
        if (out / "choice.json").exists():
            raise FileExistsError("Choice already frozen")
        grid = []
        for weight in protocol["grid"]:
            order = [ranking(r, values[r["dataset_row"]]["C3"], "C3", weight) for r in rows]
            grid.append({"weight": weight, **summarize(rows, order, [r["C3"]["base"] for r in rows])})
        eligible = [g for g in grid if g["broken_top1"] <= 1]
        chosen = max(eligible, key=lambda g: (g["MAP@10"], -g["weight"]))
        save(out / "choice.json", {"weight": chosen["weight"], "grid": grid, "source_sha256": sources,
             "protocol_sha256": sha(out / "protocol.json"), "note": "Frozen using development45 only"})
        weight = chosen["weight"]
    else:
        choice = read(out / "choice.json")
        if choice["protocol_sha256"] != sha(out / "protocol.json"):
            raise ValueError("Choice protocol changed")
        for p, digest in choice["source_sha256"].items():
            if sha(out / p) != digest:
                raise ValueError("Calibration result changed")
        weight = choice["weight"]
    orders = {}
    for method in ("C1", "C3"):
        orders[method + "_max"] = [r[method]["base"] for r in rows]
        for label, w, h in (("packet_full50", 1., 50), ("packet_head10", 1., 10), ("chosen_head10", weight, 10)):
            orders[method + "_" + label] = [ranking(r, values[r["dataset_row"]][method], method, w, h) for r in rows]
    report = {n: summarize(rows, o, orders["C3_max"]) for n, o in orders.items()}
    statistics = {}
    for suffix in ("max", "packet_full50", "packet_head10", "chosen_head10"):
        delta = [metric(c, r["gold"])["average_precision_at_k"] - metric(a, r["gold"])["average_precision_at_k"]
                 for r, a, c in zip(rows, orders["C1_"+suffix], orders["C3_"+suffix])]
        statistics["C3_minus_C1_"+suffix] = interval(rows, delta)
    save(out / (cohort + "_summary.json"), {"summary": report, "statistics": statistics,
         "source_sha256": sources, "choice_sha256": sha(out / "choice.json")})
    save(out / (cohort + "_rankings.json"), {"dataset_rows": [r["dataset_row"] for r in rows], "orders": orders})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "calibrate", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("C1", "C3"))
    parser.add_argument("--cohort", choices=("development45", "confirmation90"), default="development45")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.mode == "prepare":
        prepare(out)
    elif args.mode == "run":
        if args.method is None:
            parser.error("run requires --method")
        run(out, args.method, args.cohort, args.device)
    else:
        analyze(out, args.mode)
