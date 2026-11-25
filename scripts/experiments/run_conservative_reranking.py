"""Frozen, CPU-only tests of multi-chunk aggregation and conservative reranking.

Three separate questions: same-pool C1/C3 aggregation; residual refinement of
historical FULL using the current initial-ranking signals; guarded 8B proposals.
No inference path sees labels. Archived data are explicitly not a fresh test.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bug_localization.bm25 import ranking_metrics
from bug_localization.conservative_reranking import (
    constrain_proposal, residual_rank,
)
from bug_localization.lightweight_hypothesis import normalize

FULL = ROOT / "results/reranking_full/20261119_swe2294_gpu0"
C3 = ROOT / "results/chunk_ablation_reranking/20261112_C2_C3_500/C3"
C1 = ROOT / "results/mixed_recall_reranking/20261111_CD_500/C"
PRIOR = ROOT / "results/b_lightweight/20261122_v1"
INITIAL = ROOT / "results/baselines/20261122_ab_structure_v1"
JOINT = ROOT / "results/c3_joint500/20261121/joint10"
BL = ROOT / "results/blagent/recovery-20261118/dual-v4/full2294"
IDENTITY = ("dataset_row", "repository", "issue_id", "before_fix_sha")
METRICS = ("average_precision_at_k", "reciprocal_rank_at_k", "top_1", "top_5", "top_10", "recall_at_k")
NAMES = ("MAP@10", "MRR@10", "Top1", "Top5", "Top10", "Recall@10")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def checked(root, record, hashes):
    if record["status"] != "ok":
        return None
    p = root / record["result_file"]
    digest = sha(p)
    if record.get("result_sha256", digest) != digest:
        raise ValueError("Changed archived result: " + str(p))
    hashes[str(p)] = digest
    value = read(p)
    if "ground_truth" not in value and "gold_files" in value:
        value = {**value, "ground_truth": value["gold_files"]}
    return value


def paths(case):
    return [r if isinstance(r, str) else r["file_path"] for r in case["ranking"]] if case else []


def check_metrics(order, gold, recorded):
    computed = ranking_metrics(order, gold, 10)
    if any(abs(computed[k] - recorded[k]) > 1e-12 for k in METRICS):
        raise ValueError("Archived metrics do not reproduce")


def compact(case):
    if case is None:
        return {"base": [], "chunks": {}}
    chunks = defaultdict(list)
    ids = set()
    for c in case["scored_chunks"]:
        if c["chunk_id"] in ids:
            raise ValueError("Duplicate scored chunk")
        ids.add(c["chunk_id"])
        chunks[c["file_path"]].append(c["score"])
    order = paths(case)
    if set(order) != set(chunks) or len(order) != len(chunks):
        raise ValueError("Require the exact scored candidate pool")
    scores = {r["file_path"]: r["score"] for r in case["ranking"]}
    if any(abs(max(v) - scores[p]) > 1e-12 for p, v in chunks.items()):
        raise ValueError("Archived file score is not chunk max")
    check_metrics(order, case["ground_truth"], case["metrics"])
    return {"base": order, "chunks": dict(chunks)}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "purpose": "Historical-data exploratory validation; not a fresh blind test",
        "candidate_pool": "Historical C3/FULL unchanged; current A/B/structure scores are only priors within that pool",
        "historical_prior_missing_policy": "Historical500 files outside the current A50/B50 union receive zero structural evidence, explicitly recorded; A/B normalized scores still used. No missing prior occurs in the full2294 experiment.",
        "residual_grid": [{"temperature": t, "prior_weight": w, "head": 10}
                          for t in (0., 1., 2., 4.) for w in (0., 2., 4., 8.)],
        "aggregation_grid": [{"temperature": t, "prior_weight": 0., "head": 50} for t in (0., 1., 2., 4.)],
        "guard_grid": [{"margin": m, "prior_margin": .05, "corroboration": c, "head": 10}
                       for c in (False, True) for m in (0., 1., 2., 4., 8., None)],
        "residual_selection": "Highest development500 MAP with <=5% of original correct Top1 lost; ties prefer smaller T then prior weight",
        "aggregation_selection": "Highest C3 MAP on previous180, not largest C3-C1 gap; ties prefer smaller T",
        "guard_selection": "Highest (MAP - 0.5*lost_Top1/n) on previous180; base abstention included; ties prefer fewer losses",
        "splits": {"residual": "development500 / remaining1794", "aggregation_and_guard": "previous180 / new320"},
        "failure_policy": "Original upstream failures remain empty and count zero; optional 8B proposal failure explicitly abstains to C3. Fallback-only control is reported separately.",
        "guarantee": "Head10 methods preserve Top10 membership and tail; no guarantee of Top1 correctness or AP non-degradation",
        "no_new_model_calls": True,
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                        (Path(__file__), ROOT / "src/bug_localization/conservative_reranking.py")},
    }
    save(out / "protocol.json", protocol)
    hashes = {}
    indexes = {}
    for name, root, rel in (("full", FULL, "FULL/samples.json"), ("initial", INITIAL, "samples.json"),
                             ("bl", BL, "samples.json"), ("c3", C3, "samples.json"),
                             ("c1", C1, "samples.json"), ("joint", JOINT, "samples.json")):
        p = root / rel
        hashes[str(p)] = sha(p)
        rows = read(p)
        indexes[name] = {r["dataset_row"]: r for r in rows}
        if len(indexes[name]) != len(rows):
            raise ValueError("Duplicate sample identity")
    devrows = set(indexes["c3"])
    if len(devrows) != 500 or len(indexes["full"]) != 2294:
        raise ValueError("Unexpected cohort size")
    full_rows, historical = [], []
    for rowid, rec in indexes["full"].items():
        for method in ("initial", "bl"):
            if any(rec[k] != indexes[method][rowid][k] for k in IDENTITY):
                raise ValueError("Identity mismatch")
        case = checked(FULL, rec, hashes)
        init = checked(INITIAL, indexes["initial"][rowid], hashes)
        bl = checked(BL, indexes["bl"][rowid], hashes)
        gold = rec.get("ground_truth", indexes["initial"][rowid]["ground_truth"])
        for c in (case, init, bl):
            if c and set(c["ground_truth"]) != set(gold):
                raise ValueError("Labels differ between baselines")
            if c:
                check_metrics(paths(c), gold, c["metrics"])
        feature = compact(case)
        prior = {}
        if case:
            p = PRIOR / "inputs" / f"{rec['position']:04d}.json"
            hashes[str(p)] = sha(p)
            inputs = read(p)
            if any(inputs["sample"][k] != rec[k] for k in IDENTITY):
                raise ValueError("Prior identity mismatch")
            a, b = normalize(inputs["a"]), normalize(inputs["b"])
            for path in feature["base"]:
                if path not in inputs["anchor"]:
                    raise ValueError("Historical candidate outside A50/B50 union")
                prior[path] = .5 * a[path] + .5 * b[path] + .5 * inputs["anchor"][path]
        row = {**{k: rec[k] for k in IDENTITY}, "development": rowid in devrows,
               "gold": gold, **feature, "prior": prior, "initial": paths(init), "bl": paths(bl)}
        full_rows.append(row)
        if rowid in devrows:
            c3rec, c1rec, jointrec = (indexes[m][rowid] for m in ("c3", "c1", "joint"))
            for r in (c3rec, c1rec, jointrec):
                if any(rec[k] != r[k] for k in IDENTITY):
                    raise ValueError("Historical sample mismatch")
            c3 = checked(C3, c3rec, hashes)
            c1 = checked(C1, c1rec, hashes)
            joint = checked(JOINT, jointrec, hashes)
            if bool(c3) != bool(c1):
                raise ValueError("Unpaired selection failure")
            if c3:
                if (c3["query"] != c1["query"] or set(paths(c3)) != set(paths(c1))
                        or set(c3["ground_truth"]) != set(gold)
                        or set(c1["ground_truth"]) != set(gold)):
                    raise ValueError("Chunk ablation is not a paired input")
            if joint and (set(paths(joint)[:10]) != set(paths(c3)[:10])
                          or paths(joint)[10:] != paths(c3)[10:]):
                raise ValueError("8B proposal changed candidate membership")
            cf, af = compact(c3), compact(c1)
            # Use the same full-data prior inputs, but the historical candidate set.
            cp = {p: .5 * a[p] + .5 * b[p] + .5 * inputs["anchor"].get(p, 0.) for p in cf["base"]}
            historical.append({**{k: row[k] for k in IDENTITY}, "position": c3rec["position"],
                "gold": gold, "c3": cf, "c1": af, "prior": cp,
                "prior_anchor_unavailable": [p for p in cf["base"] if p not in inputs["anchor"]],
                "proposal": paths(joint)[:10] if joint else None, "joint": paths(joint), "bl": paths(bl)})
        if len(full_rows) % 250 == 0:
            print(f"prepared {len(full_rows)}/2294", flush=True)
    save(out / "full.json", full_rows)
    save(out / "historical500.json", sorted(historical, key=lambda r: r["position"]))
    save(out / "sources.json", hashes)
    save(out / "prepared.json", {"full": len(full_rows), "historical": len(historical),
         "sha256": {n: sha(out / n) for n in ("protocol.json", "full.json", "historical500.json", "sources.json")}})


def load(out):
    manifest = read(out / "prepared.json")
    for name, expected in manifest["sha256"].items():
        if sha(out / name) != expected:
            raise ValueError("Prepared artifact changed")
    protocol = read(out / "protocol.json")
    for name, expected in protocol["code_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen implementation changed")
    return protocol, read(out / "full.json"), read(out / "historical500.json")


def metric(order, gold):
    return ranking_metrics(order, gold, 10)


def summarize(rows, orders, baseline):
    ms = [metric(order, row["gold"]) for row, order in zip(rows, orders)]
    bs = [metric(order, row["gold"]) for row, order in zip(rows, baseline)]
    n = len(rows)
    return {"n": n, **{label: sum(m[k] for m in ms) / n for k, label in zip(METRICS, NAMES)},
            "corrected_top1": sum(m["top_1"] > b["top_1"] for m, b in zip(ms, bs)),
            "broken_top1": sum(m["top_1"] < b["top_1"] for m, b in zip(ms, bs)),
            "broken_top10": sum(m["top_10"] < b["top_10"] for m, b in zip(ms, bs)),
            "ap_wins": sum(m[METRICS[0]] > b[METRICS[0]] + 1e-12 for m, b in zip(ms, bs)),
            "ap_losses": sum(m[METRICS[0]] < b[METRICS[0]] - 1e-12 for m, b in zip(ms, bs))}


def residual(row, config, method=None):
    data = row[method] if method else row
    return residual_rank(data["base"], data["chunks"], row["prior"], **config) if data["base"] else []


def guard(row, config):
    if config == "base":
        return row["c3"]["base"]
    data = row["c3"]
    return constrain_proposal(data["base"], row["proposal"], data["chunks"], row["prior"], **config)[0]


def calibrate(out):
    if (out / "choices.json").exists():
        raise FileExistsError("Choices already frozen")
    protocol, full, old = load(out)
    dev = [r for r in full if r["development"]]
    olddev = [r for r in old if r["position"] <= 180]
    grid = []
    bases = [r["base"] for r in dev]
    correct = sum(metric(r["base"], r["gold"])["top_1"] for r in dev)
    for config in protocol["residual_grid"]:
        grid.append({"config": config, **summarize(dev, [residual(r, config) for r in dev], bases)})
    eligible = [r for r in grid if r["broken_top1"] <= math.floor(.05 * correct)]
    chosen = max(eligible, key=lambda r: (r["MAP@10"], -r["config"]["temperature"], -r["config"]["prior_weight"]))
    agg = []
    bases = [r["c3"]["base"] for r in olddev]
    for config in protocol["aggregation_grid"]:
        agg.append({"config": config, **summarize(olddev, [residual(r, config, "c3") for r in olddev], bases)})
    ac = max(agg, key=lambda r: (r["MAP@10"], -r["config"]["temperature"]))
    guards = []
    for config in ["base"] + protocol["guard_grid"]:
        item = {"config": config, **summarize(olddev, [guard(r, config) for r in olddev], bases)}
        item["utility"] = item["MAP@10"] - .5 * item["broken_top1"] / len(olddev)
        guards.append(item)
    gc = max(guards, key=lambda r: (r["utility"], -r["broken_top1"]))
    choices = {"residual": chosen["config"], "aggregation": ac["config"], "guard": gc["config"],
               "prepared_sha256": sha(out / "prepared.json"), "selection_inputs": "Only development500 and historical previous180 metrics"}
    save(out / "development_grid.json", {"residual": grid, "aggregation": agg, "guard": guards})
    choices["development_grid_sha256"] = sha(out / "development_grid.json")
    save(out / "choices.json", choices)
    print(json.dumps(choices, indent=2), flush=True)


def interval(rows, deltas, repetitions=5000):
    """Both report bootstrap and whole-project cluster bootstrap, paired."""
    import numpy as np
    d = np.array(deltas, dtype=float)
    rng = np.random.default_rng(20261123)
    boot = np.empty(repetitions)
    for start in range(0, repetitions, 100):
        size = min(100, repetitions-start)
        boot[start:start+size] = d[rng.integers(len(d), size=(size, len(d)))].mean(axis=1)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["repository"]].append(i)
    sums = np.array([d[g].sum() for g in groups.values()])
    sizes = np.array([len(g) for g in groups.values()])
    idx = rng.integers(len(groups), size=(repetitions, len(groups)))
    clusters = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    signs = 0
    for start in range(0, repetitions, 100):
        size = min(100, repetitions-start)
        flips = rng.choice([-1., 1.], size=(size, len(d)))
        signs += int((abs((flips*d).mean(axis=1)) >= abs(d.mean()) - 1e-12).sum())
    return {"delta": float(d.mean()), "ci95_report_bootstrap": np.quantile(boot, [.025, .975]).tolist(),
            "ci95_project_cluster_bootstrap": np.quantile(clusters, [.025, .975]).tolist(),
            "paired_randomization_p": (signs+1)/(repetitions+1), "projects": len(groups),
            "wins": int((d > 1e-12).sum()), "ties": int((abs(d) <= 1e-12).sum()), "losses": int((d < -1e-12).sum())}


def evaluate(out):
    if (out / "evaluation.json").exists():
        raise FileExistsError("Evaluation already exists; no retuning")
    _, full, old = load(out)
    choices = read(out / "choices.json")
    if choices["prepared_sha256"] != sha(out / "prepared.json") or choices["development_grid_sha256"] != sha(out / "development_grid.json"):
        raise ValueError("Choice provenance changed")
    summaries, statistics, details = {}, {}, {}
    primary = []
    for group, rows in (("residual_development500", [r for r in full if r["development"]]),
                        ("residual_remaining1794", [r for r in full if not r["development"]]),
                        ("historical_previous180", [r for r in old if r["position"] <= 180]),
                        ("historical_new320", [r for r in old if r["position"] > 180]),
                        ("historical_all500", old)):
        historic = group.startswith("historical")
        if historic:
            orders = {"C1_max": [r["c1"]["base"] for r in rows], "C3_max": [r["c3"]["base"] for r in rows],
                "C1_aggregate": [residual(r, choices["aggregation"], "c1") for r in rows],
                "C3_aggregate": [residual(r, choices["aggregation"], "c3") for r in rows],
                "8B_original": [r["joint"] for r in rows],
                "8B_fallback_only": [guard(r, {"margin": None}) for r in rows],
                "8B_guarded": [guard(r, choices["guard"]) for r in rows], "BLAgent": [r["bl"] for r in rows]}
            base = orders["C3_max"]
            comparisons = [("C3_aggregate", "C3_max"), ("8B_guarded", "8B_fallback_only"), ("8B_guarded", "BLAgent")]
        else:
            orders = {"FULL_max": [r["base"] for r in rows], "current_initial": [r["initial"] for r in rows],
                      "residual_head10": [residual(r, choices["residual"]) for r in rows],
                      "residual_unrestricted": [residual(r, {**choices["residual"], "head": 50}) for r in rows],
                      "BLAgent": [r["bl"] for r in rows]}
            base = orders["FULL_max"]
            comparisons = [("residual_head10", "FULL_max"), ("residual_head10", "BLAgent")]
        summaries[group] = {name: summarize(rows, order, base) for name, order in orders.items()}
        ms = {name: [metric(order, r["gold"]) for order, r in zip(paths_, rows)] for name, paths_ in orders.items()}
        for name, ranklist in orders.items():
            if name in ("residual_head10", "8B_fallback_only", "8B_guarded"):
                for order, original in zip(ranklist, base):
                    if set(order[:10]) != set(original[:10]) or order[10:] != original[10:]:
                        raise ValueError("Head membership/tail invariant violated")
        for a, b in comparisons:
            key = group + "/" + a + "_minus_" + b
            d = [x[METRICS[0]]-y[METRICS[0]] for x, y in zip(ms[a], ms[b])]
            statistics[key] = interval(rows, d)
            if group in ("residual_remaining1794", "historical_new320"):
                primary.append(key)
        if historic:
            d = [(c3a[METRICS[0]]-c1a[METRICS[0]]) - (c3[METRICS[0]]-c1[METRICS[0]])
                 for c3a, c1a, c3, c1 in zip(ms["C3_aggregate"], ms["C1_aggregate"], ms["C3_max"], ms["C1_max"])]
            key = group + "/mixed_selection_difference_in_differences"
            statistics[key] = interval(rows, d)
            if group == "historical_new320":
                primary.append(key)
        details[group] = [{**{k: r[k] for k in IDENTITY}, "gold": r["gold"],
            "methods": {name: {"ranking": orders[name][i], "metrics": ms[name][i]} for name in orders}}
            for i, r in enumerate(rows)]
    # Correct only the six predeclared validation AP contrasts, not development diagnostics.
    previous = 0.
    for i, key in enumerate(sorted(primary, key=lambda k: statistics[k]["paired_randomization_p"])):
        previous = max(previous, min(1., (len(primary)-i)*statistics[key]["paired_randomization_p"]))
        statistics[key]["holm_p_validation_contrasts"] = previous
    save(out / "evaluation.json", summaries)
    save(out / "statistics.json", statistics)
    save(out / "per_report.json", details)
    save(out / "verification.json", {"status": "passed", "choices_sha256": sha(out / "choices.json"),
         "validation_contrasts": primary, "model_calls": 0, "full_cases": len(full), "historical_cases": len(old),
         "checks": ["archived checksums and identities", "candidate membership and chunk-max scores",
                    "archived C1/C3 metrics recomputed", "same query/candidates in C1/C3 ablation",
                    "frozen choices before validation", "head membership and tail invariant", "all failures retained"],
         "limits": ["Historical data, not fresh blind validation", "Current initial rank used as a prior; its new candidate pool is not reranked",
                    "Guarded 8B result is a replay of saved responses, not a fresh model run", "No theoretical Top1 safety guarantee"]})
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "calibrate", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.mode](args.output.resolve())
