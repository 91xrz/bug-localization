"""Independent completed-run audit and post-hoc prior-weight sensitivity.

No coefficients are selected here. The original frozen choices stay immutable.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/experiments")]
from run_multidata_coefficients import load
from bug_localization.paired_statistics import inference

KEYS = ("MAP@10", "MRR@10", "Top1", "Top5", "Top10", "Recall@10", "Top50", "Recall@50")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def measure(order, gold):
    if len(set(order)) != len(order):
        raise ValueError("Duplicate file in ranking")
    gold = set(gold)
    positions = [i for i, path in enumerate(order[:10], 1) if path in gold]
    hits50 = len(gold & set(order[:50]))
    return dict(zip(KEYS, [sum(j/i for j, i in enumerate(positions, 1))/len(gold),
        1/positions[0] if positions else 0., float(1 in positions),
        float(any(i <= 5 for i in positions)), float(bool(positions)), len(positions)/len(gold),
        float(hits50 > 0), hits50/len(gold)]))


def reconstruct(item, scores, coeff):
    if item["status"] != "ok":
        return [], []
    alpha, gamma, beta, lam = coeff
    rank_a = {r["file_path"]: r["rank"] for r in item["a"]}
    rank_b = {r["file_path"]: r["rank"] for r in item["b"]}
    norm = []
    for rows in (item["a"], item["b"]):
        high, low = rows[0]["score"], rows[min(49, len(rows)-1)]["score"]
        norm.append({r["file_path"]: 0. if high == low else max(0., min(1., (r["score"]-low)/(high-low))) for r in rows})
    prior = {p: alpha*norm[0][p] + (1-alpha)*norm[1][p] + gamma*value for p, value in item["anchor"].items()}
    ab = sorted(prior, key=lambda p: (-prior[p], rank_b[p], rank_a[p], p))[:50]
    base = sorted(ab, key=lambda p: (-scores["max"][p], ab.index(p)))
    combined = {p: (1-beta)*scores["max"][p] + beta*scores["joint"][p] + lam*prior[p] for p in base[:10]}
    head = sorted(base[:10], key=lambda p: (-combined[p], base.index(p)))
    return head + base[10:], ab


def summary(values):
    return {"n": len(values), **{k: math.fsum(v[k] for v in values)/len(values) for k in KEYS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol, plan = load(run)
    analysis = run/"analysis"
    frozen = read(analysis/"frozen_choice.json")
    verification = read(analysis/"verification.json")
    if sha(analysis/"frozen_choice.json") != verification["frozen_choice_sha256"]:
        raise ValueError("Frozen choice changed")
    old = read(analysis/"summary.json")
    saved_rows = read(analysis/"validation_per_report.json")
    sources = {str(p): sha(p) for p in [run/"protocol.json", run/"selection.json", *analysis.glob("*.json"), *analysis.glob("*.csv")]}
    save(out/"protocol.json", {"purpose": "Audit and exploratory lambda-only sensitivity; no new tuning",
        "fixed_alpha_gamma_beta": [.5, .5, .25], "lambdas": protocol["prior_weights"],
        "primary_diagnostic": "lambda4 minus lambda0 on held-out projects; both datasets",
        "scope": "Added after observing original joint-selection results; not a preregistered confirmatory test",
        "model_calls": 0, "sources_sha256": sources, "implementation_sha256": sha(Path(__file__))})
    checks = Counter()
    all_summaries, contrasts, all_per_report = {}, {}, {}
    for dataset, rows in plan.items():
        expected = {r["position"]: r for r in saved_rows[dataset]}
        groups = {phase: {str(lam): [] for lam in protocol["prior_weights"]} for phase in ("development", "validation")}
        methods = defaultdict(list)
        ab_methods = defaultdict(list)
        failures = Counter()
        details = []
        for row in rows:
            pos = row["position"]
            ip, sp = (run/dataset/part/f"{pos:04d}.json" for part in ("inputs", "scores"))
            for path in (ip, sp):
                if sha(path) != path.with_suffix(".sha256").read_text().strip():
                    raise ValueError("Scoring artifact checksum changed")
                checks["checkpoint_hashes"] += 1
            item, scores = read(ip), read(sp)
            if item["status"] != scores["status"]:
                raise ValueError("Failure status mismatch")
            if item["status"] == "ok":
                pool = {r["file_path"] for records in (item["a"][:50], item["b"][:50]) for r in records}
                if any(set(obj) != pool for obj in (item["anchor"], item["ids"], scores["max"], scores["joint"])):
                    raise ValueError("Incomplete A50/B50 union scores")
                if set(scores["chunk_scores"]) != {d["chunk_id"] for d in item["documents"]}:
                    raise ValueError("Incomplete chunk scores")
                if any(max(scores["chunk_scores"][c] for c in ids) != scores["max"][p] for p, ids in item["ids"].items()):
                    raise ValueError("File Max does not match chunk scores")
                checks["successful_complete_score_inventories"] += 1
            else:
                failures[row["partition"]] += 1
            if row["partition"] == "validation":
                saved = expected[pos]
                if saved["gold"] != row["gold"] or any(saved[k] != row[k] for k in ("repository", "dataset_row", "issue_id", "before_fix_sha")):
                    raise ValueError("Validation identity/label changed")
                for name, data in saved["methods"].items():
                    coeff = frozen["default"] if name == "default" else frozen["shared"] if name == "shared" else frozen["per_dataset"][dataset][name]
                    if data["coefficients"] != coeff:
                        raise ValueError("Method differs from frozen choice")
                    order, ab = reconstruct(item, scores, coeff)
                    values = measure(order, row["gold"])
                    if order != data["ranking"] or any(abs(values[k]-data["metrics"][k]) > 1e-12 for k in KEYS):
                        raise ValueError("Independent ranking/metrics differ")
                    methods[name].append(values)
                    if name in ("default", "ab_only"):
                        ab_methods[name].append(measure(ab, row["gold"]))
                    checks["independent_validation_method_rankings"] += 1
            entry = {k: row[k] for k in ("position", "repository", "issue_id", "partition")}
            entry.update(status=item["status"], metrics={}, same_top1_as_lambda0={})
            first, reference_members = None, None
            for lam in protocol["prior_weights"]:
                order, _ = reconstruct(item, scores, [.5, .5, .25, lam])
                values = measure(order, row["gold"])
                if first is None:
                    first = order
                    reference_members = set(order[:10])
                if set(order[:10]) != reference_members or order[10:] != first[10:]:
                    raise ValueError("Lambda-only variation changed head membership or tail")
                groups[row["partition"]][str(lam)].append(values)
                entry["metrics"][str(lam)] = values
                entry["same_top1_as_lambda0"][str(lam)] = order[:1] == first[:1]
                checks["lambda_head_tail_invariants"] += 1
            details.append(entry)
            if pos % 100 == 0:
                print(dataset, pos, len(rows), flush=True)
        for name, values in methods.items():
            actual = summary(values)
            if any(abs(actual[k]-old[dataset][name][k]) > 1e-12 for k in KEYS):
                raise ValueError("Published method summary differs")
        for name, values in ab_methods.items():
            if any(abs(summary(values)[k]-old[dataset]["AB_stage"][name][k]) > 1e-12 for k in KEYS):
                raise ValueError("Published AB summary differs")
        all_summaries[dataset] = {phase: {key: summary(v) for key, v in values.items()} for phase, values in groups.items()}
        validation = [r for r in details if r["partition"] == "validation"]
        delta = [r["metrics"]["4"]["MAP@10"]-r["metrics"]["0"]["MAP@10"] for r in validation]
        stats = inference(delta, [r["repository"] for r in validation], seed=20261125, repetitions=5000)
        contrasts[dataset] = {k: v for k, v in stats.items() if k not in ("projects", "leave_one_project_out")}
        contrasts[dataset].update(top1_corrected=sum(r["metrics"]["4"]["Top1"] > r["metrics"]["0"]["Top1"] for r in validation),
            top1_broken=sum(r["metrics"]["4"]["Top1"] < r["metrics"]["0"]["Top1"] for r in validation),
            top1_changed=sum(not r["same_top1_as_lambda0"]["4"] for r in validation), failures=dict(failures))
        all_per_report[dataset] = details
    previous = 0.
    for i, dataset in enumerate(sorted(contrasts, key=lambda k: contrasts[k]["p_two_sided_paired_randomization"])):
        previous = max(previous, min(1., (len(contrasts)-i)*contrasts[dataset]["p_two_sided_paired_randomization"]))
        contrasts[dataset]["p_holm_two_exploratory_prior_comparisons"] = previous
    save(out/"prior_weight_summary.json", all_summaries)
    save(out/"lambda4_vs0_exploratory.json", contrasts)
    save(out/"prior_weight_per_report.json", all_per_report)
    with (out/"prior_weight_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "phase", "lambda", "n", *KEYS])
        for dataset, phases in all_summaries.items():
            for phase, values in phases.items():
                for lam, values in values.items():
                    writer.writerow([dataset, phase, lam, values["n"], *[values[k] for k in KEYS]])
    if any(sha(Path(path)) != checksum for path, checksum in sources.items()):
        raise ValueError("Original experimental outputs changed during audit")
    save(out/"verification.json", {"status": "passed", **dict(checks), "original_artifacts_unchanged": True,
        "no_coefficients_selected": True, "model_calls": 0})
    print(json.dumps({"audit": dict(checks), "lambda4_vs0": contrasts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
