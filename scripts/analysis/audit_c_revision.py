"""Independently recompute metrics and invariants for the C revision study."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/experiments"))
from run_conservative_reranking import interval


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def metrics(order, gold):
    if len(order) != len(set(order)):
        raise ValueError("Duplicate file in output")
    relevant = set(gold)
    found = [i for i, p in enumerate(order[:10], 1) if p in relevant]
    return {"MAP@10": sum((j+1)/i for j, i in enumerate(found))/len(relevant),
            "MRR@10": 1/found[0] if found else 0., "Top1": int(1 in found),
            "Top5": int(any(i <= 5 for i in found)), "Top10": int(bool(found)),
            "Recall@10": len(found)/len(relevant)}


def audit(cpu, packets):
    details = read(cpu / "per_report.json")
    summary = read(cpu / "evaluation.json")
    verified = 0
    for cohort, rows in details.items():
        totals = {}
        for row in rows:
            base = row["methods"]["C3_max" if cohort.startswith("historical") else "FULL_max"]["ranking"]
            for method, item in row["methods"].items():
                m = metrics(item["ranking"], row["gold"])
                if abs(m["MAP@10"] - item["metrics"]["average_precision_at_k"]) > 1e-12:
                    raise ValueError("Saved AP differs from independent computation")
                totals.setdefault(method, {k: 0. for k in m})
                for k, value in m.items():
                    totals[method][k] += value
                if method in ("8B_guarded", "8B_fallback_only", "residual_head10"):
                    if set(item["ranking"][:10]) != set(base[:10]) or item["ranking"][10:] != base[10:]:
                        raise ValueError("Fixed-head invariant failed")
                verified += 1
        for method, sums in totals.items():
            for name, total in sums.items():
                if abs(summary[cohort][method][name] - total/len(rows)) > 1e-12:
                    raise ValueError("Aggregate metric mismatch")
    risk_comparison = {}
    for cohort in ("historical_new320", "historical_all500"):
        rows = details[cohort]
        originally_correct = [r for r in rows if metrics(r["methods"]["C3_max"]["ranking"], r["gold"])["Top1"]]
        counts = {"originally_correct": len(originally_correct), "prevented_errors": 0, "introduced_errors": 0}
        deltas = []
        for row in originally_correct:
            before = metrics(row["methods"]["8B_fallback_only"]["ranking"], row["gold"])["Top1"]
            after = metrics(row["methods"]["8B_guarded"]["ranking"], row["gold"])["Top1"]
            counts["prevented_errors"] += int(after > before)
            counts["introduced_errors"] += int(after < before)
            deltas.append(before-after)  # guarded error rate minus fallback-only error rate
        counts["conditional_error_rate_change"] = interval(originally_correct, deltas)
        risk_comparison[cohort] = counts
    historical = read(cpu / "historical500.json")
    choice = read(cpu / "choices.json")["guard"]
    output = {r["dataset_row"]: r for r in details["historical_all500"]}
    protected, blocked = 0, 0
    if isinstance(choice, dict) and choice["margin"] is not None:
        for row in historical:
            base, chunks = row["c3"]["base"], row["c3"]["chunks"]
            final = output[row["dataset_row"]]["methods"]["8B_guarded"]["ranking"]
            for i, a in enumerate(base[:10]):
                support = sorted(chunks[a], reverse=True)
                if choice["corroboration"] and len(support) < 2:
                    continue
                strength = support[1] if choice["corroboration"] else support[0]
                for b in base[i+1:10]:
                    if strength-max(chunks[b]) >= choice["margin"] and row["prior"][a]-row["prior"][b] >= choice["prior_margin"]:
                        if final.index(a) >= final.index(b):
                            raise ValueError("Protected evidence order violated")
                        protected += 1
                        if row["proposal"] and row["proposal"].index(a) > row["proposal"].index(b):
                            blocked += 1
    packet_protocol = read(packets / "protocol.json")
    packet_rows = read(packets / "labels.json")
    packet_cost, more_stats = {}, {}
    for cohort in ("development45", "confirmation90"):
        rows = [r for r in packet_rows if r["cohort"] == cohort]
        ranked = read(packets / (cohort + "_rankings.json"))
        orders = ranked["orders"]
        if ranked["dataset_rows"] != [r["dataset_row"] for r in rows]:
            raise ValueError("Packet ranking order mismatch")
        packet_report = read(packets / (cohort + "_summary.json"))["summary"]
        vals = {}
        for method, ranks in orders.items():
            vals[method] = [metrics(o, r["gold"]) for r, o in zip(rows, ranks)]
            for name in vals[method][0]:
                if abs(sum(m[name] for m in vals[method])/len(rows)-packet_report[method][name]) > 1e-12:
                    raise ValueError("Packet aggregate mismatch")
            if "head10" in method:
                base_name = method.split("_")[0] + "_max"
                for order, base in zip(ranks, orders[base_name]):
                    if set(order[:10]) != set(base[:10]) or order[10:] != base[10:]:
                        raise ValueError("Packet head-membership invariant failed")
        for method in ("C1", "C3"):
            times, tokens, count = [], 0, 0
            for row in rows:
                relative = f"inputs/{method}/{row['dataset_row']:04d}.json"
                inp_path = packets / relative
                if sha(inp_path) != packet_protocol["input_sha256"][relative]:
                    raise ValueError("Packet input changed")
                inp = read(inp_path)
                if set(inp) != {"dataset_row", "query", "packets"}:
                    raise ValueError("Unexpected data at inference boundary")
                result = read(packets / method / "cases" / f"{row['dataset_row']:04d}.json")
                if result["input_sha256"] != sha(inp_path):
                    raise ValueError("Score input hash mismatch")
                if {r["file_path"] for r in result["scores"]} != {r["file_path"] for r in inp["packets"]}:
                    raise ValueError("Packet score coverage mismatch")
                if any(not math.isfinite(s["score"]) or s["input_tokens"] > packet_protocol["config"]["max_tokens"] for s in result["scores"]):
                    raise ValueError("Invalid score/context budget")
                count += len(result["scores"])
                tokens += sum(s["input_tokens"] for s in result["scores"])
                times.append(result["seconds"])
            packet_cost[cohort + "/" + method] = {"n": len(rows), "scored_file_documents": count,
                "input_tokens": tokens, "sum_scoring_seconds": sum(times), "mean_scoring_seconds": sum(times)/len(times)}
        for new, old in (("C3_packet_full50", "C3_max"), ("C3_chosen_head10", "C3_max")):
            more_stats[cohort + "/" + new + "_minus_" + old] = interval(rows, [a["MAP@10"]-b["MAP@10"] for a,b in zip(vals[new], vals[old])])
        more_stats[cohort + "/mixed_selection_difference_in_differences"] = interval(rows,
            [(c["MAP@10"]-a["MAP@10"])-(cm["MAP@10"]-am["MAP@10"])
             for c,a,cm,am in zip(vals["C3_packet_full50"],vals["C1_packet_full50"],vals["C3_max"],vals["C1_max"])])
    result = {"status": "passed", "cpu_ranking_metric_checks": verified,
              "guard_protected_pairs": protected, "guard_blocked_inversions": blocked,
              "historical_prior_missing_cases": sum(bool(r["prior_anchor_unavailable"]) for r in historical),
              "historical_prior_missing_file_instances": sum(len(r["prior_anchor_unavailable"]) for r in historical),
              "8B_explicit_fallback_cases": sum(bool(r["c3"]["base"]) and r["proposal"] is None for r in historical),
              "guard_risk_comparison_vs_fallback": risk_comparison,
              "packet_cost": packet_cost, "packet_additional_statistics": more_stats,
              "not_end_to_end_timing": True, "source_sha256": {
                  "cpu_evaluation": sha(cpu / "evaluation.json"), "cpu_choices": sha(cpu / "choices.json"),
                  "packet_choices": sha(packets / "choice.json"), "audit_script": sha(Path(__file__))}}
    save(cpu / "independent_audit.json", result)
    with (cpu / "comparison.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cohort", "method", "n", "MAP@10", "Top1", "Top10", "corrected_top1", "broken_top1"])
        for cohort, methods in summary.items():
            for name, m in methods.items():
                writer.writerow([cohort, name] + [m[k] for k in ("n", "MAP@10", "Top1", "Top10", "corrected_top1", "broken_top1")])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    args = parser.parse_args()
    audit(args.cpu.resolve(), args.packets.resolve())
