"""Run the already-frozen evidence/prior combination on the complete historical500.

All successful sources receive fresh 0.6B packet scoring. The old 45/90 scores
are not spliced into this run. The inherited scorer and evaluator are identical
to those used for the separate 180-case validation; no parameter search occurs.
"""
import argparse
from collections import Counter
from pathlib import Path

import run_c_evidence_prior_validation as frozen
from run_conservative_reranking import compact, load as load_cpu

ROOT = frozen.ROOT
REFERENCE = ROOT / "results/c_evidence_prior/20261123_validation180"
IDENTITY = ("dataset_row", "repository", "issue_id", "before_fix_sha")


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    reference = frozen.load(REFERENCE)
    if frozen.read(REFERENCE / "independent_audit.json")["status"] != "passed":
        raise ValueError("The 180-case reference was not independently verified")
    _, full, historical = load_cpu(frozen.CPU)
    index = {r["dataset_row"]: r for r in full}
    selected = [index[r["dataset_row"]] for r in historical]
    if len(selected) != 500 or len({r["dataset_row"] for r in selected}) != 500:
        raise ValueError("Require exactly the original historical500 identities")
    if {r["dataset_row"] for r in selected} != {r["dataset_row"] for r in full if r["development"]}:
        raise ValueError("Historical500 and development500 identities differ")
    previous180 = {r["dataset_row"] for r in frozen.read(REFERENCE / "labels.json")}
    if previous180 & {r["dataset_row"] for r in selected}:
        raise ValueError("Unexpected overlap with the separate 180-case validation")
    if (reference["packet_weight"], reference["prior_weight"]) != (.25, 4.):
        raise ValueError("Frozen combination changed")
    records = {r["dataset_row"]: r for r in frozen.read(frozen.FULL / "FULL/samples.json")}
    labels, plan, source_hashes = [], [], {}
    for i, (row, old) in enumerate(zip(selected, historical)):
        rec = records[row["dataset_row"]]
        if any(row[k] != old[k] or row[k] != rec[k] for k in IDENTITY):
            raise ValueError("Sample identity changed")
        if set(row["gold"]) != set(old["gold"]):
            raise ValueError("Labels differ across archived runs")
        rowid = row["dataset_row"]
        if row["base"]:
            if rec["status"] != "ok":
                raise ValueError("Inconsistent source status")
            source = frozen.FULL / rec["result_file"]
            digest = frozen.sha(source)
            if digest != rec["result_sha256"]:
                raise ValueError("Source result checksum changed")
            case = frozen.read(source)
            features = compact(case)
            if features["base"] != row["base"] or features["chunks"] != row["chunks"]:
                raise ValueError("Prepared features differ from the actual FULL result")
            if set(case["ground_truth"]) != set(row["gold"]):
                raise ValueError("Source labels differ")
            inp = {"dataset_row": rowid, "query": case["query"],
                   "packets": frozen.file_packets(case["scored_chunks"])}
            source_hashes[str(source)] = digest
            status = "ok"
        else:
            if rec["status"] == "ok":
                raise ValueError("Successful source has no candidates")
            inp = {"dataset_row": rowid, "query": "", "packets": []}
            status = "upstream_failed"
        relative = f"inputs/{rowid:04d}.json"
        frozen.save(out / relative, inp)
        plan.append({**{k: row[k] for k in IDENTITY}, "position": old["position"],
                     "worker": i % 2, "source_status": status, "input_file": relative,
                     "input_sha256": frozen.sha(out / relative)})
        labels.append(row)
        if (i + 1) % 100 == 0:
            print(f"prepared {i+1}/500", flush=True)
    keys = ("packet_weight", "prior_weight", "guard", "config", "model", "primary", "variants",
            "primary_AP_contrasts", "choice_sha256", "code_sha256")
    protocol = {k: reference[k] for k in keys}
    protocol["code_sha256"] = {**protocol["code_sha256"], str(Path(__file__).relative_to(ROOT)): frozen.sha(Path(__file__))}
    protocol.update(
        selection="Exact historical500 identities in original order; all retained, no filtering",
        not_blind="Historical development500: the prior weight was selected on these reports, and packet weight on 45 of them. Full500 is descriptive, not independent validation.",
        scoring="Fresh packet scoring for all usable sources; no reuse of the earlier45/90 packet scores",
        failure_policy="Retain upstream failures as empty rankings and zero metrics in the full denominator; inference/infrastructure errors stop instead of being silently excluded",
        baseline="FULL rerun with RRF tie ordering, identical to the separate180 experiment; legacy C3 has different scores/tie ordering",
        reference180_protocol_sha256=frozen.sha(REFERENCE / "protocol.json"),
        reference180_audit_sha256=frozen.sha(REFERENCE / "independent_audit.json"),
        historical500_sha256=frozen.sha(frozen.CPU / "historical500.json"),
        full_features_sha256=frozen.sha(frozen.CPU / "full.json"),
        sample_plan=plan, source_sha256=source_hashes,
        planned_source_status=dict(Counter(p["source_status"] for p in plan)),
        group_policy="Also report all500 excluding packet-development45 and excluding all previously packet-scored135; these subsets still participated in prior-weight selection",
    )
    frozen.save(out / "labels.json", labels)
    protocol["labels_sha256"] = frozen.sha(out / "labels.json")
    frozen.save(out / "protocol.json", protocol)
    print({"reports": len(labels), "source_status": protocol["planned_source_status"],
           "packet_weight": protocol["packet_weight"], "prior_weight": protocol["prior_weight"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.mode == "prepare":
        prepare(out)
    elif args.mode == "run":
        frozen.run(out, args.worker, args.device)
    else:
        frozen.evaluate(out)
