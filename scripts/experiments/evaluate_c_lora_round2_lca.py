"""Secondary LCA47 repeat using verified historical base and lexical scores."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
import statistics

from evaluate_c_lora import METRICS, ROOT, hard_pairs, ranking_metrics, read, read_checked, save, sha256_file
from evaluate_c_lora_lca import clustered_interval
from evaluate_c_lora_round2 import ROUND2, evaluate, verify_training

SOURCE = ROOT / "lca47"
MODES = ("base", "lexical", "hard")


def old_run(mode):
    return SOURCE / ("base" if mode == "base" else "lora")


def freeze(out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = read(SOURCE / "manifest.json")
    if manifest["n"] != 47 or any(n for s in manifest["overlap_counts"].values() for n in s.values()):
        raise ValueError("Incomplete or overlapping LCA cohort")
    paths = {"selection.json": manifest["selection_sha256"], "labels.json": manifest["labels_sha256"]}
    paths.update({f"inputs/{i:03d}.json": h for i, h in enumerate(manifest["input_sha256"], 1)})
    for p, h in paths.items():
        if sha256_file(SOURCE / p) != h:
            raise ValueError("Source data changed: " + p)
        (out / p).parent.mkdir(exist_ok=True)
        shutil.copyfile(SOURCE / p, out / p)
    old_hash = sha256_file(SOURCE / "manifest.json")
    _, adapter = verify_training(ROOT)
    runs = {}
    for m in ("base", "lexical"):
        old = old_run(m)
        r = read(old / "run.json")
        proto = read_checked(old / "protocol.json", r["protocol_sha256"])
        expected = {p.name: sha256_file(p) for p in adapter.iterdir() if p.is_file()} if m == "lexical" else None
        if (proto["manifest_sha256"] != old_hash or proto["adapter_sha256"] != expected or
                proto["model"]["device"] != "cuda:0" or proto["model"]["gpu"] != "NVIDIA GeForce RTX 4090" or
                proto["model"]["files_sha256"] != manifest["base_model_files"]):
            raise ValueError("Historical score protocol mismatch")
        runs[m] = sha256_file(old / "run.json")
    save(out / "manifest.json", {**manifest, "role": "previously evaluated LCA47; secondary paired diagnostic",
        "statistic": "hard minus base and hard minus lexical; 10000 whole-repository bootstrap draws",
        "training_data_sha256": {s: sha256_file(ROUND2 / f"data/{s}.jsonl") for s in ("train", "validation")},
        "historical_input_manifest_sha256": old_hash, "historical_runs_sha256": runs})


def summarize(out):
    man = read(out / "manifest.json")
    labs = read_checked(out / "labels.json", man["labels_sha256"])
    incoming = read(out / "hard/run.json")
    current = read_checked(out / "hard/protocol.json", incoming["protocol_sha256"])
    if current["manifest_sha256"] != sha256_file(out / "manifest.json"):
        raise ValueError("Current manifest changed")
    inputs = {}
    for m in MODES:
        folder = (out / "hard") if m == "hard" else old_run(m)
        run = incoming if m == "hard" else read_checked(folder / "run.json", man["historical_runs_sha256"][m])
        proto = read_checked(folder / "protocol.json", run["protocol_sha256"])
        if current["model"] != proto["model"] or current["config"] != proto["config"] or len(run["case_sha256"]) != len(labs):
            raise ValueError("Runtime, base model or input budget changed across variants")
        inputs[m] = (folder, run)
    rows = []
    for i, lab in enumerate(labs):
        n = f"{i+1:03d}.json"
        h = man["input_sha256"][i]
        item = read_checked(out / "inputs" / n, h)
        row = {"id": lab["id"], "repository": lab["repository"]}
        for m, (folder, run) in inputs.items():
            pr = read_checked(folder / "cases" / n, run["case_sha256"][i])
            if pr["id"] != lab["id"] or pr["input_sha256"] != h:
                raise ValueError("Unpaired model prediction")
            row[m] = ranking_metrics([p["file_path"] for p in pr["ranking"]], lab["gold"], 10)
            row[m+"_pair"] = hard_pairs(item, pr["ranking"], lab["gold"])
            row[m+"_status"] = pr["status"]
        rows.append(row)
    summary = {m: {k: statistics.mean(r[m][v] for r in rows) for k, v in METRICS.items()} for m in MODES}
    hard = [r for r in rows if all(r[m+"_pair"] is not None for m in MODES)]
    summary.update(n=len(rows), projects=len({r["repository"] for r in rows}), contrasts={},
                   role=man["role"], hard_pair_reports=len(hard),
                   status_counts={m: Counter(r[m+"_status"] for r in rows) for m in MODES},
                   hard_pair_accuracy={m: statistics.mean(r[m+"_pair"] for r in hard) for m in MODES})
    for ref in ("base", "lexical"):
        groups = defaultdict(list)
        for r in rows:
            groups[r["repository"]].append(r["hard"]["average_precision_at_k"] - r[ref]["average_precision_at_k"])
        ds = [x for v in groups.values() for x in v]
        summary["contrasts"]["hard_minus_"+ref] = dict(delta_MAP=statistics.mean(ds),
            project_cluster_CI95=clustered_interval(list(groups.values())), wins=sum(x>0 for x in ds),
            ties=ds.count(0), losses=sum(x<0 for x in ds))
    save(out / "paired_cases.json", rows)
    summary["cases_sha256"] = sha256_file(out / "paired_cases.json")
    save(out / "paired_summary.json", summary)
    print(json.dumps(summary, indent=2))


def anchor(out):
    import c_lora_round2_anchor as safeguard
    folder = ROUND2 / "anchor"
    choice = read(folder / "choice.json")
    plan = read_checked(folder / "protocol.json", choice["protocol_sha256"])
    dev = read_checked(folder / "development_scores.json", choice["scores_sha256"])
    if safeguard.choose([safeguard.summarize(dev, w) for w in plan["grid"]]) != choice["weight"]:
        raise ValueError("Development-only model mixing choice changed")
    man = read(out / "manifest.json")
    labels = read_checked(out / "labels.json", man["labels_sha256"])
    runs = {"base": read_checked(old_run("base") / "run.json", man["historical_runs_sha256"]["base"]),
            "hard": read(out / "hard/run.json")}
    hp = read_checked(out/"hard/protocol.json", runs["hard"]["protocol_sha256"])
    if hp["adapter_sha256"]["adapter.safetensors"] != choice["adapter_sha256"]:
        raise ValueError("Mismatch between development and LCA adapter")
    rows, paired = [], []
    for i, lab in enumerate(labels):
        row = {"gold": lab["gold"]}
        for mode, run in runs.items():
            source = old_run(mode) if mode == "base" else out / "hard"
            case = read_checked(source / "cases" / f"{i+1:03d}.json", run["case_sha256"][i])
            if case["id"] != lab["id"] or case["input_sha256"] != man["input_sha256"][i]:
                raise ValueError("Unpaired file scores")
            row[mode] = case["ranking"]
        vals = safeguard.metrics(row["base"], row["hard"], row["gold"], choice["weight"])
        paired.append(dict(id=lab["id"], metrics=vals))
        rows.append(row)
    save(out/"anchor.json", dict(n=len(rows), role="secondary, no LCA tuning", cases=paired,
         choice_sha256=sha256_file(folder/"choice.json"), result=safeguard.summarize(rows, choice["weight"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare", "hard", "summarize", "anchor"))
    ap.add_argument("--output", type=Path, default=ROUND2 / "lca47")
    args = ap.parse_args()
    if args.mode == "prepare":
        freeze(args.output)
    elif args.mode == "hard":
        if os.getenv("CUDA_VISIBLE_DEVICES", "0,1").split(",")[0] != "0":
            raise ValueError("Use physical GPU0 to match round-one LCA evaluation")
        evaluate(args.output, "hard", "cuda:0")
    elif args.mode == "anchor":
        anchor(args.output)
    else:
        summarize(args.output)
