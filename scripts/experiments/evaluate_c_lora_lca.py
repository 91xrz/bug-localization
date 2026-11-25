"""External LCA diagnostic. No checkpoint or parameter selection on LCA."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

from evaluate_c_lora import (PROJECT, ROOT, METRICS, checked_result, check_disjoint,
                             frozen_input, hard_pairs, identity, prepare_report, ranking_metrics,
                             read, read_checked, save, score, sha256_file)

ARCHIVE = PROJECT / "results/c3_supplement/20261112_lca_py_holdout"
DATA = ROOT / "data"
ADAPTER = ROOT / "training/adapter"
SEED = 20261120


def body_key(text):
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def prepare(root):
    from datasets import load_from_disk
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a fresh LCA evaluation directory")
    archive = ARCHIVE / "C3"
    selection = read(ARCHIVE / "selection.json")
    rows = read(archive / "samples.json")
    if (len(rows) != 47 or len(set(map(identity, rows))) != 47 or
            list(map(identity, rows)) != list(map(identity, selection["samples"]))):
        raise ValueError("LCA archive no longer matches the complete selection")
    dsroot = PROJECT / "datasets/lca_py_formatted/test"
    ds = load_from_disk(str(dsroot))
    if sha256_file(dsroot / "dataset_info.json") != selection["dataset_metadata_sha256"]:
        raise ValueError("LCA dataset revision changed")
    train = {}
    for split in ("train", "validation"):
        rows_train = [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()]
        train[split] = ({r["repo"].casefold() for r in rows_train}, {body_key(r["query"]) for r in rows_train})
    labels, inputs = [], []
    for i, row in enumerate(rows, 1):
        repo = row["repository"]
        raw = ds[row["dataset_row"]]
        official, gold, normalized = prepare_report(raw, {"repository": repo, "issue_id": row["issue_id"], "query_fields": ["body"]})
        case = checked_result(archive, row)
        if (official["before_fix_sha"] != row["before_fix_sha"] or set(gold) != set(case["ground_truth"]) or
                not normalized.startswith(case["query"]) or row["status"] != "ok"):
            raise ValueError("LCA archive no longer verified against dataset: " + str(i))
        historical = ranking_metrics([r["file_path"] for r in case["ranking"]], gold, 10)
        if historical != row["metrics"]:
            raise ValueError("Historical metric mismatch")
        uid = f'{repo}#{row["issue_id"]}@{row["before_fix_sha"]}'
        item = dict(position=i, id=uid, repository=repo, source_status=row["status"], **frozen_input(case))
        path = root / "inputs" / f"{i:03d}.json"
        save(path, item)
        inputs.append(sha256_file(path))
        overlap = {s: {"repository": repo.casefold() in rs, "normalized_body": body_key(raw["body"]) in bs}
                   for s, (rs, bs) in train.items()}
        labels.append(dict(position=i, id=uid, repository=repo, gold=gold, history_metrics=historical,
                           body_sha256=hashlib.sha256(raw["body"].encode()).hexdigest(),
                           overlap=overlap, c3_source_sha256=row["result_sha256"]))
    save(root / "labels.json", labels)
    save(root / "selection.json", selection["samples"])
    protocol = read(ARCHIVE / "protocol.json")
    configfile = PROJECT / protocol["config"]["reranker_config"]
    if sha256_file(configfile) != protocol["files_sha256"][str(configfile)]:
        raise ValueError("Historical reranker configuration changed")
    man = dict(n=47, seed=SEED, primary_dataset=False, role="external generalization diagnostic only",
               statistic="LoRA minus same-device base, all 47 reports; 10000 project-cluster bootstrap draws",
               selection="all archived LCA reports; no filtering", checkpoint="fixed final epoch 2",
               bootstrap_unit="whole repositories sampled with replacement; report-weighted mean per draw",
               source_samples_sha256=sha256_file(archive / "samples.json"),
               selection_sha256=sha256_file(root / "selection.json"), input_sha256=inputs,
               labels_sha256=sha256_file(root / "labels.json"), config=read(configfile),
               training_data_sha256={s: sha256_file(DATA / f"{s}.jsonl") for s in train},
               overlap_counts={s: {k: sum(x["overlap"][s][k] for x in labels)
                                   for k in ("repository", "normalized_body")} for s in train},
               base_model_files=read(ARCHIVE / "reranker_identity.json")["model"]["files_sha256"])
    save(root / "manifest.json", man)
    print(json.dumps({"n": man["n"], "overlaps": man["overlap_counts"]}))


def clustered_interval(groups):
    rng = random.Random(SEED)
    totals = [(sum(g), len(g)) for g in groups]
    vals = []
    for _ in range(10000):
        draws = rng.choices(totals, k=len(groups))
        vals.append(sum(t for t, n in draws) / sum(n for t, n in draws))
    vals.sort()
    return [vals[249], vals[9749]]


def summarize(root):
    manifest = read(root / "manifest.json")
    labels = read_checked(root / "labels.json", manifest["labels_sha256"])
    runs = {name: read(root / name / "run.json") for name in ("base", "lora")}
    rows = []
    for i, label in enumerate(labels):
        number = f"{i + 1:03d}.json"
        item = read_checked(root / "inputs" / number, manifest["input_sha256"][i])
        row = {k: label[k] for k in ("id", "repository", "history_metrics", "overlap")}
        for name, run in runs.items():
            result = read_checked(root / name / "cases" / number, run["case_sha256"][i])
            if result["id"] != row["id"] or result["input_sha256"] != manifest["input_sha256"][i]:
                raise ValueError("LCA paired identity mismatch")
            row[name] = ranking_metrics([r["file_path"] for r in result["ranking"]], label["gold"], 10)
            row[name + "_hard_pair"] = hard_pairs(item, result["ranking"], label["gold"])
            row[name + "_status"] = result["status"]
        rows.append(row)
    for name in runs:
        if sha256_file(root / name / "protocol.json") != runs[name]["protocol_sha256"]:
            raise ValueError("LCA run protocol changed")
    save(root / "paired_cases.json", rows)
    tot = {name: {m: statistics.mean(r[name][k] for r in rows) for m, k in METRICS.items()}
           for name in ("base", "lora", "history_metrics")}
    groups = defaultdict(list)
    for r in rows:
        groups[r["repository"]].append(r["lora"]["average_precision_at_k"] - r["base"]["average_precision_at_k"])
    deltas = [x for group in groups.values() for x in group]
    hard = [r for r in rows if r["base_hard_pair"] is not None and r["lora_hard_pair"] is not None]
    out = dict(**tot, n=len(rows), projects=len(groups), delta_MAP=statistics.mean(deltas),
               wins=sum(d > 0 for d in deltas), ties=deltas.count(0), losses=sum(d < 0 for d in deltas),
               delta_Top1=tot["lora"]["Top-1"]-tot["base"]["Top-1"],
               project_cluster_CI95=clustered_interval(list(groups.values())), hard_pair_eligible=len(hard),
               hard_pair_accuracy={n: statistics.mean(r[n+"_hard_pair"] for r in hard) if hard else None for n in runs},
               counts={n: Counter(r[n+"_status"] for r in rows) for n in runs},
               historical_metric_reproduced=sum(r["history_metrics"] == r["base"] for r in rows),
               paired_sha256=sha256_file(root / "paired_cases.json"), overlaps=manifest["overlap_counts"])
    save(root / "paired_summary.json", out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "base", "lora", "summarize"])
    parser.add_argument("--output", type=Path, default=ROOT / "lca47")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
    elif args.mode == "summarize":
        summarize(args.output)
    else:
        manifest = read(args.output / "manifest.json")
        if any(value for group in manifest["overlap_counts"].values() for value in group.values()):
            raise ValueError("Overlap found; separate marked groups before drawing cross-project conclusions")
        check_disjoint(args.output, manifest, DATA)
        if args.mode == "lora":
            protocol = read(ADAPTER.parent / "protocol.json")
            if protocol["epochs"] != 2 or protocol["selection"] != "fixed final epoch":
                raise ValueError("LCA is only valid for fixed epoch 2")
        score(args.output, args.mode, args.device, ADAPTER, DATA)
