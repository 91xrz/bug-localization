"""Check that the paired LoRA experiment only changed its training negatives."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

PROJECT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def pool(row):
    result = {f["path"]: (f["source_sha256"], tuple(f["chunks"])) for f in row["files"]}
    require(len(result) == len(row["files"]), f"Duplicate path for {row['id']}")
    return result


def compare_data(old, new):
    require(bool(old) and len(old) == len(new), "Train cardinality changed")
    require(len({r["id"] for r in old}) == len(old), "Duplicate training report")
    changed, common, new_negs, pos_count = 0, 0, 0, 0
    by_repo = Counter()
    for a, b in zip(old, new):
        for k in ("id", "repo", "sha", "dataset_row", "query", "body_sha256", "gold",
                  "available_old_python_gold", "python_paths_in_snapshot"):
            require(a[k] == b[k], f"Report field {k} differs for {a['id']}")
        pa, pb, gold = pool(a), pool(b), set(a["gold"])
        pos = pa.keys() & gold
        require(pos and pos == (pb.keys() & gold), f"Positive set changed: {a['id']}")
        for p in pa.keys() & pb.keys():
            require(pa[p] == pb[p], f"Shared evidence changed: {a['id']} {p}")
        na, nb = pa.keys() - pos, pb.keys() - pos
        require(len(na) == len(nb) == 4, f"Negative budget changed: {a['id']}")
        changed += na != nb
        common += len(na & nb)
        new_negs += len(nb - na)
        pos_count += len(pos)
        by_repo[a["repo"]] += 1
    require(bool(changed), "No training negative changed")
    return {"train_reports": len(old), "reports_with_changed_negatives": changed,
            "unchanged_positive_files": pos_count, "shared_negatives": common,
            "replaced_negative_slots": new_negs, "train_projects": dict(by_repo)}


def preflight(old, new):
    old_p = read(old / "training/protocol.json")
    data = [root / "data" for root in (old, new)]
    require(digest(data[0] / "train.jsonl") == old_p["train_sha256"], "Round 1 train drift")
    require(all(digest(d / "validation.jsonl") == old_p["validation_sha256"] for d in data),
            "Validation bytes changed")
    require(digest(PROJECT / "scripts/experiments/train_c_lora.py") == old_p["program_sha256"],
            "Training implementation changed")
    require(read(PROJECT / "configs/retrieval/reranker.json") == old_p["original"], "Model config changed")
    train_sets = [rows(d / "train.jsonl") for d in data]
    report = compare_data(*train_sets)
    val = rows(data[1] / "validation.jsonl")
    require(len(val) == old_p["validation_reports"] and len(train_sets[0]) == old_p["train_reports"],
            "Historical dataset size mismatch")
    train, valid = ({r["repo"] for r in group} for group in (train_sets[1], val))
    require(not train & valid, "Training/validation project overlap")
    report.update(validation_sha256=old_p["validation_sha256"], train_hashes=[
        digest(d / "train.jsonl") for d in data], validation_reports=len(val),
        training_program_sha256=old_p["program_sha256"])
    return report


def finish(old, new, audit):
    import torch
    from safetensors.torch import load_file

    roots = [r / "training" for r in (old, new)]
    protocols = [read(r / "protocol.json") for r in roots]
    for p, train_hash in zip(protocols, audit["train_hashes"]):
        require(p["train_sha256"] == train_hash, "Train hash mismatch")
    recipes = [{k: v for k, v in p.items() if k != "train_sha256"} for p in protocols]
    require(recipes[0] == recipes[1], "Training recipe differs")
    require(read(roots[0] / "validation_epoch0.json") == read(roots[1] / "validation_epoch0.json"),
            "Unequal validation predictions at initialization")
    summaries = [read(r / "summary.json") for r in roots]
    for k in ("base_identity", "steps", "trainable_parameters"):
        require(summaries[0][k] == summaries[1][k], f"{k} differs")
    configs = [read(r / "adapter/adapter_config.json") for r in roots]
    for r, conf, summary in zip(roots, configs, summaries):
        require(digest(r / "adapter/adapter_config.json") == summary["adapter_config_sha256"] and
                digest(r / "protocol.json") == conf["protocol_sha256"] and
                digest(r / "adapter/adapter.safetensors") == conf["adapter_sha256"], "Broken adapter chain")
        require(conf["base_files_sha256"] == summary["base_identity"]["files_sha256"], "Base weights differ")
    for k in ("format", "rank", "alpha", "targets", "base_files_sha256", "instruction"):
        require(configs[0][k] == configs[1][k], f"Adapter definition {k} differs")
    weights = [load_file(str(r / "adapter/adapter.safetensors"), device="cpu") for r in roots]
    require(weights[0].keys() == weights[1].keys(), "Adapter tensor names differ")
    changed = 0
    for k, w1 in weights[0].items():
        w2 = weights[1][k]
        require(w1.shape == w2.shape and w1.dtype == w2.dtype and bool(torch.isfinite(w2).all()),
                f"Invalid tensor {k}")
        changed += not torch.equal(w1, w2)
    require(changed > 0, "Trained tensors identical")
    metrics = []
    for r in roots:
        v = read(r / "validation_epoch2.json")
        for k, row_key in (("sampled_pool_MAP@10", "AP@10"), ("mean_pair_accuracy", "pair_accuracy")):
            require(abs(v[k] - mean(x[row_key] for x in v["reports"])) < 1e-12, "Validation aggregation error")
        metrics.append({k: v[k] for k in ("sampled_pool_MAP@10", "mean_pair_accuracy")})
    audit.update(same_base_and_recipe=True, epoch0_equal=True, steps=summaries[1]["steps"],
                 different_adapter_tensors=changed, adapter_tensors=len(weights[1]),
                 adapter_sha256_rounds=[c["adapter_sha256"] for c in configs], final_validation=metrics)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1", type=Path, default=PROJECT / "results/c_lora/round1")
    parser.add_argument("--round2", type=Path, default=PROJECT / "results/c_lora/round2")
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = preflight(args.round1, args.round2)
    if args.finish:
        report = finish(args.round1, args.round2, report)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as out:
            out.write(text + "\n")
    print(text)
