"""Choose a file-score interpolation only on a disjoint, archived development set."""
import argparse
from pathlib import Path
from statistics import fmean

from evaluate_c_lora import (C3, PILOT, PROJECT, METRICS, LocalReranker, checked_result, frozen_input,
                             identity, infer, ranking_metrics, read, read_checked, save, sha256_file)

ROUND = PROJECT / "results/c_lora/round2"
GRID = (0., .25, .5, 1.)


def metrics(base, hard, gold, weight):
    cold = {r["file_path"]: r["score"] for r in base}
    hot = {r["file_path"]: r["score"] for r in hard}
    ties = {r["file_path"]: r["semantic_rank"] for r in base}
    if len(cold) != len(base) or cold.keys() != hot.keys() or len(hot) != len(hard):
        raise ValueError("The models must score the identical file set")
    ranking = sorted(cold, key=lambda p: (-((1-weight)*cold[p] + weight*hot[p]), ties[p]))
    return ranking_metrics(ranking, gold, 10)


def summarize(rows, weight):
    blends = [metrics(row["base"], row["hard"], row["gold"], weight) for row in rows]
    original = [metrics(row["base"], row["hard"], row["gold"], 0) for row in rows]
    return dict(weight=weight, **{k: fmean(b[v] for b in blends) for k, v in METRICS.items()},
                lost=sum(b["top_1"] > h["top_1"] for b, h in zip(original, blends)),
                rescued=sum(b["top_1"] < h["top_1"] for b, h in zip(original, blends)))


def choose(grid):
    eligible = [r for r in grid if r["lost"] <= 1 and r["rescued"] >= r["lost"]]
    return max(eligible, key=lambda r: (r["MAP@10"], -r["weight"]))["weight"]


def prepare(folder):
    if folder.exists():
        raise FileExistsError(folder)
    reports = read(PILOT/"development45.json")["samples"]
    if len({identity(r) for r in reports}) != 45:
        raise ValueError("Incorrect development set")
    c3 = {identity(r): r for r in read(C3/"samples.json")}
    save(folder/"protocol.json", dict(grid=GRID, development=[{**r, "source_sha256":c3[identity(r)]["result_sha256"]}
                                                               for r in reports],
        rule="max development MAP among weights losing at most one correct top1 and gaining at least as many; lower weight wins ties",
        role="secondary safeguard only; raw second round LoRA is the primary result",
        archived_selection_sha256=sha256_file(PILOT/"development45.json"),
        script_sha256=sha256_file(Path(__file__))))


def calibrate(folder, device):
    from train_c_lora import load_adapter
    protocol = read(folder/"protocol.json")
    if (folder/"choice.json").exists() or list(GRID) != protocol["grid"] or sha256_file(Path(__file__)) != protocol["script_sha256"]:
        raise ValueError("Calibration repeated or protocol changed")
    records = {identity(r): r for r in read(C3/"samples.json")}
    conf = read(C3/"effective_config.json")
    adapter = ROUND/"training/adapter"
    model = LocalReranker(PROJECT, {**conf["reranker"], "device":device})
    if model.identity["files_sha256"] != conf["model"]["files_sha256"]:
        raise ValueError("Wrong base checkpoint")
    load_adapter(model, adapter)
    rows = []
    for i, item in enumerate(protocol["development"], 1):
        source = records[identity(item)]
        if source["result_sha256"] != item["source_sha256"] or source["status"] != "ok":
            raise ValueError("Archived development cohort differs")
        case = checked_result(C3, source)
        hard = infer(dict(source_status="ok", id=str(identity(item)), **frozen_input(case)), model)
        rows.append(dict(id=hard["id"], gold=case["ground_truth"], base=case["ranking"], hard=hard["ranking"]))
        if i % 10 == 0:
            print(f"development: {i}/45", flush=True)
    save(folder/"development_scores.json", rows)
    grid = [summarize(rows, w) for w in GRID]
    choice = dict(weight=choose(grid), development=grid, n=len(rows),
                  protocol_sha256=sha256_file(folder/"protocol.json"),
                  adapter_sha256=sha256_file(adapter/"adapter.safetensors"),
                  scores_sha256=sha256_file(folder/"development_scores.json"))
    save(folder/"choice.json", choice)
    print(choice, flush=True)


def evaluate(folder, evaluation):
    if (folder/"test.json").exists():
        raise FileExistsError(folder/"test.json")
    choice = read(folder/"choice.json")
    plan = read_checked(folder/"protocol.json", choice["protocol_sha256"])
    dev = read_checked(folder/"development_scores.json", choice["scores_sha256"])
    if choose([summarize(dev, w) for w in plan["grid"]]) != choice["weight"]:
        raise ValueError("Choice not reproducible from development")
    manifest = read(evaluation/"manifest.json")
    testing = read_checked(evaluation/"selection.json", manifest["selection_sha256"])
    if {identity(r) for r in plan["development"]} & {identity(r) for r in testing}:
        raise ValueError("Test/development overlap")
    labels = read_checked(evaluation/"labels.json", manifest["labels_sha256"])
    runs = {key:read(evaluation/key/"run.json") for key in ("base","hard")}
    hp = read_checked(evaluation/"hard/protocol.json", runs["hard"]["protocol_sha256"])
    if hp["adapter_sha256"]["adapter.safetensors"] != choice["adapter_sha256"]:
        raise ValueError("Unexpected test weights")
    rows = []
    for i, lab in enumerate(labels):
        row = {"id":lab["id"], "gold":lab["gold"]}
        for k, r in runs.items():
            case = read_checked(evaluation/k/f"cases/{i+1:03d}.json", r["case_sha256"][i])
            if case["id"] != lab["id"] or case["input_sha256"] != manifest["input_sha256"][i]:
                raise ValueError("Test identity mismatch")
            row[k] = case["ranking"]
        rows.append(row)
    out = dict(n=len(rows), choice_sha256=sha256_file(folder/"choice.json"),
               selection_sha256=manifest["selection_sha256"], result=summarize(rows, choice["weight"]))
    save(folder/"test.json", out)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare","calibrate","evaluate"))
    ap.add_argument("--folder", type=Path, default=ROUND/"anchor")
    ap.add_argument("--evaluation", type=Path, default=ROUND/"evaluation")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare(args.folder)
    elif args.mode == "calibrate":
        calibrate(args.folder, args.device)
    else:
        evaluate(args.folder, args.evaluation)
