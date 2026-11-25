"""Paired intervals for the already-frozen secondary interpolation, not grid selection."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from c_lora_round2_anchor import ROUND, metrics
from evaluate_c_lora import read, read_checked, save, sha256_file
from evaluate_c_lora_round2 import contrast


def compare(folder, evaluation):
    result = read(folder/"test.json")
    choice = read_checked(folder/"choice.json", result["choice_sha256"])
    manifest = read(evaluation/"manifest.json")
    labels = read_checked(evaluation/"labels.json", manifest["labels_sha256"])
    if result["selection_sha256"] != manifest["selection_sha256"]:
        raise ValueError("Frozen secondary result differs from evaluation")
    runs = {k:read(evaluation/k/"run.json") for k in ("base", "hard")}
    rows = []
    for i, label in enumerate(labels):
        rankings = {}
        for m, run in runs.items():
            c = read_checked(evaluation/m/f"cases/{i+1:03d}.json", run["case_sha256"][i])
            if c["id"] != label["id"] or c["input_sha256"] != manifest["input_sha256"][i]:
                raise ValueError("Input identity differs")
            rankings[m] = c["ranking"]
        rows.append(dict(repository=label["repository"], id=label["id"], blagent=label["blagent_metrics"],
                         **{m:metrics(rankings["base"],rankings["hard"],label["gold"],w)
                            for m,w in [("base",0),("hard",1),("anchored",choice["weight"])]}))
    if abs(sum(r["anchored"]["average_precision_at_k"] for r in rows)/len(rows) -
           result["result"]["MAP@10"]) > 1e-12:
        raise ValueError("Could not reproduce frozen test result")
    return {"test_sha256":sha256_file(folder/"test.json"), "comparisons":{
        "anchor_minus_" + m:contrast(rows, "anchored", m) for m in ("base","hard","blagent")}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, default=ROUND/"anchor")
    p.add_argument("--evaluation", type=Path, default=ROUND/"evaluation")
    args = p.parse_args()
    out = args.folder/"paired_intervals.json"
    if out.exists():
        raise FileExistsError(out)
    save(out, compare(args.folder, args.evaluation))
