"""Independent metric and inference-boundary audit of the frozen 180-case test."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from audit_c_revision import metrics, read, save, sha


def audit(out):
    p = read(out / "protocol.json")
    rows = read(out / "labels.json")
    ranks = read(out / "rankings.json")
    summary = read(out / "summary.json")
    old = read(ROOT / "results/conservative_reranking/20261123_v3/historical500.json")
    if {r["dataset_row"] for r in old} & {r["dataset_row"] for r in rows}:
        raise ValueError("Development/test overlap")
    if ranks["dataset_rows"] != [r["dataset_row"] for r in rows] or len(rows) != 180:
        raise ValueError("Wrong test cohort")
    if p["packet_weight"] != .25 or p["prior_weight"] != 4.:
        raise ValueError("Not the previously frozen component weights")
    inputs, protected = 0, 0
    for i, (row, meta) in enumerate(zip(rows,p["sample_plan"])):
        inp = read(out / meta["input_file"])
        if sha(out / meta["input_file"]) != meta["input_sha256"] or set(inp) != {"dataset_row","query","packets"}:
            raise ValueError("Inference input changed or includes unexpected fields")
        result = read(out / "cases" / f"{row['dataset_row']:04d}.json")
        if result["input_sha256"] != meta["input_sha256"] or result["protocol_sha256"] != sha(out / "protocol.json"):
            raise ValueError("Result provenance mismatch")
        scores = {s["file_path"]:s["score"] for s in result["scores"]}
        if (len(scores) != len(result["scores"]) or set(scores) != set(row["base"])
                or set(scores) != {d["file_path"] for d in inp["packets"]}):
            raise ValueError("Candidate pool mismatch")
        if any(not math.isfinite(s["score"]) or s["input_tokens"] > p["config"]["max_tokens"] for s in result["scores"]):
            raise ValueError("Invalid scoring result")
        inputs += len(scores)
        base = row["base"]
        for name in ("prior_head10","packet_head10","combined_head10","combined_guarded"):
            final = ranks["orders"][name][i]
            if set(final[:10]) != set(base[:10]) or final[10:] != base[10:]:
                raise ValueError("Head or tail changed")
        for name, packet_weight, prior_weight in (("prior_head10",0.,4.),("packet_head10",.25,0.),("combined_head10",.25,4.)):
            features = {f:(1-packet_weight)*max(row["chunks"][f])+packet_weight*scores[f]+prior_weight*row["prior"][f] for f in base}
            expected = sorted(base[:10], key=lambda f:(-features[f],base.index(f)))+base[10:]
            if ranks["orders"][name][i] != expected:
                raise ValueError("Frozen interpolation mismatch")
        final = ranks["orders"]["combined_guarded"][i]
        for j, a in enumerate(base[:10]):
            for b in base[j+1:10]:
                if max(row["chunks"][a])-max(row["chunks"][b]) >= p["guard"]["margin"] and row["prior"][a]-row["prior"][b] >= p["guard"]["prior_margin"]:
                    if final.index(a) >= final.index(b):
                        raise ValueError("Protected pair violated")
                    protected += 1
    values = {name:[metrics(o,r["gold"]) for o,r in zip(order,rows)] for name,order in ranks["orders"].items()}
    for name, ms in values.items():
        for key in ms[0]:
            if abs(sum(m[key] for m in ms)/len(rows)-summary[name][key]) > 1e-12:
                raise ValueError("Independent aggregate mismatch")
        base = values["FULL_max"]
        corrected = sum(a["Top1"]>b["Top1"] for a,b in zip(ms,base))
        broken = sum(a["Top1"]<b["Top1"] for a,b in zip(ms,base))
        if corrected != summary[name]["corrected_top1"] or broken != summary[name]["broken_top1"]:
            raise ValueError("Transition count mismatch")
    projects = {}
    for repo in sorted({r["repository"] for r in rows}):
        idx = [i for i,r in enumerate(rows) if r["repository"]==repo]
        projects[repo] = {name:{"n":len(idx),"MAP@10":sum(ms[i]["MAP@10"] for i in idx)/len(idx),
                                   "Top1":sum(ms[i]["Top1"] for i in idx)/len(idx)} for name,ms in values.items()}
    save(out / "project_summary.json", projects)
    saved = {"status":"passed","reports":len(rows),"scored_file_documents":inputs,"protected_pairs":protected,
             "ranking_checks":len(rows)*len(values),"historical500_overlap":0,"new_8B_calls":0,
             "checks":["disjoint identity sets","frozen component weights","label-free input keys and hashes", "finite scores and context limit",
                       "complete identical candidate pools","independent AP/MRR/Top-k","head membership and unchanged tail","protected precedence pairs"],
             "script_sha256":sha(Path(__file__)),"summary_sha256":sha(out/"summary.json")}
    save(out / "independent_audit.json", saved)
    print(json.dumps(saved,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    audit(args.run.resolve())
