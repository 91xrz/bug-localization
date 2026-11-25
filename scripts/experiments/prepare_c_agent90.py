"""Freeze the previously discussed 90 cases before joint or verification inference."""
from pathlib import Path
from pilot_c3_joint import PROJECT, C3, read, write_json, sha256_file

ROOT = PROJECT/"results/c_agent90/round1"
SOURCE = PROJECT/"results/c_lora/round2/evaluation/selection.json"


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    rows = read(SOURCE)
    old = {r["position"]:r for r in read(C3/"samples.json")}
    selected = []
    for row in rows:
        pos = row["c3_position"]
        if any(row[k] != old[pos][k] for k in ("repository","issue_id","before_fix_sha","dataset_row")):
            raise ValueError("Historical identity mismatch")
        selected.append({**row,"position":pos})
    if len(set(r["position"] for r in selected)) != 90:
        raise ValueError("Incomplete cohort")
    write_json(ROOT/"selection.json", {"samples":selected,"source_sha256":sha256_file(SOURCE)})
    write_json(ROOT/"design.json", {
        "cohort":"all historical round2 SWE90, known BLAgent performance; no filtering",
        "selection_sha256":sha256_file(ROOT/"selection.json"),
        "gate":"C3 first != valid ordinary 8B first; otherwise unchanged",
        "primary":"active minus fixed paired AP@10 on all 90; compare active to joint and local BLAgent separately",
        "actions":"Two bounded source lookups in the two disputed pre-fix files; fixed and active receive same menu, model planner/question, judge and caps. Fixed uses deterministic report/code match per file; active uses planner choices.",
        "decision":"On verifiable cited evidence prefer a disputed path or abstain. Move preferred C3 challenger to first, preserving all other relative positions. Invalid or insufficient review retains ordinary 8B; failed joint stays failed for both.",
        "budget":"shared one planning call and one judging call per arm on triggered reports; same individual caps, measured actual tokens. No tuning or retries on outcomes.",
        "statistics":"10000 paired bootstrap draws within project; descriptive secondary metrics; report full denominator, recovered/lost top1, call/token costs"})


if __name__ == "__main__":
    prepare()
