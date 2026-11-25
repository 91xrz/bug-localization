"""Exclude official Lite and Verified issues (and shared base snapshots) from custom training."""
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from datasets import load_from_disk
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
SETS = {"Lite": ("b0dde1093fe417d83b7184254edf8199c1f0dff5", 300),
        "Verified": ("78f471bf655a3137b2e8a75af1501690ec009ec3", 500)}


def body_key(text):
    return sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def run(out):
    local = load_from_disk(str(ROOT/"datasets/swe_bench_formatted/test"))
    commits, ids, bodies = (defaultdict(set) for _ in range(3))
    for i, row in enumerate(local):
        repo = row["repo_name"].lower()
        commits[(repo, row["before_fix_sha"])].add(i)
        bodies[(repo, body_key(row["body"] or ""))].add(i)
        if row["pull_url"]:
            num = urlparse(row["pull_url"]).path.rstrip("/").split("/")[-1]
            if num.isdigit():
                ids[repo.replace("/", "__")+"-"+num].add(i)
    reserved = set()
    details = {}
    for name, (rev, count) in SETS.items():
        url = f"https://huggingface.co/datasets/SWE-bench/SWE-bench_{name}/resolve/{rev}/data/test-00000-of-00001.parquet"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        table = pq.read_table(BytesIO(data), columns=["repo", "instance_id", "base_commit", "problem_statement"])
        if len(table) != count:
            raise ValueError(f"Unexpected {name} size")
        rows = {}
        for official in table.to_pylist():
            ident, repo = official["instance_id"].lower(), official["repo"].lower()
            by_id = ids[ident]
            by_sha = commits[repo, official["base_commit"]]
            by_body = bodies[repo, body_key(official["problem_statement"])]
            matches = by_id | by_sha | by_body
            if not matches or ident in rows:
                raise ValueError(f"Unmapped or duplicate official instance: {ident}")
            rows[ident] = {"base_commit":official["base_commit"], "dataset_rows":sorted(matches),
                           "by_pr":sorted(by_id), "by_sha":sorted(by_sha), "by_body":sorted(by_body)}
            reserved.update(matches)
        details[name] = {"url":url, "parquet_sha256":sha256(data).hexdigest(), "n":len(rows), "instances":rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump({"dataset_rows":sorted(reserved), "rule":"union of PR identity, repository + base SHA, or repository + normalized body",
                   "benchmarks":details}, f, indent=2)
        f.write("\n")
    print("Reserved", len(reserved), "local reports; mapped all official instances.")


if __name__ == "__main__":
    run(ROOT/"results/c_lora/round3/reserved_lite_ids.json")
