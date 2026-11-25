"""Independently audit file-Max rankings and fixed symptom-proxy pairs."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
from statistics import fmean


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verified(path, checksum):
    data = path.read_bytes()
    if sha256(data).hexdigest() != checksum:
        raise ValueError(f"Checksum mismatch: {path}")
    return json.loads(data)


def looks_like_test(path):
    path = PurePosixPath(path.lower())
    return (bool({"test", "tests", "testing"} & set(path.parts[:-1]))
            or path.stem.startswith("test_") or path.stem.endswith("_test"))


def score_map(source, result):
    expected = [(r["file_path"], r["chunk_id"]) for r in source["chunks"]]
    scored = result["scored_chunks"]
    actual = [(r["file_path"], r["chunk_id"]) for r in scored]
    if len(set(expected)) != len(expected) or actual != expected:
        raise ValueError("Missing, duplicate or reordered chunk scores")
    maxima = {}
    for row in scored:
        val, path = row["score"], row["file_path"]
        if not math.isfinite(val):
            raise ValueError("Non-finite model score")
        maxima[path] = max(val, maxima.get(path, -math.inf))
    original = [r["file_path"] for r in source["original_order"]]
    if len(set(original)) != len(original) or set(original) != set(maxima):
        raise ValueError("Candidate pool differs between retrieval and reranking")
    desired = sorted(original, key=lambda p: -maxima[p])
    observed = result["ranking"]
    if ([r["file_path"] for r in observed] != desired or
            [r["score"] for r in observed] != [maxima[p] for p in desired]):
        raise ValueError("Ranking differs from stable file-Max reconstruction")
    return maxima, desired


def ap10(order, gold):
    hits = [i for i, p in enumerate(order[:10], 1) if p in gold]
    return sum(i / rank for i, rank in enumerate(hits, 1)) / len(gold)


def pair_counts(positives, negatives, scores):
    outcomes = Counter()
    for p in positives:
        for n in negatives:
            a, b = scores[p], scores[n]
            outcomes["win" if a > b else "loss" if a < b else "tie"] += 1
    return dict(outcomes)


def accuracy(counts):
    total = sum(counts.values())
    return (counts.get("win", 0) + 0.5 * counts.get("tie", 0)) / total if total else None


def run(root, adapted="lora"):
    manifest = read(root / "manifest.json")
    labels = verified(root / "labels.json", manifest["labels_sha256"])
    arms = ("base", adapted)
    runs = {m: read(root / m / "run.json") for m in arms}
    protocols = {m: verified(root / m / "protocol.json", r["protocol_sha256"]) for m, r in runs.items()}
    if (any(protocols["base"][k] != protocols[adapted][k] for k in ("model", "config", "manifest_sha256"))
            or protocols["base"]["adapter_sha256"] or not protocols[adapted]["adapter_sha256"]):
        raise ValueError("The two model arms must differ only by an identified adapter")
    if len(labels) != manifest["n"] or any(len(r["case_sha256"]) != len(labels) for r in runs.values()):
        raise ValueError("Incomplete evaluation")
    cases = []
    for i, label in enumerate(labels):
        name = f"{label['position']:03}.json"
        input_digest = manifest["input_sha256"][i]
        source = verified(root / "inputs" / name, input_digest)
        models = {m: verified(root / m / "cases" / name, r["case_sha256"][i]) for m, r in runs.items()}
        if any(r["input_sha256"] != input_digest for r in models.values()):
            raise ValueError("Scores were created with a different input")
        if any(x["id"] != label["id"] for x in [source, *models.values()]):
            raise ValueError(f"Identity mismatch for {name}")
        gold = set(label["gold"])
        if not gold:
            raise ValueError(f"Empty gold for {name}")
        case = {"id": label["id"], "repository": label["repository"], "metrics": {}, "pairs": {}}
        order = [r["file_path"] for r in source.get("original_order", [])]
        pos = [p for p in order if p in gold]
        neg = [p for p in order if p not in gold]
        strata = {"first5_unmodified": neg[:5], "test_like_unmodified": [p for p in neg if looks_like_test(p)]}
        for model, result in models.items():
            if result["status"] != "ok":
                if source["source_status"] == "ok" or result["ranking"] or result["scored_chunks"]:
                    raise ValueError("Unexpected inference failure or nonempty failed ranking")
                case["metrics"][model] = {"ap": 0.0, "status": result["status"]}
                continue
            scores, ranking = score_map(source, result)
            case["metrics"][model] = {"ap": ap10(ranking, gold), "status": "ok"}
            for key, negatives in strata.items():
                case["pairs"].setdefault(key, {})[model] = pair_counts(pos, negatives, scores)
        cases.append(case)
    summary = {"n": len(cases), "map10": {m: fmean(c["metrics"][m]["ap"] for c in cases)
                                         for m in arms}, "slices": {}}
    summary["project_delta_map10"] = {
        repo: fmean(c["metrics"][adapted]["ap"] - c["metrics"]["base"]["ap"]
                    for c in cases if c["repository"] == repo)
        for repo in sorted({c["repository"] for c in cases})}
    for key in ("first5_unmodified", "test_like_unmodified"):
        paired = [c["pairs"][key] for c in cases if all(c["pairs"].get(key, {}).get(m)
                                                       for m in arms)]
        summary["slices"][key] = {"reports": len(paired), "pairs": sum(sum(x["base"].values()) for x in paired)}
        for m in arms:
            summary["slices"][key][m] = fmean(accuracy(x[m]) for x in paired) if paired else None
    return {"summary": summary, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--adapter-arm", choices=("lora", "lexical", "hard"), default="lora")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.evaluation, args.adapter_arm)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
