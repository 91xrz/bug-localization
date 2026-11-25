"""Compute multilingual E, optionally fuse complete A/B rankings, from JSON."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bug_localization.structural_matching import feature_identity, rank_candidates, score_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    allowed = {"report", "sources", "a", "b", "alpha", "anchor_weight", "language_overrides"}
    if set(data) - allowed or not {"report", "sources"} <= data.keys():
        raise ValueError("Require report/sources, optional a/b/weights/language_overrides; no evaluation fields")
    if not isinstance(data["report"], str) or not isinstance(data["sources"], dict):
        raise ValueError("report must be text and sources must map relative paths to source text")
    overrides = data.get("language_overrides")
    if "a" in data or "b" in data:
        result = rank_candidates(data["report"], data["a"], data["b"], data["sources"],
            alpha=data.get("alpha", .5), anchor_weight=data.get("anchor_weight", .5), language_overrides=overrides)
    else:
        scores, matches = score_sources(data["report"], data["sources"], language_overrides=overrides)
        result = {"anchor": scores, "matches": matches}
    result.update(identity=feature_identity(), input_sha256=hashlib.sha256(raw).hexdigest(),
                  language_overrides=overrides or {})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "files": len(result["anchor"]),
                      "nonzero": sum(v > 0 for v in result["anchor"].values())}))


if __name__ == "__main__":
    main()
