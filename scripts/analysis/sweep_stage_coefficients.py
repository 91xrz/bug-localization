"""Historical AB/C coefficient diagnostics; never label these data GHUB/GHRB.

AB: select on the existing development500, freeze, evaluate remaining1794.
C: fixed new-AB pool, descriptive grid and leave-one-project-out selection.
These are separate stage studies, NOT a four-parameter end-to-end search.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bug_localization.lightweight_hypothesis import normalize
from bug_localization.bm25 import ranking_metrics
from bug_localization.paired_statistics import inference

AB = ROOT / "results/b_lightweight/20261122_v1"
C = ROOT / "results/ab_structure_newc/20261123_fixed500"
KEYS = ("MAP@10", "MRR@10", "Top1", "Top10", "Recall@10", "Top50")
AB_GRID = list(itertools.product([i / 20 for i in range(21)],
                                [0, .1, .2, .25, .3, .4, .5, .6, .75, 1, 1.5, 2]))
C_GRID = list(itertools.product([i / 20 for i in range(21)],
                               [0, .25, .5, .75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 64]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, expected=None):
    raw = path.read_bytes()
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Changed source: {path}")
    return json.loads(raw)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def metric_matrix(orders, relevant, gold_count):
    """Independent vector metrics, AP denominator includes ALL annotated files."""
    hit = relevant[orders[:, :10]]
    ranks = np.arange(1, hit.shape[1] + 1)
    ap = (hit.cumsum(axis=1) * hit / ranks).sum(axis=1) / gold_count
    rr = np.max(hit / ranks, axis=1)
    return np.column_stack((ap, rr, hit[:, 0], hit.any(axis=1),
                            hit.sum(axis=1) / gold_count,
                            relevant[orders[:, :50]].any(axis=1))).astype(float)


def check_metric(order, gold, values):
    m = ranking_metrics(order, set(gold), 10)
    expected = [m[k] for k in ("average_precision_at_k", "reciprocal_rank_at_k",
                               "top_1", "top_10", "recall_at_k")]
    expected.append(float(bool(set(order[:50]) & set(gold))))
    np.testing.assert_allclose(values, expected, atol=1e-12, rtol=0)


def choose(values, grid, baseline):
    means = values[:, :, 0].mean(axis=1)
    # Exact ties prefer the registered setting, then lower coefficient magnitude.
    return min(range(len(grid)), key=lambda i: (-means[i], grid[i] != baseline,
                                               sum(grid[i]), grid[i]))


def summarize(values, base):
    return {"n": len(values), **dict(zip(KEYS, values.mean(axis=0).tolist())),
            "corrected_top1": int((values[:, 2] > base[:, 2]).sum()),
            "broken_top1": int((values[:, 2] < base[:, 2]).sum()),
            "ap_wins": int((values[:, 0] > base[:, 0] + 1e-12).sum()),
            "ap_losses": int((values[:, 0] < base[:, 0] - 1e-12).sum())}


def grid_csv(path, grid, values, base, names):
    rows = [dict(zip(names, pair)) | summarize(values[i], base) for i, pair in enumerate(grid)]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ab_case(meta, grid, checkpoints):
    pos = meta["position"]
    row = read(AB / "cases" / f"{pos:04d}.json", checkpoints[str(pos)]["case"])
    for key in ("dataset_row", "repository", "issue_id", "before_fix_sha"):
        if row[key] != meta[key]:
            raise ValueError("AB identity mismatch")
    if row["status"] != "ok":
        return np.zeros((len(grid), len(KEYS))), [[] for _ in grid]
    inp = read(AB / "inputs" / f"{pos:04d}.json", checkpoints[str(pos)]["prepared_inputs"])
    if inp["sample"] != meta:
        raise ValueError("AB input identity mismatch")
    a, b = normalize(inp["a"]), normalize(inp["b"])
    ar = {r["file_path"]: r["rank"] for r in inp["a"]}
    br = {r["file_path"]: r["rank"] for r in inp["b"]}
    pool = {r["file_path"] for rs in (inp["a"], inp["b"]) for r in rs[:50]}
    paths = sorted(pool, key=lambda p: (br[p], ar[p], p))
    features = np.array([[a[p], b[p], inp["anchor"][p]] for p in paths])
    alpha, weight = np.array(grid).T
    scores = alpha[:, None] * features[:, 0] + (1-alpha[:, None]) * features[:, 1] + weight[:, None] * features[:, 2]
    orders = np.argsort(-scores, axis=1, kind="stable")[:, :50]
    gold = set(row["ground_truth"])
    values = metric_matrix(orders, np.array([p in gold for p in paths]), len(gold))
    ranks = [[paths[i] for i in order] for order in orders]
    for index in {0, len(grid)-1, grid.index((.5, .5))}:
        check_metric(ranks[index], gold, values[index])
    if ranks[grid.index((.5, .5))] != row["methods"]["score_a0.5_e0.5"]["ranking"]:
        raise ValueError("Registered AB ranking not reproduced")
    return values, ranks


def run_ab(out):
    selection = read(AB / "selection.json")
    checkpoints = read(ROOT / "results/baselines/20261122_ab_structure_v1/source_checkpoints.json")
    dev = set(selection["development_rows"])
    train = [r for r in selection["samples"] if r["dataset_row"] in dev]
    test = [r for r in selection["samples"] if r["dataset_row"] not in dev]
    assert (len(train), len(test)) == (500, 1794)
    base_i = AB_GRID.index((.5, .5))
    values = np.stack([ab_case(r, AB_GRID, checkpoints)[0] for r in train], axis=1)
    selected = choose(values, AB_GRID, (.5, .5))
    chosen = AB_GRID[selected]
    save(out / "ab_frozen_choice.json", {"alpha": chosen[0], "anchor_weight": chosen[1],
        "selection": "maximum development500 MAP@10, no risk constraint", "validation_loaded": False})
    grid_csv(out / "ab_development_grid.csv", AB_GRID, values, values[base_i], ("alpha", "anchor_weight"))
    print("AB frozen", chosen, flush=True)
    validation_grid = [( .5, .5)] if chosen == (.5, .5) else [(.5, .5), chosen]
    test_values, changed_pools = [], 0
    for r in test:
        v, orders = ab_case(r, validation_grid, checkpoints)
        test_values.append(v)
        changed_pools += set(orders[0]) != set(orders[-1])
    vv = np.stack(test_values, axis=1)
    result = {"choice": dict(zip(("alpha", "anchor_weight"), chosen)),
        "development_baseline": summarize(values[base_i], values[base_i]),
        "development_selected": summarize(values[selected], values[base_i]),
        "validation_baseline": summarize(vv[0], vv[0]), "validation_selected": summarize(vv[-1], vv[0]),
        "validation_changed_top50_sets": changed_pools,
        "validation_inference": inference(vv[-1, :, 0]-vv[0, :, 0], [r["repository"] for r in test],
                                           seed=20261124, repetitions=5000)}
    save(out / "ab_summary.json", result)
    save(out / "ab_validation_per_report.json", [{**r, "baseline": dict(zip(KEYS, v[0].tolist())),
        "selected": dict(zip(KEYS, v[-1].tolist()))} for r, v in zip(test, test_values)])
    return result


def run_c(out):
    plans = read(C / "plan.json")
    audit = read(C / "independent_audit.json")
    base_i = C_GRID.index((.25, 4))
    values, diagnostics, order_changes, per_report = [], [], [], []
    for plan in plans:
        f = read(C / plan["feature_file"], plan["feature_sha256"])
        pos = plan["position"]
        published = read(C / "cases" / f"{pos:03d}.json", audit["case_sha256"][f"{pos:03d}.json"])
        if f["status"] != "ok":
            values.append(np.zeros((len(C_GRID), len(KEYS))))
            order_changes.append(np.zeros(len(C_GRID), dtype=int))
            per_report.append({"dataset_row": plan["dataset_row"], "status": "failed", "rankings": {}})
            continue
        record = read(C / "records" / f"{pos:03d}.json")
        fresh = read(C / record["path"], record["sha256"])
        joint = {p: s["score"] for p, s in f["cached_joint"].items()}
        joint.update({s["file_path"]: s["score"] for s in fresh["scores"]})
        paths = f["new_base"]
        z = np.array([max(f["chunks"][p]) for p in paths[:10]])
        u = np.array([joint[p] for p in paths[:10]])
        prior = np.array([f["prior"][p] for p in paths[:10]])
        beta, lam = np.array(C_GRID).T
        score = (1-beta[:, None])*z + beta[:, None]*u + lam[:, None]*prior
        if not np.isfinite(score).all():
            raise ValueError("Nonfinite score")
        head = np.argsort(-score, axis=1, kind="stable")
        orders = np.column_stack((head, np.tile(np.arange(len(z), len(paths)), (len(C_GRID), 1))))
        gold = set(f["gold"])
        vv = metric_matrix(orders, np.array([p in gold for p in paths]), len(gold))
        current = [paths[i] for i in orders[base_i]]
        if current != published["orders"]["new_pool_new_C"]:
            raise ValueError("Registered C ranking not reproduced")
        # Check every grid point against the production metric and head/tail invariants.
        for i, order in enumerate(orders):
            rank = [paths[j] for j in order]
            check_metric(rank, gold, vv[i])
            assert set(rank[:10]) == set(paths[:10]) and rank[10:] == paths[10:]
        values.append(vv)
        order_changes.append(np.any(head != head[base_i], axis=1).astype(int))
        q = .75*z + .25*u
        pair_x, pair_y = np.triu_indices(len(z), 1)
        dq, dp = q[pair_x]-q[pair_y], 4*(prior[pair_x]-prior[pair_y])
        active = np.abs(dq)+np.abs(dp) > 0
        diagnostics.append({"dataset_row": plan["dataset_row"], "model_range": float(np.ptp(q)),
            "weighted_prior_range": float(np.ptp(4*prior)),
            "active_pairs": int(active.sum()), "prior_larger_pairs": int((np.abs(dp[active]) > np.abs(dq[active])).sum()),
            "opposing_pairs": int((dp*dq < 0).sum()),
            "prior_overrules_model_pairs": int(((dp*dq < 0) & (np.abs(dp) > np.abs(dq))).sum())})
        per_report.append({"dataset_row": plan["dataset_row"], "status": "ok", "gold": sorted(gold),
            "rankings": {f"beta{b:g}_lambda{l:g}": [paths[j] for j in orders[i]]
                         for i, (b, l) in enumerate(C_GRID) if b == .25 and l in (0, 1, 2, 4, 8, 16)}})
    values = np.stack(values, axis=1)
    order_changes = np.stack(order_changes, axis=1)
    grid_csv(out / "c_descriptive_grid.csv", C_GRID, values, values[base_i], ("beta", "prior_weight"))
    best = choose(values, C_GRID, (.25, 4))
    projects = np.array([r["repository"] for r in plans])
    oof = np.zeros_like(values[base_i])
    folds = []
    for project in sorted(set(projects)):
        train, test = projects != project, projects == project
        chosen = choose(values[:, train], C_GRID, (.25, 4))
        oof[test] = values[chosen, test]
        folds.append({"held_out_project": project, "n": int(test.sum()),
            "beta": C_GRID[chosen][0], "prior_weight": C_GRID[chosen][1],
            "held_out_baseline": summarize(values[base_i, test], values[base_i, test]),
            "held_out_selected": summarize(values[chosen, test], values[base_i, test])})
    fixed_beta = [{"prior_weight": l, **summarize(values[i], values[base_i]),
                   "changed_top10_order_vs_lambda4": int(order_changes[i].sum())}
                  for i, (b, l) in enumerate(C_GRID) if b == .25]
    result = {"baseline": summarize(values[base_i], values[base_i]),
        "descriptive_best_on_all500": {"beta": C_GRID[best][0], "prior_weight": C_GRID[best][1],
            **summarize(values[best], values[base_i])}, "fixed_beta_025": fixed_beta,
        "leave_one_project_out": {"selected": summarize(oof, values[base_i]), "folds": folds,
            "limitation": "Historical data and original coefficients already observed these projects; not a blind test. OOF fits overlap, so no naive fixed-model significance claim."},
        "score_scale": {"successful_reports": len(diagnostics),
            "median_model_head_range": float(np.median([r["model_range"] for r in diagnostics])),
            "median_4prior_head_range": float(np.median([r["weighted_prior_range"] for r in diagnostics])),
            **{k: sum(r[k] for r in diagnostics) for k in ("active_pairs", "prior_larger_pairs", "opposing_pairs", "prior_overrules_model_pairs")}}}
    save(out / "c_summary.json", result)
    save(out / "c_score_scale_per_report.json", diagnostics)
    save(out / "c_per_report.json", per_report)
    np.savez_compressed(out / "c_grid_metrics.npz", values=values, grid=np.array(C_GRID), projects=projects)
    print("C descriptive best", C_GRID[best], "LOPO MAP", oof[:, 0].mean(), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dependencies = [Path(__file__), AB / "selection.json", AB / "independent_audit.json",
                    C / "protocol.json", C / "plan.json", C / "independent_audit.json"]
    sources = {str(p): sha(p) for p in dependencies}
    for source in (AB, C):
        assert read(source / "independent_audit.json")["status"] == "passed"
    save(out / "protocol.json", {"dataset": "Existing SWE-bench formatted data; NOT GHUB/GHRB",
        "ab_formula": "p=alpha*N(A)+(1-alpha)*N(B)+gamma*E", "ab_grid": AB_GRID,
        "c_formula": "s=(1-beta)*z+beta*u+lambda*p", "c_grid": C_GRID,
        "primary_metric": "MAP@10, denominator all labeled files", "ab_candidate_policy": "A50 union B50, rerank and take50 for each setting",
        "c_candidate_policy": "Frozen current AB Top50; reorder original Max C Top10 only; alpha=gamma=.5",
        "selection": "AB development500 only then frozen validation1794; C descriptive all500 and leave-one-project-out",
        "limits": ["Separate stage tuning, not joint AB+C optimization", "Observed historical data, not an independent new benchmark",
                   "GHUB identity/location unresolved; no GHUB experiment claimed", "No global optimum claim for finite grid",
                   "A changed AB pool needs missing C scores before evaluating a combined pipeline"],
        "files_sha256": sources, "model_calls": 0})
    ab = run_ab(out)
    c = run_c(out)
    assert all(sha(Path(p)) == digest for p, digest in sources.items())
    save(out / "verification.json", {"status": "passed", "ab_reports": 2294, "c_reports": 500,
        "ab_grid_size": len(AB_GRID), "c_grid_size": len(C_GRID), "model_calls": 0,
        "checks": ["Archived input/result hashes", "Registered AB/C rankings exactly reproduced",
                   "AB frozen choice written before validation", "C all grid metrics independently recomputed",
                   "C Top10 membership/tail unchanged at every coefficient", "Upstream failures retained as zero",
                   "Source metadata unchanged"], "protocol_sha256": sha(out / "protocol.json")})
    print(json.dumps({"ab": ab["choice"], "c": c["descriptive_best_on_all500"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
