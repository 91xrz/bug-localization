"""Vary only the new C reranking prefix, using immutable saved file scores.

K is the original FULL prefix being reordered, never the evaluation cutoff.
Both historical cohorts are reported separately; no weights or windows are fit.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/experiments"), str(ROOT / "scripts/analysis"), str(ROOT / "src")]
from run_c_evidence_prior_validation import load, rank
from run_conservative_reranking import interval, metric, save, sha, read
from audit_c_revision import metrics as independent_metrics

DEPTHS = (10, 15, 20, 30, 40, 50)
SOURCES = {
    "historical500": ROOT / "results/c_evidence_prior/20261123_full500",
    "separate180": ROOT / "results/c_evidence_prior/20261123_validation180",
}


def summarize(rows, values, baseline):
    n = len(rows)
    return {"n": n, **{k: sum(v[k] for v in values) / n for k in values[0]},
        "ap_wins_vs_K10": sum(v["MAP@10"] > b["MAP@10"] + 1e-12 for v,b in zip(values,baseline)),
        "ap_losses_vs_K10": sum(v["MAP@10"] < b["MAP@10"] - 1e-12 for v,b in zip(values,baseline)),
        "corrected_top1_vs_K10": sum(v["Top1"] > b["Top1"] for v,b in zip(values,baseline)),
        "broken_top1_vs_K10": sum(v["Top1"] < b["Top1"] for v,b in zip(values,baseline)),
        "rescued_top10_vs_K10": sum(v["Top10"] > b["Top10"] for v,b in zip(values,baseline)),
        "broken_top10_vs_K10": sum(v["Top10"] < b["Top10"] for v,b in zip(values,baseline))}


def table(summary):
    lines = ["| 调整原前K | MAP@10 | 相对K10的ΔMAP | MRR@10 | Top1 | Top5 | Top10 | 前十找回 / 丢失 | AP胜 / 平 / 负 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in DEPTHS:
        m = summary[str(k)]
        ties = m["n"] - m["ap_wins_vs_K10"] - m["ap_losses_vs_K10"]
        lines.append(f"| {k} | {m['MAP@10']:.6f} | {m['MAP@10']-summary['10']['MAP@10']:+.6f} | "
                     f"{m['MRR@10']:.6f} | {m['Top1']:.2%} | {m['Top5']:.2%} | {m['Top10']:.2%} | "
                     f"{m['rescued_top10_vs_K10']} / {m['broken_top10_vs_K10']} | {m['ap_wins_vs_K10']} / {ties} / {m['ap_losses_vs_K10']} |")
    return "\n".join(lines)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    source_meta = {}
    ids = {}
    for name, source in SOURCES.items():
        p = load(source)
        if read(source / "independent_audit.json")["status"] != "passed":
            raise ValueError("Source has not passed independent audit")
        source_meta[name] = {"path": str(source), "files_sha256": {f: sha(source / f) for f in
            ("protocol.json", "labels.json", "rankings.json", "summary.json", "verification.json", "independent_audit.json")}}
        if (p["packet_weight"], p["prior_weight"]) != (.25, 4.):
            raise ValueError("Unexpected source weights")
        ids[name] = {s["dataset_row"] for s in p["sample_plan"]}
    if len(ids["historical500"]) != 500 or len(ids["separate180"]) != 180 or ids["historical500"] & ids["separate180"]:
        raise ValueError("Unexpected cohort membership/overlap")
    protocol = {
        "depths": DEPTHS, "primary_question": "K20 versus K10; other depths are sensitivity checks",
        "evaluation_cutoff": 10, "prefix_source": "Original FULL max-score ranking before the new combination",
        "score": ".75*max_chunk_logit + .25*joint_file_logit + 4*initial_score; fixed",
        "ties": "Original FULL rank", "tail": "Preserve original order after rank K",
        "scope": "Reuse saved independent per-file scores over all50; no new model calls or scoring-context changes",
        "failure_policy": "Keep every original report; unavailable source remains empty/zero",
        "selection_policy": "No data-dependent choice of K; report the full grid and both cohorts separately",
        "statistics": "Paired AP bootstrap and randomization; Holm over all 10 nonbaseline cohort/depth contrasts",
        "limitations": ["Both cohorts are historical and already examined; 180 is disjoint, not a fresh blind test",
                        "All original candidates have already been scored; this does not measure online latency at each K",
                        "No evidence is available beyond the existing50 candidates",
                        "Label-based candidate-prefix ceilings are diagnostics, never ranking inputs"],
        "sources": source_meta,
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
            (Path(__file__), ROOT / "scripts/experiments/run_c_evidence_prior_validation.py",
             ROOT / "scripts/experiments/run_conservative_reranking.py", ROOT / "scripts/analysis/audit_c_revision.py")},
    }
    save(out / "protocol.json", protocol)
    summaries, stats, diagnostics, all_rows, projects = {}, {}, {}, {}, {}
    checks = Counter()
    for cohort, source in SOURCES.items():
        p = read(source / "protocol.json")
        rows = read(source / "labels.json")
        prior_rankings = read(source / "rankings.json")
        prior_summary = read(source / "summary.json")
        verification = read(source / "verification.json")
        if prior_rankings["dataset_rows"] != [r["dataset_row"] for r in rows] or len(rows) != len(p["sample_plan"]):
            raise ValueError("Saved result and label order differ")
        orders = {k: [] for k in DEPTHS}
        values = {k: [] for k in DEPTHS}
        diagnostic = {k: Counter() for k in DEPTHS}
        entrants = {k: Counter() for k in DEPTHS}
        case_rows = []
        for i, (row, meta) in enumerate(zip(rows, p["sample_plan"])):
            path = source / "cases" / f"{row['dataset_row']:04d}.json"
            if sha(path) != verification["result_sha256"][str(path.relative_to(source))]:
                raise ValueError("Saved file score checksum changed")
            result = read(path)
            if (result["dataset_row"] != row["dataset_row"] or result["input_sha256"] != meta["input_sha256"]
                    or result["protocol_sha256"] != source_meta[cohort]["files_sha256"]["protocol.json"]):
                raise ValueError("Saved inference provenance mismatch")
            scores = {s["file_path"]: s["score"] for s in result["scores"]}
            if len(scores) != len(result["scores"]):
                raise ValueError("Duplicate file scores")
            base = row["base"]
            ties = {f: j for j,f in enumerate(base)}
            fixed_scores = {f: .75*max(row["chunks"][f])+.25*scores[f]+4*row["prior"][f] for f in base}
            gold = set(row["gold"])
            case = {k: row[k] for k in ("dataset_row", "repository", "issue_id", "before_fix_sha")}
            case.update(ground_truth=row["gold"], source_failed=not bool(base), depths={})
            for k in DEPTHS:
                order = rank(row, scores, .25, 4., head=k)
                independent_order = sorted(base[:k], key=lambda f: (-fixed_scores[f], ties[f])) + base[k:]
                if order != independent_order or set(order[:k]) != set(base[:k]) or order[k:] != base[k:]:
                    raise ValueError("Fixed score/prefix/tail invariant failed")
                m = independent_metrics(order, gold)
                other = metric(order, gold)
                for a,b in (("MAP@10", "average_precision_at_k"), ("MRR@10", "reciprocal_rank_at_k"),
                            ("Top1", "top_1"), ("Top5", "top_5"), ("Top10", "top_10"), ("Recall@10", "recall_at_k")):
                    if abs(m[a]-other[b]) > 1e-12:
                        raise ValueError("Independent metric implementation differs")
                orders[k].append(order); values[k].append(m)
                old = orders[10][-1]
                if [f for f in order if f in set(base[:10])] != old[:10]:
                    raise ValueError("Expanding K altered the relative order of the original top10")
                entered = [f for f in order[:10] if f not in old[:10]]
                displaced = [f for f in old[:10] if f not in order[:10]]
                delta = m["MAP@10"]-values[10][-1]["MAP@10"]
                if not any(f in gold for f in entered) and delta > 1e-12:
                    raise ValueError("AP improved without a new relevant entrant under unchanged old-head order")
                diag = diagnostic[k]
                diag["changed_top10_membership_reports"] += bool(entered)
                diag["changed_top10_order_reports"] += order[:10] != old[:10]
                diag["changed_top1_reports"] += bool(order) and order[0] != old[0]
                diag["entered_file_instances"] += len(entered)
                diag["entered_gold_file_instances"] += sum(f in gold for f in entered)
                diag["displaced_gold_file_instances"] += sum(f in gold for f in displaced)
                diag["reports_with_new_gold_in_top10"] += any(f in gold for f in entered)
                diag["candidate_prefix_hit_ceiling_sum"] += bool(gold & set(base[:k]))
                diag["candidate_prefix_AP10_ceiling_sum"] += min(10,len(gold & set(base[:k]))) / len(gold)
                for f in entered:
                    entrants[k][ties[f]+1] += 1
                case["depths"][str(k)] = {"ranking": order, "metrics": m, "delta_ap_vs_K10": delta,
                    "entered_top10": [{"file_path": f, "original_rank": ties[f]+1, "gold": f in gold} for f in entered],
                    "displaced_from_top10": [{"file_path": f, "gold": f in gold} for f in displaced]}
                checks["rankings_and_metrics_checked"] += 1
            if orders[10][-1] != prior_rankings["orders"]["combined_head10"][i] or orders[50][-1] != prior_rankings["orders"]["combined_full50"][i]:
                raise ValueError("K10/K50 failed to exactly reproduce previously validated results")
            checks["archived_endpoint_rankings_reproduced"] += 2
            case_rows.append(case)
        summary = {str(k): summarize(rows, values[k], values[10]) for k in DEPTHS}
        for k,name in ((10,"combined_head10"), (50,"combined_full50")):
            for key in values[k][0]:
                if abs(summary[str(k)][key]-prior_summary[name][key]) > 1e-12:
                    raise ValueError("Archived endpoint summary changed")
        for k in DEPTHS[1:]:
            stats[f"{cohort}/K{k}_minus_K10"] = interval(rows, [a["MAP@10"]-b["MAP@10"] for a,b in zip(values[k],values[10])])
        summaries[cohort] = summary
        diagnostics[cohort] = {str(k): {**dict(diagnostic[k]),
            "candidate_prefix_hit_ceiling": diagnostic[k]["candidate_prefix_hit_ceiling_sum"]/len(rows),
            "candidate_prefix_AP10_ceiling": diagnostic[k]["candidate_prefix_AP10_ceiling_sum"]/len(rows),
            "original_ranks_of_entering_files": dict(sorted(entrants[k].items()))} for k in DEPTHS}
        all_rows[cohort] = case_rows
        projects[cohort] = {repo:{str(k):summarize([rows[i] for i in ix], [values[k][i] for i in ix], [values[10][i] for i in ix]) for k in DEPTHS}
            for repo in sorted({r["repository"] for r in rows})
            for ix in [[i for i,r in enumerate(rows) if r["repository"]==repo]]}
    previous = 0.
    for i,key in enumerate(sorted(stats, key=lambda k:stats[k]["paired_randomization_p"])):
        previous = max(previous,min(1.,(len(stats)-i)*stats[key]["paired_randomization_p"]))
        stats[key]["holm_p_all_depth_cohort_contrasts"] = previous
    save(out / "summary.json", summaries)
    save(out / "statistics.json", stats)
    save(out / "boundary_diagnostics.json", diagnostics)
    save(out / "per_report.json", all_rows)
    save(out / "project_summary.json", projects)
    save(out / "review_selection.json", {g:{str(k):[
        {key:row[key] for key in ("dataset_row","repository","issue_id","ground_truth")} | {"K10": row["depths"]["10"], f"K{k}": row["depths"][str(k)]}
        for row in rows if abs(row["depths"][str(k)]["delta_ap_vs_K10"])>1e-12]
        for k in (20,50)} for g,rows in all_rows.items()})
    with (out / "comparison.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cohort","K","n","MAP@10","MRR@10","Top1","Top5","Top10","rescued_top10_vs_K10","broken_top10_vs_K10"])
        for cohort, ss in summaries.items():
            for k,m in ss.items():
                writer.writerow([cohort,k]+[m[name] for name in ("n","MAP@10","MRR@10","Top1","Top5","Top10","rescued_top10_vs_K10","broken_top10_vs_K10")])
    for cohort,source in SOURCES.items():
        for name,digest in source_meta[cohort]["files_sha256"].items():
            if sha(source/name)!=digest:
                raise ValueError("Original result changed during analysis")
    save(out / "verification.json", {"status":"passed", **dict(checks), "cases":680,
        "cohort_overlap":0,"model_calls":0,"source_artifacts_unchanged":True,
        "prefix_source":"Original FULL, never new-score topK", "evaluation_k":10,
        "protocol_sha256":sha(out/"protocol.json"),"summary_sha256":sha(out/"summary.json")})
    text = ["# 新C调整范围：K10到K50，指标统一为MAP@10", "",
        "固定0.75×原Max＋0.25×文件联合评分＋4×初排分数，只改变原FULL排序中参与调整的前K个文件。其后尾部不变。复用全部50候选的已保存分数，没有新增模型调用，也没有改变每文件联合输入。", "",
        "500条与不重叠180条分别报告。两批都是历史数据；500曾参与参数开发，180也已经用于观察，不称新的盲测。K20是预先指定的主要问题，其他K是敏感性检查；没有选择新的最佳K。", ""]
    for cohort,ss in summaries.items():
        text.extend(["## "+cohort,"",table(ss),""])
        for k in (20,50):
            s=stats[f"{cohort}/K{k}_minus_K10"];ci=s["ci95_report_bootstrap"];pc=s["ci95_project_cluster_bootstrap"];d=diagnostics[cohort][str(k)]
            text.extend([f"K{k}−K10：ΔMAP={s['delta']:+.6f}；逐报告95%区间[{ci[0]:+.6f},{ci[1]:+.6f}]；"
                f"项目整体区间[{pc[0]:+.6f},{pc[1]:+.6f}]；十项比较Holm p={s['holm_p_all_depth_cohort_contrasts']:.6f}。", "",
                f"前十集合变化{d['changed_top10_membership_reports']}条，首位文件变化{d['changed_top1_reports']}条；"
                f"新增进入前十的文件实例{d['entered_file_instances']}个，其中标注文件{d['entered_gold_file_instances']}个，"
                f"挤出标注文件{d['displaced_gold_file_instances']}个。", ""])
    text.extend(["## 解释与边界","",
        "分数固定时，扩大K不会改变原前十文件之间的相对顺序；收益和损失来自原第11—K名的候选插入前十。只有非标注文件进入时，原正确文件只能不动或后移，AP不会提高。", "",
        "K>10取消了原前十集合保持不变的保证，必须同时查看新找回与丢失，不能只看净Top10。标签上界只表示该候选前缀内可达到的理想覆盖/AP，不参与排序。", "",
        "现有候选最多50，本轮没有对更大候选池作推断。全部文件分数此前已计算，本次CPU耗时不代表线上增加评分文件的代价。", "",
        "每组K10和K50逐例精确复现此前已审计的结果；全部4,080组排名通过独立公式、前缀/尾部和指标校验。完整来源、参数与校验和在protocol.json和verification.json中。"])
    (out/"report.md").write_text("\n".join(text)+"\n")
    for name,ss in summaries.items():
        print(name,flush=True); print(table(ss),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.output.resolve())
