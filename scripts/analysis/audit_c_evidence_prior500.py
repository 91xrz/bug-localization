"""Audit and report the complete500 experiment without changing any ranking."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
sys.path.insert(0, str(ROOT / "scripts/experiments"))
from audit_c_revision import metrics, read, save, sha
from run_c_evidence_prior_validation import load, CPU, PACKETS, FULL, ROOT as PROJECT
from run_conservative_reranking import interval

IDENTITY = ("dataset_row", "repository", "issue_id", "before_fix_sha")
METHOD_LABELS = {
    "FULL_max": "原C（FULL逐块Max）", "prior_head10": "仅初排分数修正",
    "packet_head10": "仅25%联合证据评分", "combined_head10": "最终组合：只调整前十",
    "combined_guarded": "组合＋偏序保护", "combined_full50": "组合调整全部50（消融）",
    "BLAgent": "本地BLAgent 8B",
}


def summarize(rows, values, baseline):
    n = len(rows)
    return {"n": n, **{k: sum(r[k] for r in values)/n for k in values[0]},
        "corrected_top1": sum(a["Top1"] > b["Top1"] for a,b in zip(values,baseline)),
        "broken_top1": sum(a["Top1"] < b["Top1"] for a,b in zip(values,baseline)),
        "broken_top10": sum(a["Top10"] < b["Top10"] for a,b in zip(values,baseline)),
        "ap_wins": sum(a["MAP@10"] > b["MAP@10"]+1e-12 for a,b in zip(values,baseline)),
        "ap_losses": sum(a["MAP@10"] < b["MAP@10"]-1e-12 for a,b in zip(values,baseline))}


def table(summary):
    lines = ["| 方法 | MAP@10 | MRR@10 | Top1 | Top5 | Top10 | 首位纠正 / 破坏 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in summary.items():
        lines.append(f"| {METHOD_LABELS[name]} | {row['MAP@10']:.6f} | {row['MRR@10']:.6f} | "
                     f"{row['Top1']:.2%} | {row['Top5']:.2%} | {row['Top10']:.2%} | "
                     f"{row['corrected_top1']} / {row['broken_top1']} |")
    return "\n".join(lines)


def audit(out):
    p = load(out)
    rows = read(out / "labels.json")
    ranking = read(out / "rankings.json")
    summary = read(out / "summary.json")
    stats = read(out / "statistics.json")
    verification = read(out / "verification.json")
    historical = read(CPU / "historical500.json")
    full_index = {r["dataset_row"]:r for r in read(CPU / "full.json")}
    source_index = {r["dataset_row"]:r for r in read(FULL / "FULL/samples.json")}
    if sha(out / "labels.json") != p["labels_sha256"]:
        raise ValueError("Evaluation labels changed")
    expected_ids = [r["dataset_row"] for r in historical]
    if len(rows) != 500 or len(set(expected_ids)) != 500 or [r["dataset_row"] for r in rows] != expected_ids:
        raise ValueError("Not the complete original500 in original order")
    if ranking["dataset_rows"] != expected_ids or len(p["sample_plan"]) != 500:
        raise ValueError("Ranking/sample-plan identity mismatch")
    if (p["packet_weight"],p["prior_weight"]) != (.25,4.):
        raise ValueError("Frozen component weights changed")
    separate180 = read(ROOT / "results/c_evidence_prior/20261123_validation180/labels.json")
    if set(expected_ids) & {r["dataset_row"] for r in separate180}:
        raise ValueError("Unexpected overlap with separate180")
    for worker in (0,1):
        identity = read(out / f"model{worker}.json")
        if identity["files_sha256"] != p["model"]:
            raise ValueError("Scorer checkpoint mismatch")
    counts = Counter()
    protected, score_pairs = 0, 0
    for i,(row,meta,old) in enumerate(zip(rows,p["sample_plan"],historical)):
        if row != full_index[row["dataset_row"]] or any(row[k]!=meta[k] or row[k]!=old[k] for k in IDENTITY):
            raise ValueError("Prepared labels/features/identity differ from frozen sources")
        path = out / meta["input_file"]
        inp = read(path)
        if sha(path)!=meta["input_sha256"] or set(inp)!={"dataset_row","query","packets"} or inp["dataset_row"]!=row["dataset_row"]:
            raise ValueError("Model-visible input changed or contains unexpected fields")
        result_path = out / "cases" / f"{row['dataset_row']:04d}.json"
        if sha(result_path) != verification["result_sha256"][str(result_path.relative_to(out))]:
            raise ValueError("Scoring result changed after evaluation")
        result = read(result_path)
        if (result["input_sha256"]!=meta["input_sha256"] or result["protocol_sha256"]!=sha(out/"protocol.json")
                or result["dataset_row"]!=row["dataset_row"]):
            raise ValueError("Scoring provenance mismatch")
        scores = {s["file_path"]:s["score"] for s in result["scores"]}
        base = row["base"]
        if len(scores)!=len(result["scores"]) or set(scores)!=set(base) or set(scores)!={d["file_path"] for d in inp["packets"]}:
            raise ValueError("Candidate coverage mismatch")
        if any(not math.isfinite(s["score"]) or s["input_tokens"]>p["config"]["max_tokens"] for s in result["scores"]):
            raise ValueError("Invalid score or token budget")
        score_pairs += len(scores)
        if base:
            counts["successful"] += 1
            source_path = FULL / source_index[row["dataset_row"]]["result_file"]
            if sha(source_path)!=p["source_sha256"][str(source_path)]:
                raise ValueError("Original FULL source changed")
            source = read(source_path)
            if inp["query"]!=source["query"] or set(source["ground_truth"])!=set(row["gold"]):
                raise ValueError("Query or labels differ from the source")
            chunks = defaultdict(list)
            for chunk in source["scored_chunks"]:
                chunks[chunk["file_path"]].append((chunk["chunk_id"],chunk["embedding_text"]))
            for packet in inp["packets"]:
                parts=sorted(chunks[packet["file_path"]])
                text="File: "+packet["file_path"]+"\n\n"+"\n\n".join(f"[Code fragment {j}]\n{t}" for j,(_,t) in enumerate(parts,1))
                if packet["chunk_ids"]!=[ident for ident,_ in parts] or packet["document"]!=text:
                    raise ValueError("Packet differs from the actual selected source chunks")
        else:
            counts["upstream_failed"] += 1
            if meta["source_status"]!="upstream_failed" or inp["query"] or inp["packets"] or scores:
                raise ValueError("Failure was silently changed into usable input")
        for name in ("prior_head10","packet_head10","combined_head10","combined_guarded"):
            order=ranking["orders"][name][i]
            if set(order[:10])!=set(base[:10]) or order[10:]!=base[10:]:
                raise ValueError("Head membership or tail changed")
        for name,pw,rw,head in (("prior_head10",0.,4.,10),("packet_head10",.25,0.,10),
                                ("combined_head10",.25,4.,10),("combined_full50",.25,4.,50)):
            score={f:(1-pw)*max(row["chunks"][f])+pw*scores[f]+rw*row["prior"][f] for f in base}
            expected=sorted(base[:head],key=lambda f:(-score[f],base.index(f)))+base[head:]
            if ranking["orders"][name][i]!=expected:
                raise ValueError("Frozen ranking formula does not reproduce")
        if ranking["orders"]["FULL_max"][i]!=base or ranking["orders"]["BLAgent"][i]!=row["bl"]:
            raise ValueError("Baseline changed")
        guard=ranking["orders"]["combined_guarded"][i]
        for j,a in enumerate(base[:10]):
            for b in base[j+1:10]:
                if (max(row["chunks"][a])-max(row["chunks"][b])>=p["guard"]["margin"]
                        and row["prior"][a]-row["prior"][b]>=p["guard"]["prior_margin"]):
                    if guard.index(a)>=guard.index(b):
                        raise ValueError("Evidence-protected order violated")
                    protected+=1
    if counts!={"successful":499,"upstream_failed":1}:
        raise ValueError("Unexpected full-denominator source status")
    values={n:[metrics(o,r["gold"]) for o,r in zip(order,rows)] for n,order in ranking["orders"].items()}
    baseline=values["FULL_max"]
    independent={n:summarize(rows,v,baseline) for n,v in values.items()}
    for name,expected in independent.items():
        if any(abs(summary[name][k]-v)>1e-12 for k,v in expected.items()):
            raise ValueError("Independent aggregate/transition counts disagree")
    ablation_stats={}
    for a,b in (("combined_head10","prior_head10"),("combined_head10","packet_head10"),
                ("combined_guarded","combined_head10")):
        ablation_stats[a+"_minus_"+b]=interval(rows,[x["MAP@10"]-y["MAP@10"] for x,y in zip(values[a],values[b])])
    previous=0.
    for i,name in enumerate(sorted(ablation_stats,key=lambda n:ablation_stats[n]["paired_randomization_p"])):
        previous=max(previous,min(1.,(len(ablation_stats)-i)*ablation_stats[name]["paired_randomization_p"]))
        ablation_stats[name]["holm_p_exploratory_ablation_contrasts"]=previous
    save(out/"ablation_statistics.json",ablation_stats)
    packet_cohorts=read(PACKETS/"protocol.json")["cohorts"]
    dev45=set(packet_cohorts["development45"])
    used135=dev45|set(packet_cohorts["confirmation90"])
    groups={"all500":list(range(500)),
            "excluding_packet_development45":[i for i,r in enumerate(rows) if r["dataset_row"] not in dev45],
            "excluding_previous_packet135":[i for i,r in enumerate(rows) if r["dataset_row"] not in used135]}
    if [len(groups[k]) for k in groups]!=[500,455,365]:
        raise ValueError("Historical packet cohorts do not align")
    grouped={g:{n:summarize([rows[i] for i in idx],[v[i] for i in idx],[baseline[i] for i in idx])
                for n,v in values.items()} for g,idx in groups.items()}
    projects={}
    for repo in sorted({r["repository"] for r in rows}):
        idx=[i for i,r in enumerate(rows) if r["repository"]==repo]
        projects[repo]={n:summarize([rows[i] for i in idx],[v[i] for i in idx],[baseline[i] for i in idx]) for n,v in values.items()}
    save(out/"group_summary.json",grouped)
    save(out/"project_summary.json",projects)
    changes=[]
    for i,row in enumerate(rows):
        old,new=baseline[i],values["combined_head10"][i]
        changes.append({**{k:row[k] for k in IDENTITY},"gold":row["gold"],"delta_ap":new["MAP@10"]-old["MAP@10"],
            "transition":("kept_correct" if new["Top1"] else "broken") if old["Top1"] else ("corrected" if new["Top1"] else "still_wrong"),
            "original_top10":row["base"][:10],"new_top10":ranking["orders"]["combined_head10"][i][:10],
            "source_failed":not bool(row["base"])})
    save(out/"changes.json",changes)
    save(out/"review_selection.json",{
        "largest_gains":sorted(changes,key=lambda r:(-r["delta_ap"],r["dataset_row"]))[:3],
        "largest_losses":sorted(changes,key=lambda r:(r["delta_ap"],r["dataset_row"]))[:3],
        "all_broken_top1":[r for r in changes if r["transition"]=="broken"],
        "note":"Post-hoc diagnostic cases, not representative mechanism-frequency estimates"})
    with (out/"comparison.csv").open("w") as handle:
        writer=csv.writer(handle);writer.writerow(["cohort","method","n","MAP@10","MRR@10","Top1","Top5","Top10","corrected_top1","broken_top1"])
        for group,methods in grouped.items():
            for name,m in methods.items():writer.writerow([group,name]+[m[k] for k in ("n","MAP@10","MRR@10","Top1","Top5","Top10","corrected_top1","broken_top1")])
    final=summary["combined_head10"];original=summary["FULL_max"];bl=summary["BLAgent"]
    kept=sum(a["Top1"] and b["Top1"] for a,b in zip(baseline,values["combined_head10"]))
    audit_result={"status":"passed","reports":500,"source_status":dict(counts),"scored_file_documents":score_pairs,
        "ranking_checks":500*len(values),"protected_pairs":protected,"separate180_overlap":0,"new_8B_calls":0,
        "original_correct_top1":sum(v["Top1"] for v in baseline),"preserved_correct_top1":kept,
        "BLAgent_successful":sum(bool(r["bl"]) for r in rows),"BLAgent_failed":sum(not r["bl"] for r in rows),
        "guard_changed_rankings":sum(a!=b for a,b in zip(ranking["orders"]["combined_head10"],ranking["orders"]["combined_guarded"])),
        "summary_sha256":sha(out/"summary.json"),"audit_script_sha256":sha(Path(__file__)),
        "checks":["exact historical500 identities, all failures retained","unchanged frozen weights/checkpoint",
                  "fresh inference input/source/result checksums","exact query and selected code fragments",
                  "no evaluation fields in model inputs","independent AP/MRR/Top-k and transition counts",
                  "ranking-formula recomputation","head-set and tail preservation","protected pair ordering"]}
    save(out/"independent_audit.json",audit_result)
    lines=["# 冻结C组合：完整500条结果","","2026-11-23。使用已冻结公式 `0.75×原Max + 0.25×文件联合评分 + 4×初排分数`，仅调整原前十。没有重新调参，没有新增8B重排。", "",
        "相同历史500条、原顺序，499条源码输入可用，1条已有上游失败保留计零；对499条全部重新执行0.6B联合评分，没有拼接此前45/90的分数。", "",table(summary),"",
        f"最终组合相对原C：MAP {original['MAP@10']:.6f} → {final['MAP@10']:.6f}（{(final['MAP@10']-original['MAP@10'])*100:+.2f}个百分点）；"
        f"首位纠正{final['corrected_top1']}条、改坏{final['broken_top1']}条，原本正确首位保留{kept}/{audit_result['original_correct_top1']}。"
        f"前十集合及尾部逐例保持不变。相对本地BLAgent，MAP差为{final['MAP@10']-bl['MAP@10']:+.6f}。","",
        "## 配对统计","","四项预设主比较进行Holm校正；本批包含开发数据，区间用于描述本批差异，不能作为全新独立验证。",""]
    for name,s in stats.items():
        ci=s["ci95_report_bootstrap"];pc=s["ci95_project_cluster_bootstrap"]
        lines.extend([f"- {name}：ΔMAP={s['delta']:+.6f}；逐报告95%区间[{ci[0]:+.6f},{ci[1]:+.6f}]；"
                      f"项目整体重采样区间[{pc[0]:+.6f},{pc[1]:+.6f}]；Holm p={s['holm_p']:.6f}。"])
    lines.extend(["","## 消融诊断","","以下为补充探索性比较，三项另作Holm校正，不替代四项主比较。",""])
    for name,s in ablation_stats.items():
        ci=s["ci95_report_bootstrap"]
        lines.append(f"- {name}：ΔMAP={s['delta']:+.6f}；逐报告95%区间[{ci[0]:+.6f},{ci[1]:+.6f}]；"
                     f"Holm p={s['holm_p_exploratory_ablation_contrasts']:.6f}。")
    for group in ("excluding_packet_development45","excluding_previous_packet135"):
        lines.extend(["","## "+group,"",table(grouped[group])])
    lines.extend(["","这些子集仍参与过初排先验权重选择，不能称为独立盲测。与此前500之外的180条是不同报告群体，不能直接用两批绝对MAP差异判断模型退化。",
        "","## 口径、成本与核验","",
        "原C沿用与180条相同的FULL重跑版本及RRF平局顺序，其500条MAP为0.647302；历史C3的0.646211属于另一份分数/平局记录，不混用。当前初排仅提供分数，候选池仍是旧FULL，未冒充当前新候选池+C。",
        "",f"本地BLAgent复用同一批500条的已保存结果，其中{audit_result['BLAgent_successful']}条成功、{audit_result['BLAgent_failed']}条失败；失败同样计入500分母。",
        "","500条曾参与先验系数选择，45条还用于联合评分权重选择；本轮固定原参数做完整对照，不能称为新测试集上的泛化证明。",
        "",f"新增文件联合评分{score_pairs:,}个，累计评分{verification['sum_scoring_seconds']:.2f}秒；这是两卡任务的评分时间之和，"
        "不含原流程、模型加载和输入准备，不是端到端墙钟。两个工作进程完成后退出。",
        "",f"独立核验通过：{audit_result['ranking_checks']:,}组方法—报告指标、实际输入来源、固定系数、候选集合和尾部，以及{protected:,}条受保护关系。",
        "","逐例变化见 `changes.json`，回退样本见 `review_selection.json`，分项目见 `project_summary.json`，完整分母结果见 `summary.json`。"])
    (out/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"audit":audit_result,"main":final,"BLAgent":bl},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    audit(args.run.resolve())
