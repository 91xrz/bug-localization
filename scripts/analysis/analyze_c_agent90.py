"""Audit source spans, constrained moves, costs and paired outcomes of the agent experiment."""
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from pilot_c3_joint import PROJECT, BL, C3, read, write_json, sha256_file, ranking_metrics, json_object
from prepare_c_agent90 import ROOT
from bug_localization import agent_review as tools
from bug_localization.embeddings import digest


def cost(folder):
    total=Counter()
    for f in folder.glob("call_*.json"):
        c=read(f)
        if digest(c["request"])!=c["request_sha256"] or c["status"] not in ("ok","failed"):
            raise ValueError(f"Invalid model record {f}")
        total.update(calls=1,input_tokens=c.get("input_tokens",0),output_tokens=c.get("output_tokens",0),
                     failed_calls=int(c["status"]=="failed"))
    return total


def paired(rows,left,right):
    groups=defaultdict(list)
    rescued=lost=0
    for r in rows:
        a,b=r["metrics"][left],r["metrics"][right]
        groups[r["repository"]].append(a["average_precision_at_k"]-b["average_precision_at_k"])
        rescued+=a["top_1"]>b["top_1"]
        lost+=a["top_1"]<b["top_1"]
    ds=[v for g in groups.values() for v in g]
    rng=random.Random(20261120)
    boots=sorted(sum(sum(rng.choices(g,k=len(g))) for g in groups.values())/len(ds) for _ in range(10000))
    return dict(delta=fmean(ds),ci95=[boots[249],boots[9749]],ap_wins=sum(x>0 for x in ds),
                ap_ties=ds.count(0),ap_losses=sum(x<0 for x in ds),top1_rescued=rescued,top1_lost=lost)


def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(PROJECT/"models/Qwen3-8B",local_files_only=True)
    samples=read(ROOT/"selection.json")["samples"]
    complete=read(ROOT/"review/complete.json")
    c3={r["position"]:r for r in read(C3/"samples.json")}
    bl={r["dataset_row"]:r for r in read(BL/"samples.json")}
    labels={r["id"]:r for r in read(PROJECT/"results/c_lora/round2/evaluation/labels.json")}
    totals=defaultdict(Counter)
    rows=[]
    prot=sha256_file(ROOT/"review/protocol.json")
    for s in samples:
        key=f"{s['position']:04d}"
        j=ROOT/"joint/cases"/key
        folder=ROOT/"review/cases"/key
        v=read(folder/"result.json")
        rec=read(j/"record.json")
        uid=f"{s['repository']}#{s['issue_id']}@{s['before_fix_sha']}"
        label=labels[uid]
        gold=label["gold"]
        cr=c3[s["position"]]
        br=bl[s["dataset_row"]]
        for rec0 in (cr,br,rec):
            if any(rec0[k]!=s[k] for k in ("repository","before_fix_sha","dataset_row","issue_id")):
                raise ValueError("Report pairing mismatch")
        def hist(directory,record,kind):
            if record["status"]!="ok":
                return []
            f=directory/record["result_file"]
            if sha256_file(f)!=record["result_sha256"]:
                raise ValueError("Archive checksum mismatch")
            data=read(f)
            if set(data["ground_truth" if kind=="c3" else "gold_files"])!=set(gold):
                raise ValueError("Historical labels differ")
            return [r["file_path"] for r in data["ranking"]] if kind=="c3" else data["ranking"]
        ranks={"c3":hist(C3,cr,"c3"),"blagent":hist(BL,br,"blagent")}
        if v["stamp"]["protocol_sha256"]!=prot or v["stamp"]["record_sha256"]!=sha256_file(j/"record.json"):
            raise ValueError("Changed joint/review provenance")
        ranks["joint"]=[]
        if rec["status"]=="ok":
            if sha256_file(j/"result.json")!=rec["result_sha256"] or sha256_file(j/"inference_input.json")!=v["stamp"]["joint_input_sha256"]:
                raise ValueError("Joint case checksum mismatch")
            ranks["joint"]=read(j/"result.json")["ranking"]
            if set(ranks["joint"][:10])!=set(ranks["c3"][:10]) or ranks["joint"][10:]!=ranks["c3"][10:]:
                raise ValueError("Ordinary reranking changed candidates or tail")
        gate=bool(ranks["joint"] and ranks["joint"][0]!=ranks["c3"][0])
        if v["triggered"]!=gate:
            raise ValueError("Gate mismatch")
        costs={"joint":cost(j),"planner":cost(folder/"planner")}
        if gate and v.get("plan") is not None:
            if v["plan"]!=json_object(read(folder/"planner/call_001.json")["response"]):
                raise ValueError("Planner record differs from actual response")
        actions={}
        for m in ("active","fixed"):
            costs[m]=cost(folder/m)
            arm=v["arms"][m]
            actions[m]=arm["action"]
            ranks[m]=arm["ranking"]
            expected=ranks["joint"]
            if gate and "evidence" in arm:
                old=read(j/"inference_input.json")
                paths=sorted([ranks["joint"][0],ranks["c3"][0]])
                if v["paths"]!=paths:
                    raise ValueError("Changed dispute")
                base=PROJECT/"repos"/s["repository"]/s["before_fix_sha"]
                manifest=read(base/"snapshot.json")
                for i,p in enumerate(paths):
                    lines=tools.checked_lines(base/"source",p,manifest["files"][p])
                    choices,idx=tools.menu(lines,old["query"])
                    expect_id=idx if m=="fixed" else int(v["plan"]["AB"[i]])
                    orig="\n\n".join(c["text"] for c in old["evidence"][p])
                    check=tools.excerpt(lines,choices[expect_id],tok,orig)
                    if arm["lookup_ids"][i]!=expect_id or arm["evidence"][i]!=check:
                        raise ValueError("Could not reproduce actual source lookup")
                if arm["verdict"]:
                    if arm["verdict"]!=json_object(read(folder/m/"call_001.json")["response"]):
                        raise ValueError("Verdict differs from actual response")
                    expected,decision=tools.apply_verdict(expected,paths,arm["evidence"],arm["verdict"])
                    if decision!=arm["action"]:
                        raise ValueError("Decision differs from evidence")
            if expected!=ranks[m]:
                raise ValueError("Unexpected ranking change")
        for k,c in costs.items():
            totals[k].update(c)
        met={k:ranking_metrics(order,gold,10) for k,order in ranks.items()}
        if met["c3"]!=label["history_metrics"] or met["blagent"]!=label["blagent_metrics"]:
            raise ValueError("Failed historical metric reproduction")
        challenger=[ranks["c3"][0]]+[p for p in ranks["joint"] if p!=ranks["c3"][0]] if gate else ranks["joint"]
        best_possible=max(met["joint"]["average_precision_at_k"], ranking_metrics(challenger,gold,10)["average_precision_at_k"])
        same=gate and all("evidence" in v["arms"][m] for m in ("active","fixed")) and all(
            a["text"]==b["text"] for a,b in zip(v["arms"]["active"]["evidence"],v["arms"]["fixed"]["evidence"]))
        rows.append(dict(id=uid,repository=s["repository"],position=s["position"],triggered=gate,
                         oracle_allowed_ap=best_possible,identical_pair_evidence=same,
                         actions=actions,top_files={k:v[0] if v else None for k,v in ranks.items()},
                         gold=gold,metrics=met,cost=costs,joint_status=rec["status"]))
    if len(rows)!=90 or len(rows)!=complete["n"] or sum(r["triggered"] for r in rows)!=complete["triggered"]:
        raise ValueError("Incomplete evaluation")
    summary={"n":len(rows),"triggered":complete["triggered"],"joint_status":Counter(r["joint_status"] for r in rows),
             "metrics":{m:{k:fmean(r["metrics"][m][k] for r in rows) for k in
                 ("average_precision_at_k","reciprocal_rank_at_k","top_1","top_5","top_10","recall_at_k")}
                 for m in ("c3","joint","fixed","active","blagent")},
             "comparisons":{a+"_minus_"+b:paired(rows,a,b) for a,b in
                   (("active","fixed"),("active","joint"),("active","blagent"),("fixed","joint"),("joint","blagent"))},
             "oracle_allowed_map10":fmean(r["oracle_allowed_ap"] for r in rows),
             "same_evidence_on_trigger":sum(r["identical_pair_evidence"] for r in rows),
             "actions":{m:Counter(r["actions"][m] for r in rows) for m in ("fixed","active")},
             "physical_cost":dict(totals),"logical_joint_plus_review":{m:dict(totals["joint"]+totals["planner"]+totals[m])
                  for m in ("fixed","active")}}
    out=ROOT/"analysis"
    out.mkdir(exist_ok=True)
    write_json(out/"paired_cases.json",rows)
    write_json(out/"summary.json",summary)
    print(summary)


if __name__=="__main__":
    main()
