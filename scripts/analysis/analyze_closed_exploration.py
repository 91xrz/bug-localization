"""Replay bounded evidence and compute whole-cohort paired ranking effects."""
import json
import sys
from pathlib import Path
from collections import Counter
from statistics import fmean

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from pilot_closed_exploration import ROOT, COHORTS, evidence, update, checked_lines
from pilot_c3_joint import PROJECT, BL, C3, read, write_json, sha256_file, ranking_metrics
from analyze_c_agent90 import cost, paired


def archived(index, folder, key):
    r=index[key]
    if r["status"]!="ok": return None
    path=folder/r["result_file"]
    if sha256_file(path)!=r["result_sha256"]:
        raise ValueError("Historical record changed")
    return read(path)


def verify(sample,d,obj,old,order,tok):
    if not obj["triggered"]:
        return
    pre=PROJECT/"repos"/sample["repository"]/sample["before_fix_sha"]
    manifest=read(pre/"snapshot.json")
    paths=obj["paths"]
    if not set(paths)<=set(order[:10]) or paths[0]!=order[0] or len(set(paths))!=3:
        raise ValueError("Invalid inspection set")
    def check(id,expected):
        letter=id[0]; path=paths["ABC".index(letter)]
        if manifest["files"][path]!=obj["sources"][path]:
            raise ValueError("Source did not match snapshot")
        lines=checked_lines(pre/"source",path,obj["sources"][path])
        if evidence(lines,obj["symbols"][letter][id],id,tok)!=expected:
            raise ValueError("Unreproducible source evidence")
    if "first_evidence" not in obj:
        if any(obj["arms"][m]["ranking"]!=order for m in obj["arms"]):raise ValueError("Invalid first lookup")
        return
    check(obj["first"]["id"],obj["first_evidence"])
    for mode,a in obj["arms"].items():
        prompt=read(d/mode/"second/call_001.json")["request"]["messages"][-1]["content"]
        suffix="\nFirst inspection result: "+json.dumps(obj["first_evidence"])
        if prompt.endswith(suffix)!=(mode=="feedback"):
            raise ValueError("Incorrect observation exposed to selector")
        if "second_evidence" in a:
            id=a["second"]["id"]
            check(id,a["second_evidence"])
            if len({id[0],obj["first"]["id"][0]})!=2: raise ValueError("Repeated file")
            target=paths["ABC".index(id[0])]
            judge=read(d/mode/"judge/call_001.json")
            payload=json.loads(judge["request"]["messages"][-1]["content"])
            if (payload["leader"]["evidence"],payload["rival"]["evidence"])!=(obj["first_evidence"],a["second_evidence"]):
                raise ValueError("Different evidence sent to judge")
            if a["verdict"] and json.loads(judge["response"])!=a["verdict"]:
                raise ValueError("Recorded verdict differs")
            if update(order,target,a["action"])!=a["ranking"]:
                raise ValueError("Unauthorized move")
        elif a["ranking"]!=order:
            raise ValueError("Invalid second lookup moved a file")


def analyze():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(PROJECT/"models/Qwen3-8B",local_files_only=True)
    bl={r["dataset_row"]:r for r in read(BL/"samples.json")}; c3={r["position"]:r for r in read(C3/"samples.json")}
    all_summaries={}
    for mode,(sel,joint) in COHORTS.items():
        rows=[]; calls=Counter(); token_in=Counter(); token_out=Counter(); acts={m:Counter() for m in ("feedback","static")}
        for s in read(sel)["samples"]:
            key=f"{s['position']:04d}"
            d=ROOT/mode/key
            obj=read(d/"result.json")
            j=joint/"cases"/key
            rec=read(j/"record.json")
            if (obj["dataset_row"]!=s["dataset_row"] or obj["stamp"]["record"]!=sha256_file(j/"record.json") or
                    obj["stamp"]["protocol"]!=sha256_file(ROOT/"protocol.json")):raise ValueError("Identity mismatch")
            order=[];old=None
            if rec["status"]=="ok":
                if obj["stamp"]["input"]!=sha256_file(j/"inference_input.json") or rec["result_sha256"]!=sha256_file(j/"result.json"):
                    raise ValueError("Input changed")
                order=read(j/"result.json")["ranking"];old=read(j/"inference_input.json")
            cr=archived(c3,C3,s["position"]); br=archived(bl,BL,s["dataset_row"])
            for x in (cr,br):
                if x and any(x[k]!=s[k] for k in ("repository","issue_id","before_fix_sha")): raise ValueError("Bad pairing")
            gold=(cr["ground_truth"] if cr else br["gold_files"])
            if cr and br and set(gold)!=set(br["gold_files"]): raise ValueError("Label mismatch")
            fired=bool(order and order[0]!=old["candidates"][0] and len(old["candidates"])>=3)
            if fired!=obj["triggered"]: raise ValueError("Gate altered")
            verify(s,d,obj,old,order,tok)
            ranks={"joint":order,"blagent":br["ranking"] if br else [],"c3":[r["file_path"] for r in cr["ranking"]] if cr else []}
            if mode=="primary90":
                ranks["old_agent"]=read(joint.parent/"review/cases"/key/"result.json")["arms"]["active"]["ranking"]
            for m,arm in obj["arms"].items():
                r=arm["ranking"]
                if len(r)!=len(order) or set(r[:10])!=set(order[:10]) or r[10:]!=order[10:] or (not fired and r!=order):
                    raise ValueError("Candidate inventory changed")
                acts[m].update([arm["action"]])
                ranks[m]=r
                for place in [d/"first",d/m/"second",d/m/"judge"]:
                    c=cost(place);calls[m]+=c["calls"];token_in[m]+=c["input_tokens"];token_out[m]+=c["output_tokens"]
            def selected(m): return (obj["arms"][m].get("second") or {}).get("id")
            rows.append(dict(repository=s["repository"],issue_id=s["issue_id"],position=s["position"],triggered=fired,
                             changed_lookup=fired and selected("feedback")!=selected("static"),
                             tops={k:(r[0] if r else None) for k,r in ranks.items()},gold=gold,
                             metrics={k:ranking_metrics(r,gold,10) for k,r in ranks.items()}))
        names=rows[0]["metrics"].keys()
        contrasts=[("feedback","joint"),("feedback","static"),("feedback","blagent"),("static","joint")]
        if mode=="primary90": contrasts.append(("feedback","old_agent"))
        summary=dict(n=len(rows),fired=sum(r["triggered"] for r in rows),changed_second=sum(r["changed_lookup"] for r in rows),
                     metrics={m:{k:fmean(r["metrics"][m][k] for r in rows) for k in ("average_precision_at_k","reciprocal_rank_at_k","top_1","top_10")} for m in names},
                     comparisons={f"{a}_minus_{b}":paired(rows,a,b) for a,b in contrasts},
                     extra_logical_calls=calls,extra_input_tokens=token_in,extra_output_tokens=token_out,actions=acts)
        write_json(ROOT/mode/"paired_cases.json",rows)
        all_summaries[mode]=summary
    write_json(ROOT/"summary.json",all_summaries)
    print(json.dumps(all_summaries,indent=2))


if __name__=="__main__":
    analyze()
