"""Project-held-out matched evaluation on frozen FULL2294 retrieval inputs."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import mean
import sys

from evaluate_c_lora import (BL, C3, KEY, METRICS, PROJECT, checked_result, digest,
                             frozen_input, hard_pairs, identity, prepare_report, ranking_metrics,
                             read, read_checked, save, score, sha256_file)
from evaluate_c_lora_round2 import contrast

sys.path.insert(0, str(PROJECT / "scripts/analysis"))
from audit_c_lora import score_map

FULL = PROJECT / "results/reranking_full/20261119_swe2294_gpu0"
ROUND = PROJECT / "results/c_lora/round3"
REPOS = ("mwaskom/seaborn", "pallets/flask", "psf/requests", "pytest-dev/pytest", "scikit-learn/scikit-learn")
SEED = "round3-objective-project-heldout-v1"
MODES = ("base", "pair", "head")
PRIMARY = "head minus pair (objective) and head minus FULL base (replacement), report MAP@10 and Top-1"


def normalized(body):
    return sha256(" ".join(body.casefold().split()).encode()).hexdigest()


def select(rows, old, limit=20, bodykey=None):
    exclude = {(r["repository"].lower(), r["issue_id"]) for r in old}
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        repo = row["repository"]
        if repo not in REPOS:
            continue
        issue = (repo.lower(), row["issue_id"])
        if issue in seen:
            raise ValueError("Duplicate issue identity in FULL")
        seen.add(issue)
        if issue not in exclude:
            groups[repo].append(row)
    if not any(groups.values()):
        raise ValueError("Empty eligible cohort")
    previous = {bodykey(r) for r in old} if bodykey else set()
    selected = []
    for repo in REPOS:
        count = 0
        for row in sorted(groups[repo], key=lambda x:digest([SEED, identity(x)])):
            if bodykey:
                key = bodykey(row)
                if key in previous:
                    continue
                previous.add(key)
            selected.append(row)
            count += 1
            if count == limit:
                break
    return selected


def freeze(root):
    from datasets import load_from_disk
    if (root / "manifest.json").exists():
        raise FileExistsError(root)
    old = read(C3 / "samples.json")
    full = read(FULL / "FULL/samples.json")
    ds = load_from_disk(str(PROJECT/"datasets/swe_bench_formatted/test"))
    reserved_file = ROUND/"reserved_lite_ids.json"
    reserved = set(read(reserved_file)["dataset_rows"])
    official = [r for r in full if r["dataset_row"] in reserved]
    excluded = old + official
    rows = select(full, excluded, bodykey=lambda r:normalized(ds[r["dataset_row"]]["body"]))
    selection = [dict(position=i, full_position=r["position"], **{k:r[k] for k in KEY}) for i,r in enumerate(rows,1)]
    save(root / "selection.json", selection)
    bs = {identity(r): r for r in read(BL/"samples.json")}
    old_texts = {normalized(ds[r["dataset_row"]]["body"]) for r in excluded}
    hashes, labels, prior = [], [], set()
    for i,r in enumerate(rows,1):
        b = bs[identity(r)]
        raw = ds[r["dataset_row"]]
        sr, gold, body = prepare_report(raw, {"repository":r["repository"],"issue_id":r["issue_id"],"query_fields":["body"]})
        if sr["before_fix_sha"] != r["before_fix_sha"]:
            raise ValueError("Snapshot changed")
        nh = normalized(body)
        if nh in old_texts or nh in prior:
            raise ValueError("Unexpected duplicate report body: preserve selection and audit before running")
        prior.add(nh)
        uid = f"{r['repository']}#{r['issue_id']}@{r['before_fix_sha']}"
        case = checked_result(FULL,r)
        bcase = checked_result(BL,b) if b["status"] == "ok" else None
        item = dict(position=i,id=uid,repository=r["repository"],source_status=r["status"])
        hist = ranking_metrics([],gold,10)
        if r["status"] == "ok":
            item.update(frozen_input(case))
            hist = ranking_metrics([x["file_path"] for x in case["ranking"]],gold,10)
            if (not body.startswith(item["query"]) or hist != r["metrics"] or
                set(case["ground_truth"]) != set(gold) or
                case["ab_ranking"] != [p["file_path"] for p in item["original_order"]]):
                raise ValueError("Original FULL query, label, ranking, or tie-break not reproduced")
        bm = ranking_metrics(bcase["ranking"] if bcase else [],gold,10)
        if bcase and (set(bcase["gold_files"]) != set(gold) or bm != b["metrics"]):
            raise ValueError("BLAgent evaluation does not reproduce")
        oldinput = BL/f"cases/{b['position']:04d}/inputs.json"
        if read(oldinput)["query"].strip() != body:
            raise ValueError("BLAgent report differs")
        path = root/f"inputs/{i:03d}.json"
        save(path,item)
        hashes.append(sha256_file(path))
        labels.append(dict(position=i,id=uid,repository=r["repository"],gold=gold,
                           normalized_sha256=nh,body_sha256=sha256(raw["body"].encode()).hexdigest(),
                           history_metrics=hist,blagent_metrics=bm,blagent_status=b["status"],
                           archive_sha256=r["result_sha256"],archive_file=r["result_file"],
                           blagent_result_sha256=b.get("result_sha256"),blagent_inputs_sha256=sha256_file(oldinput)))
    save(root/"labels.json",labels)
    protocol = read(FULL/"protocol.json")
    sources = (FULL/"FULL/samples.json", C3/"samples.json", BL/"samples.json", FULL/"protocol.json",
               FULL/"dual_resume/verification.json", FULL/"model.json", reserved_file)
    save(root/"manifest.json",dict(n=len(rows),seed=SEED,repositories=REPOS,limit=20,primary=PRIMARY,
         method="Historical FULL pipeline, true H0:40+S0-to-50 pool, fused RRF k=60 tie-break; not the old S0 tie-break",
         comparator="local BLAgent (8B), all gold files @10; not the paper's configuration",
         selection_sha256=sha256_file(root/"selection.json"),input_sha256=hashes,labels_sha256=sha256_file(root/"labels.json"),
         source_sha256={str(p):sha256_file(p) for p in sources},base_model_files=read(FULL/"model.json")["identity"]["files_sha256"],
         config=protocol["reranker"],history500_or_official_normalized_body_overlap=0,
         official_reserved_row_count=len(reserved),per_project=Counter(r["repository"] for r in rows)))
    print(f"Frozen {len(rows)} held-out cases, {sum(len(read(root/f'inputs/{i:03d}.json').get('chunks',[])) for i in range(1,len(rows)+1))} chunks")


def joint_disjoint(data, labels):
    from json import loads
    repos = {r["repository"].casefold() for r in labels}
    hashes = {r["normalized_sha256"] for r in labels}
    for part in ("train","validation"):
        for line in (data/(part+".jsonl")).read_text().splitlines():
            r = loads(line)
            if r["repo"].casefold() in repos or normalized(r["query"]) in hashes or r.get("normalized_body_sha256") in hashes:
                raise ValueError("Data not held out from training/validation")


def run(root, mode, device):
    if mode == "base":
        score(root, mode, device, None, None)
        return
    data = ROUND/"data"
    trainroot = ROUND/mode/"training"
    proto = read(trainroot/"protocol.json")
    adapter = trainroot/"adapter"
    cfg = read(adapter/"adapter_config.json")
    if (cfg["protocol_sha256"] != sha256_file(trainroot/"protocol.json") or proto["epochs"] != 2 or
        proto["selection"] != "fixed final epoch" or any(proto[k+"_sha256"] != sha256_file(data/(k+".jsonl")) for k in ("train","validation"))):
        raise ValueError("Not the intended fixed two-epoch adapter")
    man = read(root/"manifest.json")
    joint_disjoint(data, read_checked(root/"labels.json",man["labels_sha256"]))
    score(root,"lora",device,adapter,data,output_name=mode)


def summarize(root):
    man = read(root/"manifest.json")
    labs = read_checked(root/"labels.json",man["labels_sha256"])
    runs = {k:read(root/k/"run.json") for k in MODES}
    for k,run_ in runs.items():
        conf = read_checked(root/k/"protocol.json",run_["protocol_sha256"])
        if conf["manifest_sha256"] != sha256_file(root/"manifest.json"):
            raise ValueError("Mismatched manifest")
    report, same_metrics, same_blocks, blocks = [], 0, 0, 0
    for i, lab in enumerate(labs):
        fn = f"{i+1:03d}.json"
        item = read_checked(root/"inputs"/fn,man["input_sha256"][i])
        case = read_checked(FULL/lab["archive_file"],lab["archive_sha256"])
        row = {k:lab[k] for k in ("position","id","repository","blagent_metrics","history_metrics")}
        for k, run_ in runs.items():
            result = read_checked(root/k/"cases"/fn,run_["case_sha256"][i])
            if result["id"] != lab["id"] or result["input_sha256"] != man["input_sha256"][i]:
                raise ValueError("Case identity mismatch")
            if result["status"] == "ok":
                _,order = score_map(item,result)
            elif item["source_status"] != "ok" and not result["ranking"] and not result["scored_chunks"]:
                order = []
            else:
                raise ValueError("Unexpected model failure or ranking")
            row[k] = ranking_metrics(order,lab["gold"],10)
            row[k+"_hard_pair_accuracy"] = hard_pairs(item,result["ranking"],lab["gold"])
            row[k+"_status"] = result["status"]
            if k == "base":
                same_metrics += row[k] == row["history_metrics"]
                for new, archived in zip(result["scored_chunks"],case.get("scored_chunks",[]),strict=True):
                    same_blocks += all(new[f]==archived[f] for f in ("chunk_id","file_path","score","probability","input_tokens"))
                    blocks += 1
        report.append(row)
    save(root/"paired_cases.json",report)
    methods = (*MODES,"blagent_metrics")
    stats = lambda rows,key:{display:mean(r[key][v] for r in rows) for display,v in METRICS.items()}
    paired = [r for r in report if all(r[k+"_hard_pair_accuracy"] is not None for k in MODES)]
    comparisons = {}
    for a,b in (("head","pair"),("head","base"),("pair","base"),("head","blagent_metrics")):
        value = contrast(report,a,b)
        binary = [{"repository":r["repository"],
                   **{k:{**r[k],"average_precision_at_k":r[k]["top_1"]} for k in (a,b)}} for r in report]
        top1 = contrast(binary,a,b)
        comparisons[a+"_minus_"+b] = dict(**value,delta_top1=top1["delta_MAP"],
                                          stratified_top1_CI95=top1["project_stratified_bootstrap_CI95"])
    out = dict(n=len(report),recomputed_base_matching_report_metrics=same_metrics,recomputed_matching_base_chunks=same_blocks,
               chunks_per_arm=blocks,per_project_counts=man["per_project"],hard_pair_eligible=len(paired),
               hard_pair_accuracy={k:mean(r[k+"_hard_pair_accuracy"] for r in paired) if paired else None for k in MODES},
               status={k:Counter(r[k+"_status"] for r in report) for k in MODES},
               metrics={k:stats(report,k) for k in methods},
               projects={repo:{k:stats([r for r in report if r["repository"]==repo],k) for k in methods}
                         for repo in man["per_project"]},
               contrasts=comparisons,
               paired_cases_sha256=sha256_file(root/"paired_cases.json"))
    save(root/"paired_summary.json",out)
    print({k:out[k] for k in ("n","metrics","hard_pair_accuracy","recomputed_matching_base_chunks")})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("prepare","summarize",*MODES))
    p.add_argument("--output",type=Path,default=ROUND/"evaluation")
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    if args.mode=="prepare":
        freeze(args.output)
    elif args.mode=="summarize":
        summarize(args.output)
    else:
        run(args.output,args.mode,args.device)
