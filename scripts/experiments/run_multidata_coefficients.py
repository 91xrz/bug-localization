"""Frozen SWE500/GHRB134 score collection for multilingual joint coefficient tests.

Each stage is resumable. No parameter choice is made while collecting scores.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/experiments")]
from bug_localization.structural_matching import extract_symbols, anchors, feature_identity
from bug_localization.snapshot import sha256_file, snapshot_inventory
from bug_localization.evidence_packets import file_packets
from bug_localization.lightweight_hypothesis import normalize


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare(out):
    from datasets import load_from_disk
    out.mkdir(parents=True, exist_ok=False)
    datasets = {"GHRB": list(load_from_disk(str(ROOT / "datasets/ghrb_formatted"))["test"]),
                "SWE": list(load_from_disk(str(ROOT / "datasets/swe_bench_formatted"))["test"])}
    selected = read(ROOT / "configs/experiments/selections/crossproject_500_seed20261111.json")["samples"]
    ghrb = [{"position": i+1, "dataset_row": i, "repository": r["repo_name"], "issue_id": r["issue_id"],
             "before_fix_sha": r["before_fix_sha"], "body": r["body"], "gold": __import__("ast").literal_eval(r["updated_files"]),
             "issue_url": r["issue_url"]} for i, r in enumerate(datasets["GHRB"])]
    swe = [{**m, "body": datasets["SWE"][m["dataset_row"]]["body"],
            "gold": __import__("ast").literal_eval(datasets["SWE"][m["dataset_row"]]["updated_files"])} for m in selected]
    plan = {"GHRB": ghrb, "SWE": swe}
    for dataset, rows in plan.items():
        projects = sorted({r["repository"] for r in rows}, key=lambda p: fingerprint([20261124, dataset, p]))
        dev = set(projects[:len(projects)//2])
        for r in rows:
            r["partition"] = "development" if r["repository"] in dev else "validation"
        # Repeated reports cannot cross project partitions, even with cross-repo issues.
        groups = defaultdict(set)
        for r in rows:
            groups[fingerprint(" ".join(r["body"].split()))].add(r["partition"])
            if r.get("issue_url"):
                groups[r["issue_url"]].add(r["partition"])
        if any(len(v) > 1 for v in groups.values()):
            raise ValueError("Duplicate report crosses frozen project partitions")
    # Cross-dataset text/identity leakage check (not a claim about model pretraining).
    if {fingerprint(" ".join(r["body"].split())) for r in swe} & {fingerprint(" ".join(r["body"].split())) for r in ghrb}:
        raise ValueError("Cross-dataset duplicate report")
    semantic = read(ROOT / "configs/retrieval/ghrb/vector.json") | {"gpu_count": 1, "embedding_batch_size": 64}
    reranker = read(ROOT / "configs/retrieval/reranker.json")
    protocol = {"datasets": {k: len(v) for k, v in plan.items()}, "structure": feature_identity(),
        "seed": 20261124, "ab_alphas": [i/20 for i in range(21)],
        "anchor_weights": [0,.1,.2,.25,.3,.4,.5,.6,.75,1,1.5,2],
        "packet_weights": [i/20 for i in range(21)],
        "prior_weights": [0,.25,.5,.75,1,1.5,2,3,4,5,6,8,10,12,16,24,32,64],
        "default_coefficients": [.5,.5,.25,4], "head": 10, "evaluation_k": 10,
        "candidate_policy": "For every coefficient rebuild AB Top50 from A50 union B50, Max C order then refine original C top10",
        "query_policy": "A/B generation uses original full report only; C uses A encoder-visible report without instruction",
        "retrieval_policy": "GHRB exact NumPy cosine, same Max aggregation and +1 offset; SWE audited historical ES scores reused",
        "semantic": semantic, "reranker": reranker, "joint_reranker": reranker | {"max_tokens":4096,"batch_size":8},
        "generation": read(ROOT / "configs/experiments/report_hypothesis_crossproject180.json"),
        "selection_rule": "All GHRB134; existing historical SWE500. Deterministic disjoint project halves within dataset, never outcome-selected",
        "selection_sha256": fingerprint(plan), "historical_limits": "SWE500 previously observed and tuned. GHRB reports are currently downloaded bodies, not historical edit versions",
        "label_policy": "All original labels retained including absent/new/non-Java files; no outcome-based eligibility filtering",
        "failure_policy": "Generation overflow/invalid output retained as zero. Infrastructure, source and integrity errors remain pending and stop; never drop reports",
        "selection_objective": "maximum development MAP@10; exact ties prefer current coefficients then lower sum",
        "validation_policy": "Freeze per-dataset and shared equal-dataset-weight choice before validation; shared choice never weights larger dataset by sample count",
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in [Path(__file__), ROOT/'src/bug_localization/structural_matching.py']}}
    save(out / "protocol.json", protocol)
    save(out / "selection.json", plan)
    print({k: {s: sum(r["partition"] == s for r in v) for s in ("development", "validation")} for k,v in plan.items()}, flush=True)


def load(out):
    protocol, plan = read(out / "protocol.json"), read(out / "selection.json")
    if fingerprint(plan) != protocol["selection_sha256"] or feature_identity() != protocol["structure"]:
        raise ValueError("Frozen plan or structural implementation changed")
    for p, checksum in protocol["code_sha256"].items():
        if sha256_file(ROOT / p) != checksum:
            raise ValueError("Frozen implementation changed")
    return protocol, plan


def checked_resume(path, input_sha):
    if not path.exists():
        return None
    result = read(path)
    if result["input_sha256"] != input_sha:
        raise ValueError("Checkpoint input changed")
    checkpoint = path.with_suffix(".sha256")
    if checkpoint.read_text().strip() != sha256_file(path):
        raise ValueError("Checkpoint checksum changed")
    return result


def checkpoint(path, result, input_sha):
    save(path, result | {"input_sha256": input_sha})
    path.with_suffix(".sha256").write_text(sha256_file(path) + "\n")


def generate(out):
    from run_direct_hypothesis import Generator
    protocol, plan = load(out)
    model = Generator(protocol["generation"])
    save(out / "generation_identity.json", model.identity)
    for i, row in enumerate(plan["GHRB"], 1):
        path = out / "GHRB/generation" / f"{row['position']:04d}.json"
        sig = fingerprint([row["body"], model.identity])
        if checked_resume(path, sig) is None:
            try:
                result = model.generate(row["body"])
                result = {"status": "failed" if result["generation_error"] else "ok", **result}
            except ValueError as error:
                if str(error) != "Input exceeds limit":
                    raise
                result = {"status": "failed", "error": str(error)}
            checkpoint(path, result, sig)
        save(out / "generation_progress.json", {"completed": i, "expected": len(plan["GHRB"])})
        print("generation", i, len(plan["GHRB"]), flush=True)


def structure(report, paths, source, manifest, cache):
    symbols = {}
    for path in sorted(paths):
        raw = (source / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["files"][path]:
            raise ValueError("Source hash mismatch")
        key = (path, manifest["files"][path])
        if key not in cache:
            cache[key] = extract_symbols(path, raw.decode("utf-8", errors="replace"))
        symbols[path] = cache[key]
    return anchors(report, symbols)


def prepare_swe(out):
    from run_b_lightweight import field
    from run_ab_structure_c500 import mixed_ids, extract_chunks
    protocol, plan = load(out)
    recall = ROOT / "results/recall_full/20261119_swe2294"
    # Resolve authoritative source from the AB study rather than guessing a run.
    recall = ROOT / read(ROOT / "configs/experiments/b_lightweight_20261122.json")["source_run"]
    aa, bb = [{r["dataset_row"]:r for r in read(recall / f"{method}/samples.json")} for method in ("S0", "H0")]
    ab_root = ROOT / "results/b_lightweight/20261122_v1"
    refs = read(ROOT / "results/baselines/20261122_ab_structure_v1/source_checkpoints.json")
    cache = {}
    for row in plan["SWE"]:
        pos = row["position"]
        path = out / "SWE/inputs" / f"{pos:04d}.json"
        sig = fingerprint([row, protocol["structure"]])
        if checked_resume(path, sig) is not None:
            continue
        ra, rb = aa[row["dataset_row"]], bb[row["dataset_row"]]
        if any(r["repository"] != row["repository"] or r["before_fix_sha"] != row["before_fix_sha"] for r in (ra,rb)):
            raise ValueError("SWE identity mismatch")
        if rb["status"] != "ok":
            checkpoint(path, {"status":"failed", "error": rb.get("error", "Upstream B failure")}, sig)
            continue
        input_path = ab_root / "inputs" / f"{ra['position']:04d}.json"
        if sha256_file(input_path) != refs[str(ra["position"])]["prepared_inputs"]:
            raise ValueError("Historical AB source changed")
        inputs = read(input_path)
        a, b = field(recall/ra["result_file"], "ranking"), field(recall/rb["result_file"], "ranking")
        for ranks, saved in ((a, inputs["a"]), (b, inputs["b"])):
            if [{k:r[k] for k in ("rank","file_path","score")} for r in ranks] != saved:
                raise ValueError("Historical retrieval does not reproduce audited AB inputs")
        pool = {r["file_path"] for ranks in (a[:50],b[:50]) for r in ranks}
        ids = mixed_ids(sorted(pool), a, b)
        chunks = extract_chunks(recall/ra["assets"] / "chunks.json", {c for v in ids.values() for c in v})
        source = ROOT / "repos" / row["repository"] / row["before_fix_sha"] / "source"
        features, matches = structure(row["body"], pool, source, {"files": inputs["source_content_sha256"]}, cache)
        old = read(ROOT / "results/ab_structure_c/20261123_fixed500/prepared" / f"{pos:03d}.json")
        docs = [chunks[c] for p in sorted(ids) for c in ids[p]]
        result = {"status":"ok", "a": inputs["a"], "b": inputs["b"], "anchor": features,
            "matches": matches, "query": old["query"], "documents": docs, "ids": ids,
            "retrieval_source_sha256": {"a":ra["result_sha256"],"b":rb["result_sha256"]},
            "source_features_sha256": sha256_file(input_path)}
        checkpoint(path,result,sig)
        save(out / "swe_preparation_progress.json", {"completed":pos,"expected":len(plan["SWE"])})
        if pos % 10 == 0:
            print("SWE inputs",pos,len(plan["SWE"]),flush=True)


def retrieve_ghrb(out):
    import numpy as np
    from bug_localization.embeddings import LocalEmbedder
    from bug_localization.ast_chunking import RoutingChunkCache
    from bug_localization.aggregation import aggregate_chunk_scores
    from run_ab_structure_c500 import mixed_ids
    protocol, plan = load(out)
    encoder = LocalEmbedder(ROOT, protocol["semantic"])
    chunk_cache = RoutingChunkCache(ROOT, encoder.tokenizer, protocol["semantic"], encoder.fingerprint)
    save(out / "embedding_identity.json", encoder.identity)
    cache = {}
    for row in plan["GHRB"]:
        pos = row["position"]
        target = out / "GHRB/inputs" / f"{pos:04d}.json"
        gen_path = out / "GHRB/generation" / f"{pos:04d}.json"
        gen = read(gen_path)
        if sha256_file(gen_path) != gen_path.with_suffix(".sha256").read_text().strip():
            raise ValueError("Generation checkpoint changed")
        sig = fingerprint([row, protocol["structure"], encoder.identity, sha256_file(gen_path)])
        if checked_resume(target, sig) is not None:
            continue
        if gen["status"] != "ok":
            checkpoint(target, {"status":"failed", "error":gen.get("generation_error", gen.get("error"))}, sig)
            continue
        folder = ROOT / "datasets/GHRB_sources" / row["repository"] / row["before_fix_sha"]
        manifest = read(folder / "snapshot.json")
        if snapshot_inventory(folder / "source") != manifest["files"]:
            raise ValueError("GHRB source content changed")
        chunks = []
        for path, sha in sorted(manifest["files"].items()):
            raw = (folder/"source"/path).read_bytes()
            doc = {"file_path":path,"sha256":sha,"content":raw.decode("utf-8",errors="replace")}
            cc, _ = chunk_cache.get(doc,"blagent_ast")
            chunks.extend(cc)
        vectors = encoder.encode([c["embedding_text"] for c in chunks])
        rankings, query_info = [], None
        for text in (row["body"], gen["query"]):
            vector, info = encoder.query(text)
            cosine = (vectors @ vector)/(np.linalg.norm(vectors,axis=1)*np.linalg.norm(vector))
            scores = [{"chunk_id":c["chunk_id"],"file_path":c["file_path"],"cosine_similarity":float(s)} for c,s in zip(chunks,cosine)]
            ranking = aggregate_chunk_scores(scores,.1)["max"]
            rankings.append([{**r,"score":r["score"]+1} for r in ranking])
            if query_info is None:
                query_info = info
        a,b=rankings
        pool={r["file_path"] for ranks in (a[:50],b[:50]) for r in ranks}
        ids=mixed_ids(sorted(pool),a,b)
        lookup={c["chunk_id"]:c for c in chunks}
        features,matches=structure(row["body"],pool,folder/"source",manifest,cache)
        prefix='Instruct: '+protocol["semantic"]["query_instruction"]+'\nQuery: '
        assert query_info["embedding_text"].startswith(prefix)
        checkpoint(target,{"status":"ok","a":a,"b":b,"anchor":features,"matches":matches,
            "query":query_info["embedding_text"][len(prefix):],"query_info":query_info,
            "ids":ids,"documents":[lookup[c] for p in sorted(ids) for c in ids[p]],
            "snapshot_sha256":sha256_file(folder/"snapshot.json"),"chunk_count":len(chunks)},sig)
        save(out/"ghrb_retrieval_progress.json",{"completed":pos,"expected":len(plan["GHRB"])})
        print("GHRB retrieval",pos,len(plan["GHRB"]),"chunks",len(chunks),flush=True)
    encoder.close()


def rerank(out,dataset):
    from bug_localization.reranking import LocalReranker
    protocol,plan=load(out)
    model=None
    for row in plan[dataset]:
        pos=row["position"]
        ip=out/dataset/"inputs"/f"{pos:04d}.json"
        inp=read(ip)
        if sha256_file(ip)!=ip.with_suffix(".sha256").read_text().strip():
            raise ValueError("Prepared scoring input changed")
        sig=fingerprint([sha256_file(ip),protocol["reranker"],protocol["joint_reranker"]])
        target=out/dataset/"scores"/f"{pos:04d}.json"
        if checked_resume(target,sig) is not None:
            continue
        if inp["status"]!='ok':
            checkpoint(target,{"status":"failed","error":inp.get("error")},sig)
            continue
        if model is None:
            model=LocalReranker(ROOT,protocol["reranker"])
            save(out/f"reranker_identity_{dataset}.json",model.identity)
        start=time.monotonic()
        model.config=protocol["reranker"]
        scored=model.score(inp["query"],[d["embedding_text"] for d in inp["documents"]])
        chunk_scores={d["chunk_id"]:s["score"] for d,s in zip(inp["documents"],scored)}
        packets=file_packets(inp["documents"])
        model.config=protocol["joint_reranker"]
        joint=model.score(inp["query"],[p["document"] for p in packets])
        checkpoint(target,{"status":"ok","max":{p:max(chunk_scores[c] for c in ids) for p,ids in inp["ids"].items()},
            "joint":{p["file_path"]:s["score"] for p,s in zip(packets,joint)},
            "chunk_scores":chunk_scores,"seconds":time.monotonic()-start},sig)
        save(out/f"rerank_{dataset}_progress.json",{"completed":pos,"expected":len(plan[dataset])})
        print(dataset,"C scoring",pos,len(plan[dataset]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','generate','prepare-swe','retrieve-ghrb','rerank'],required=True)
    parser.add_argument('--dataset',choices=['SWE','GHRB'],default='GHRB')
    args=parser.parse_args()
    out=args.output.resolve()
    if args.stage=='prepare':prepare(out)
    elif args.stage=='generate':generate(out)
    elif args.stage=='prepare-swe':prepare_swe(out)
    elif args.stage=='retrieve-ghrb':retrieve_ghrb(out)
    elif args.stage=='rerank':rerank(out,args.dataset)
