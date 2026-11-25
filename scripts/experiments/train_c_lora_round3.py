"""Matched real-candidate LoRA experiment: pairwise versus head-sensitive ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from statistics import fmean

import torch
from train_c_lora import (PROJECT, LocalReranker, adapter_params, dump, group_loss, install, logits,
                          ranking_metrics, save_adapter, score_files, sha256_file, split_list)

ROUND = PROJECT / "results/c_lora/round3"
SETTINGS = dict(epochs=2, rank=8, alpha=16., lr=5e-5, seed=20261120, accumulate_reports=4,
                selection="fixed final epoch", temperature=1., head_weight=.8)


def objective(scores, labels, arm):
    pos = torch.as_tensor(labels, dtype=torch.bool, device=scores.device)
    pair = group_loss(scores, pos)
    if arm == "pair":
        return pair
    if arm != "head":
        raise ValueError("Unknown objective: " + arm)
    # Probability that the first file drawn from the score softmax is ANY repair file.
    nll = torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[pos], dim=0)
    return SETTINGS["head_weight"] * nll + (1 - SETTINGS["head_weight"]) * pair


def encode(rr, row):
    paths = [f["path"] for f in row["files"]]
    gold = set(row["gold"])
    flags = [p in gold for p in paths]
    if not any(flags) or all(flags) or len(set(paths)) != len(paths):
        raise ValueError("Invalid training report: " + row["id"])
    texts, sizes = [], []
    for file in row["files"]:
        parts = file["chunks"]
        if not parts or not all(isinstance(c, str) and c for c in parts):
            raise ValueError("Missing evidence")
        sizes.append(len(parts))
        texts.extend(f"<Instruct>: {rr.config['instruction']}\n<Query>: {row['query']}\n<Document>: {c}"
                     for c in parts)
    ids = rr.tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    ids = [rr.prefix + t + rr.suffix for t in ids]
    if max(map(len, ids)) > rr.config["max_tokens"]:
        raise ValueError("Archived prompt too long; refusing to truncate")
    return {"id":row["id"], "ids":ids, "sizes":sizes, "positives":flags}


def update_report(rr, group, arm, divisor):
    # Replay file batches so this is the true Max gradient without retaining every chunk graph.
    selected = score_files(rr, group)
    detached = torch.stack([v for _, _, v in selected]).detach().requires_grad_()
    loss = objective(detached, group["positives"], arm)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite ranking loss")
    (weights,) = torch.autograd.grad(loss / divisor, detached)
    for (batch, best, original), weight in zip(selected, weights):
        live = logits(rr, batch)[best]
        if not torch.isclose(live.detach(), original, atol=1e-5, rtol=0):
            raise ValueError("Gradient replay did not match the selected file score")
        live.backward(weight)
    return loss.item()


@torch.no_grad()
def validate(rr, rows, prior=None):
    records = []
    for row in rows:
        files = row["files"]
        texts = [text for f in files for text in f["chunks"]]
        order = row.get("chunk_order", list(range(len(texts))))
        if sorted(order) != list(range(len(texts))):
            raise ValueError("Invalid archived chunk order")
        scores = [None] * len(texts)
        for idx, item in zip(order, rr.score(row["query"], [texts[i] for i in order]) if texts else [], strict=True):
            scores[idx] = item
        groups = split_list(scores, [len(f["chunks"]) for f in files])
        file_scores = [max(c["score"] for c in group) for group in groups]
        ranked = sorted(zip(files, file_scores), key=lambda pair: (-pair[1], pair[0]["semantic_rank"]))
        m = ranking_metrics([f["path"] for f,s in ranked], row["gold"], 10)
        record = {"id":row["id"], **m, "ranking":[f["path"] for f,s in ranked]}
        if files and all("chunk_scores" in f for f in files):
            archive_scores = [v for f in files for v in f["chunk_scores"]]
            if len(archive_scores) != len(scores):
                raise ValueError("Archived scores do not cover validation")
            record["archive_scores_reproduced"] = all(s["score"] == v for s,v in zip(scores,archive_scores))
            record["archive_rank_reproduced"] = record["ranking"] == [
                f["path"] for f in sorted(files,key=lambda f: f["base_rank"])]
        records.append(record)
    out = {k:fmean(r[key] for r in records) for k,key in [
        ("MAP@10","average_precision_at_k"),("MRR@10","reciprocal_rank_at_k"),
        ("Top-1","top_1"),("Top-5","top_5"),("Top-10","top_10")]}
    if prior is not None:
        previous = {r["id"]:r["top_1"] for r in prior["reports"]}
        out.update(top1_rescued=sum(r["top_1"] > previous[r["id"]] for r in records),
                   top1_lost=sum(r["top_1"] < previous[r["id"]] for r in records))
    return {**out, "reports":records}


def train(arm, data, output, device):
    paths = [data/(s+".jsonl") for s in ("train","validation")]
    train_rows, val = [[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
    all_ids = [r["id"] for r in train_rows + val]
    if (not train_rows or not val or len(set(all_ids)) != len(all_ids)
            or {r["repo"].casefold() for r in train_rows} & {r["repo"].casefold() for r in val}):
        raise ValueError("Empty, repeated or project-overlapping data")
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((PROJECT/"configs/retrieval/reranker.json").read_text())
    p = dict(**SETTINGS, arm=arm, objective="pairwise" if arm=="pair" else "0.8 top1 positive-set NLL + 0.2 pairwise",
             pool="file Max; all archived chunks; no additional truncation", device=device,
             original=config, train_sha256=sha256_file(paths[0]), validation_sha256=sha256_file(paths[1]),
             source_sha256={f.name:sha256_file(f) for f in (Path(__file__), Path(__file__).with_name("train_c_lora.py"))},
             train_reports=len(train_rows),validation_reports=len(val))
    dump(output/"protocol.json", p)
    torch.manual_seed(p["seed"])
    rng = random.Random(p["seed"])
    rr = LocalReranker(PROJECT, {**config, "device":device})
    groups = [encode(rr, row) for row in train_rows]
    start = time.monotonic()
    baseline = validate(rr, val)
    dump(output/"validation_epoch0.json", baseline)
    targets = install(rr.model, p["rank"], p["alpha"])
    params = list(adapter_params(rr.model).values())
    fingerprint = hashlib.sha256()
    for param in params:
        fingerprint.update(param.detach().cpu().numpy().tobytes())
    dump(output/"inputs.json", dict(chunk_pairs=sum(len(g["ids"]) for g in groups),
         max_prompt_tokens=max(len(ids) for g in groups for ids in g["ids"]),
         initial_adapter_sha256=fingerprint.hexdigest()))
    opt = torch.optim.AdamW(params, lr=p["lr"], weight_decay=0.)
    history, steps = [], 0
    for epoch in range(1,p["epochs"]+1):
        order = groups.copy()
        rng.shuffle(order)
        dump(output/f"epoch{epoch}_order.json", [g["id"] for g in order])
        losses = []
        for offset in range(0, len(order), p["accumulate_reports"]):
            batch = order[offset:offset+p["accumulate_reports"]]
            opt.zero_grad(set_to_none=True)
            for group in batch:
                losses.append(update_report(rr, group, arm, len(batch)))
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            opt.step()
            steps += 1
            if steps%5 == 0 or offset+len(batch)==len(order):
                print(f"{arm}: epoch={epoch}, processed={offset+len(batch)}/{len(order)}, "
                      f"steps={steps}, mean_loss={fmean(losses):.4f}",flush=True)
        v = validate(rr, val, baseline)
        dump(output/f"validation_epoch{epoch}.json", v)
        history.append(dict(epoch=epoch, mean_loss=fmean(losses), **{k:v[k] for k in v if k!="reports"}))
    save_adapter(rr, output/"adapter", p, targets)
    summary = dict(epochs=history, steps=steps, trainable_parameters=sum(v.numel() for v in params),
                   elapsed_seconds=time.monotonic()-start, base_identity=rr.identity,
                   adapter_config_sha256=sha256_file(output/"adapter/adapter_config.json"),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(rr.model.device)
                       if rr.model.device.type=="cuda" else None)
    dump(output/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("pair","head"))
    parser.add_argument("--data", type=Path, default=ROUND/"data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args=parser.parse_args()
    train(args.arm, args.data, args.output or (ROUND/args.arm/"training"), args.device)
