"""Train report-group LoRA on pre-fix candidate file evidence; never read the test set."""

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from bug_localization.bm25 import ranking_metrics
from bug_localization.reranking import LocalReranker
from bug_localization.snapshot import sha256_file


def dump(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class LowRankLinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                               dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                               dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        update = F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
        return self.base(x) + update * self.scale


def install(model, rank, alpha):
    if rank < 1 or alpha <= 0 or any(isinstance(x, LowRankLinear) for x in model.modules()):
        raise ValueError("Invalid or already installed adapter")
    model.requires_grad_(False)
    replacements = [(name, module) for name, module in model.named_modules()
                    if name.split(".")[-1] in {"q_proj", "v_proj"} and isinstance(module, nn.Linear)]
    if not replacements:
        raise ValueError("Missing Q/V projections")
    for name, module in replacements:
        parent, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent), attr, LowRankLinear(module, rank, alpha))
    return [name for name, _ in replacements]


def adapter_params(model):
    return {k: p for k, p in model.named_parameters() if k.endswith((".lora_a", ".lora_b"))}


def save_adapter(reranker, folder, protocol, targets):
    folder.mkdir()
    dest = folder / "adapter.safetensors"
    save_file({k: p.detach().cpu().contiguous() for k, p in adapter_params(reranker.model).items()}, str(dest))
    dump(folder / "adapter_config.json", {
        "format": "c_lora_qv_v1", "rank": protocol["rank"], "alpha": protocol["alpha"],
        "targets": targets, "base_files_sha256": reranker.identity["files_sha256"],
        "instruction": reranker.config["instruction"], "adapter_sha256": sha256_file(dest),
        "protocol_sha256": sha256_file(folder.parent / "protocol.json")})


def load_adapter(reranker, folder):
    folder = Path(folder)
    config = json.loads((folder / "adapter_config.json").read_text())
    path = folder / "adapter.safetensors"
    if (config["format"] != "c_lora_qv_v1" or config["base_files_sha256"] != reranker.identity["files_sha256"]
            or config["adapter_sha256"] != sha256_file(path)
            or config["instruction"] != reranker.config["instruction"]):
        raise ValueError("Adapter identity mismatch")
    targets = install(reranker.model, config["rank"], config["alpha"])
    params, state = adapter_params(reranker.model), load_file(str(path))
    if targets != config["targets"] or params.keys() != state.keys():
        raise ValueError("Adapter targets mismatch")
    with torch.no_grad():
        for k, p in params.items():
            if p.shape != state[k].shape or not torch.isfinite(state[k]).all():
                raise ValueError("Invalid adapter tensor " + k)
            p.copy_(state[k])
    reranker.model.requires_grad_(False).eval()
    return reranker


def group_loss(scores, positives):
    positives = torch.as_tensor(positives, dtype=torch.bool, device=scores.device)
    if not bool(positives.any()) or not bool((~positives).any()):
        raise ValueError("Report must have both labeled and competing files")
    return F.softplus(scores[~positives][None, :] - scores[positives][:, None]).mean()


def capped(tokenizer, text, limit):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= limit:
        return text, False
    return tokenizer.decode(ids[:limit], skip_special_tokens=False, clean_up_tokenization_spaces=False), True


def encode_group(reranker, row):
    files = row["files"]
    paths = [f["path"] for f in files]
    positives = [p in set(row["gold"]) for p in paths]
    if (not row["query"].strip() or len(paths) != len(set(paths)) or not any(positives)
            or all(positives) or any(not f["chunks"] for f in files)):
        raise ValueError("Invalid sampled report group: " + str(row["id"]))
    tokenizer = reranker.tokenizer
    query, trunc_q = capped(tokenizer, row["query"], 512)
    trunc_docs, chunks, sizes = 0, [], []
    for f in files:
        sizes.append(len(f["chunks"]))
        for text in f["chunks"]:
            doc, truncated = capped(tokenizer, text, 512)
            trunc_docs += truncated
            chunks.append(f"<Instruct>: {reranker.config['instruction']}\n<Query>: {query}\n<Document>: {doc}")
    tokens = tokenizer(chunks, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    ids = [reranker.prefix + t + reranker.suffix for t in tokens]
    if max(map(len, ids)) > reranker.config["max_tokens"]:
        raise ValueError("Combined prompt exceeds C3 context limit")
    return {"id": row["id"], "paths": paths, "gold": row["gold"], "positives": positives,
            "sizes": sizes, "ids": ids, "truncated_query": trunc_q, "truncated_chunks": trunc_docs}


def logits(reranker, ids):
    inputs = reranker.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
    inputs = {k: v.to(reranker.model.device) for k, v in inputs.items()}
    logits_ = reranker.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    return logits_[:, reranker.yes_id[0]] - logits_[:, reranker.no_id[0]]


def score_files(reranker, group):
    # Full file-max gradients without retaining graphs for all the losing chunks.
    # Replay the same file minibatch when taking the winning score's gradient so
    # padding, BF16 rounding, and the first-index tie rule stay identical.
    selected = []
    for ids in split_list(group["ids"], group["sizes"]):
        with torch.no_grad():
            scores = logits(reranker, ids)
            best = int(scores.argmax())
        selected.append((ids, best, scores[best]))
    return selected


def split_list(items, sizes):
    start = 0
    for n in sizes:
        yield items[start:start + n]
        start += n
    if start != len(items):
        raise ValueError("Invalid chunk grouping")


def train_report(reranker, group, grad_divisor):
    # Small groups fit in memory: take all maxima in a single autograd graph.
    batches = list(split_list(group["ids"], group["sizes"]))
    if sum(len(ids) * max(map(len, ids)) for ids in batches) <= 6000:
        scores = torch.stack([logits(reranker, ids).max(dim=0).values for ids in batches])
        loss = group_loss(scores, group["positives"])
        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss")
        (loss / grad_divisor).backward()
        return loss.item()
    selected = score_files(reranker, group)
    leaves = torch.stack([v for _, _, v in selected]).detach().requires_grad_()
    loss = group_loss(leaves, group["positives"])
    if not torch.isfinite(loss):
        raise ValueError("Non-finite loss")
    (grad,) = torch.autograd.grad(loss / grad_divisor, leaves)
    for index, (ids, best, score) in enumerate(selected):
        live = logits(reranker, ids)[best]
        if not torch.isclose(live.detach(), score, rtol=0, atol=1e-5):
            raise ValueError("Recomputed file score differs from selected maximum")
        live.backward(grad[index])
    return loss.item()


@torch.no_grad()
def validate(reranker, groups):
    records = []
    for group in groups:
        scores = [v.item() for _, _, v in score_files(reranker, group)]
        n = len(scores)
        # Lexical path, never input ordering or labels, breaks sampling-pool ties.
        rank = sorted(range(n), key=lambda i: (-scores[i], group["paths"][i]))
        m = ranking_metrics([group["paths"][i] for i in rank], group["gold"], 10)
        p, neg = [scores[i] for i in range(n) if group["positives"][i]], [
            scores[i] for i in range(n) if not group["positives"][i]]
        accuracy = sum((a > b) + .5 * (a == b) for a in p for b in neg) / (len(p) * len(neg))
        records.append({"id": group["id"], "AP@10": m["average_precision_at_k"], "pair_accuracy": accuracy})
    return {"sampled_pool_MAP@10": statistics.mean(r["AP@10"] for r in records),
            "mean_pair_accuracy": statistics.mean(r["pair_accuracy"] for r in records), "reports": records}


def run(args):
    if min(args.epochs, args.accumulation, args.rank) < 1:
        raise ValueError("Invalid training configuration")
    train_path, val_path = (args.data / (s + ".jsonl") for s in ("train", "validation"))
    def records(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    train, val = records(train_path), records(val_path)
    ids = [r["id"] for r in train + val]
    if (not train or not val or len(set(ids)) != len(ids) or
            {r["repo"] for r in train} & {r["repo"] for r in val}):
        raise ValueError("Empty, duplicate, or overlapping training/validation groups")
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((PROJECT / "configs/retrieval/reranker.json").read_text())
    protocol = {"seed": args.seed, "epochs": args.epochs, "rank": args.rank, "alpha": args.alpha,
                "lr": args.lr, "accumulate_reports": args.accumulation, "selection": "fixed final epoch",
                "objective": "per report mean all positive-negative pair softplus(negative - positive)",
                "pool": "maximum chunk score; first chunk on ties", "query_doc_train_token_cap_each": 512,
                "optimizer": "AdamW; betas=(.9,.999), eps=1e-8, no weight decay, clip grad norm 1",
                "train_sha256": sha256_file(train_path), "validation_sha256": sha256_file(val_path),
                "program_sha256": sha256_file(Path(__file__)), "original": original,
                "device": args.device, "train_reports": len(train), "validation_reports": len(val)}
    dump(args.output / "protocol.json", protocol)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rr = LocalReranker(PROJECT, {**original, "device": args.device})
    train = [encode_group(rr, row) for row in train]
    val = [encode_group(rr, row) for row in val]
    dump(args.output / "inputs.json", {split: {"reports": len(groups),
        "files": sum(len(g["paths"]) for g in groups), "chunk_pairs": sum(len(g["ids"]) for g in groups),
        "truncated_queries": sum(g["truncated_query"] for g in groups),
        "truncated_chunks": sum(g["truncated_chunks"] for g in groups)} for split, groups in
        [("train", train), ("validation", val)]})
    start = time.monotonic()
    dump(args.output / "validation_epoch0.json", validate(rr, val))
    targets = install(rr.model, args.rank, args.alpha)
    params = list(adapter_params(rr.model).values())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.)
    steps = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        shuffled = train.copy()
        rng.shuffle(shuffled)
        losses = []
        for start_i in range(0, len(shuffled), args.accumulation):
            batch = shuffled[start_i:start_i + args.accumulation]
            optimizer.zero_grad(set_to_none=True)
            for group in batch:
                loss = train_report(rr, group, len(batch))
                losses.append(loss)
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            if steps % 5 == 0 or start_i + len(batch) == len(shuffled):
                print(f"epoch={epoch} reports={start_i + len(batch)}/{len(shuffled)} steps={steps} "
                      f"mean_loss={statistics.mean(losses):.4f} grad_norm={norm.item():.3f}", flush=True)
        metrics = validate(rr, val)
        dump(args.output / f"validation_epoch{epoch}.json", metrics)
        history.append({"epoch": epoch, "mean_training_loss": statistics.mean(losses),
                        **{k: v for k, v in metrics.items() if k != "reports"}})
    save_adapter(rr, args.output / "adapter", protocol, targets)
    summary = {"epochs": history, "steps": steps, "trainable_parameters": sum(p.numel() for p in params),
               "elapsed_seconds": time.monotonic() - start, "base_identity": rr.identity,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(rr.model.device)
                    if rr.model.device.type == "cuda" else None,
               "adapter_config_sha256": sha256_file(args.output / "adapter/adapter_config.json")}
    dump(args.output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "results/c_lora/round1/data")
    parser.add_argument("--output", type=Path, default=PROJECT / "results/c_lora/round1/training")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261120)
    run(parser.parse_args())
