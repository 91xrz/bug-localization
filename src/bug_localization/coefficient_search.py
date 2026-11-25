"""Joint AB/C coefficient replay with candidate rebuilding at every setting."""
import itertools
import math
import numpy as np

from .bm25 import ranking_metrics
from .lightweight_hypothesis import normalize

METRICS = ("MAP@10", "MRR@10", "Top1", "Top5", "Top10", "Recall@10", "Top50", "Recall@50")


def grids(protocol):
    ab = list(itertools.product(protocol["ab_alphas"], protocol["anchor_weights"]))
    c = list(itertools.product(protocol["packet_weights"], protocol["prior_weights"]))
    return ab, c, [tuple(x+y) for x in ab for y in c]


def validate_case(case):
    if case["status"] != "ok":
        return
    for rows in (case["a"], case["b"]):
        if not rows or [r["rank"] for r in rows] != list(range(1,len(rows)+1)):
            raise ValueError("Incomplete retrieval ranking")
        if len({r["file_path"] for r in rows}) != len(rows) or not all(math.isfinite(r["score"]) for r in rows):
            raise ValueError("Duplicate paths or nonfinite retrieval")
        if any(x["score"] < y["score"] for x,y in zip(rows,rows[1:])):
            raise ValueError("Unsorted retrieval")
    if {r["file_path"] for r in case["a"]} != {r["file_path"] for r in case["b"]}:
        raise ValueError("A/B inventories differ")
    pool = {r["file_path"] for ranks in (case["a"][:50],case["b"][:50]) for r in ranks}
    for key in ("anchor", "max", "joint"):
        if set(case[key]) != pool or not all(math.isfinite(v) for v in case[key].values()):
            raise ValueError("Incomplete union scores: " + key)
    if not all(0 <= v <= 1 for v in case["anchor"].values()):
        raise ValueError("Structural score outside [0,1]")


def metric(order, gold):
    gold = set(gold)
    m = ranking_metrics(order,gold,10)
    return np.array([m[k] for k in ("average_precision_at_k","reciprocal_rank_at_k","top_1","top_5","top_10","recall_at_k")]
                    + [float(bool(set(order[:50]) & gold)), len(set(order[:50]) & gold)/len(gold)])


def rank(case, coefficients):
    if case["status"] != "ok":
        return [], []
    validate_case(case)
    alpha,gamma,beta,lam = coefficients
    a,b = normalize(case["a"]),normalize(case["b"])
    ar={r["file_path"]:r["rank"] for r in case["a"]}
    br={r["file_path"]:r["rank"] for r in case["b"]}
    prior={p:alpha*a[p]+(1-alpha)*b[p]+gamma*e for p,e in case["anchor"].items()}
    ab=sorted(prior,key=lambda p:(-prior[p],br[p],ar[p],p))[:50]
    base=sorted(ab,key=lambda p:-case["max"][p])  # stable ties preserve AB order
    head=sorted(base[:10],key=lambda p:-((1-beta)*case["max"][p]+beta*case["joint"][p]+lam*prior[p]))
    return head+base[10:],ab


def _metrics(head, tail_hits, gold_count):
    positions=np.arange(1,head.shape[-1]+1)
    ap=(head.cumsum(axis=-1)*head/positions).sum(axis=-1)/gold_count
    rr=(head/positions).max(axis=-1)
    h50=head.sum(axis=-1)+tail_hits
    return np.stack((ap,rr,head[...,0],head[...,:5].any(axis=-1),head.any(axis=-1),
                     head.sum(axis=-1)/gold_count,h50>0,h50/gold_count),axis=-1).astype(float)


def evaluate_grid(case, gold, ab_grid, c_grid):
    """Return [AB*C, metric] and [AB, metric] in itertools.product order."""
    if not gold:
        raise ValueError("No gold files")
    if case["status"] != "ok":
        return np.zeros((len(ab_grid)*len(c_grid),len(METRICS))),np.zeros((len(ab_grid),len(METRICS)))
    validate_case(case)
    ar={r["file_path"]:r["rank"] for r in case["a"]}
    br={r["file_path"]:r["rank"] for r in case["b"]}
    paths=sorted(case["anchor"],key=lambda p:(br[p],ar[p],p))
    a,b=normalize(case["a"]),normalize(case["b"])
    aa=np.array([a[p] for p in paths]);bb=np.array([b[p] for p in paths]);ee=np.array([case["anchor"][p] for p in paths])
    z=np.array([case["max"][p] for p in paths]);u=np.array([case["joint"][p] for p in paths])
    relevant=np.array([p in set(gold) for p in paths])
    alpha,gamma=np.asarray(ab_grid).T
    prior=alpha[:,None]*aa+(1-alpha[:,None])*bb+gamma[:,None]*ee
    ab_order=np.argsort(-prior,axis=1,kind='stable')[:,:50]
    ab_hits=relevant[ab_order]
    ab_metrics=_metrics(ab_hits[:,:10],ab_hits[:,10:].sum(axis=-1),len(set(gold)))
    max_order=np.take_along_axis(ab_order,np.argsort(-z[ab_order],axis=1,kind='stable'),axis=1)
    head=max_order[:,:10]
    phead=np.take_along_axis(prior,head,axis=1)
    beta,lam=np.asarray(c_grid).T
    scores=(1-beta[None,:,None])*z[head][:,None,:]+beta[None,:,None]*u[head][:,None,:]+lam[None,:,None]*phead[:,None,:]
    permutation=np.argsort(-scores,axis=2,kind='stable')
    heads=np.take_along_axis(np.broadcast_to(head[:,None,:],permutation.shape),permutation,axis=2)
    tail_hits=relevant[max_order[:,10:]].sum(axis=1)[:,None]
    result=_metrics(relevant[heads],tail_hits,len(set(gold))).reshape(-1,len(METRICS))
    return result,ab_metrics
