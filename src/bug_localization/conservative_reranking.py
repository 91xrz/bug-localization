"""Label-free evidence aggregation and constrained refinement of a file ranking.

The caller supplies a frozen candidate order.  Head-only refinement preserves its
Top-k membership exactly; it cannot guarantee that its original Top-1 was right.
"""

import heapq
import math


def _unique(paths):
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate file in ranking")


def evidence_score(scores, temperature=0.):
    """Max at T=0, otherwise length-normalized log-mean-exp of chunk logits.

    Normalization prevents files with more selected chunks receiving a free
    count bonus. This is score aggregation, not PARADE representation learning.
    """
    if not scores or any(not math.isfinite(s) for s in scores):
        raise ValueError("Require finite, nonempty evidence scores")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Invalid temperature")
    peak = max(scores)
    if temperature == 0:
        return peak
    return peak + temperature * math.log(
        sum(math.exp((s - peak) / temperature) for s in scores) / len(scores))


def residual_rank(base, chunks, prior, *, temperature=0., prior_weight=0., head=10):
    """Combine chunk evidence with a retrieval prior inside a fixed prefix.

    The prior is the existing A/B/structure score, not a label or a predicted
    probability. Logit and prior scales are calibrated on development data only.
    """
    _unique(base)
    if type(head) is not int or head < 1:
        raise ValueError("head must be a positive integer")
    if not math.isfinite(prior_weight) or prior_weight < 0:
        raise ValueError("Invalid prior weight")
    if any(p not in chunks or p not in prior for p in base):
        raise ValueError("Incomplete candidate features")
    if any(not math.isfinite(prior[p]) for p in base):
        raise ValueError("Non-finite prior")
    values = {p: evidence_score(chunks[p], temperature) + prior_weight * prior[p] for p in base}
    index = {p: i for i, p in enumerate(base)}
    return sorted(base[:head], key=lambda p: (-values[p], index[p])) + base[head:]


def constrain_proposal(base, proposal, chunks, prior, *, margin=4., prior_margin=.05,
                       corroboration=False, head=10):
    """Project a proposed permutation onto evidence-supported precedence edges.

    A baseline-ordered pair is protected when its max-logit margin and its
    independent retrieval-prior margin both support the baseline. Optional
    corroboration requires the *second* chunk of the higher file to beat the
    best chunk of the lower file by the same margin. Protected edges always
    point forward in the baseline, so they cannot form a cycle.

    Invalid/failed proposals abstain, explicitly retaining the valid base.
    With margin=None there are no protected edges (fallback-only comparator).
    """
    _unique(base)
    if type(head) is not int or head < 1:
        raise ValueError("head must be a positive integer")
    if margin is not None and (not math.isfinite(margin) or margin < 0):
        raise ValueError("Invalid guard margin")
    if not math.isfinite(prior_margin) or prior_margin < 0:
        raise ValueError("Invalid prior margin")
    prefix = base[:head]
    if proposal is None or len(proposal) != len(prefix) or set(proposal) != set(prefix):
        return list(base), {"abstained": True, "protected_pairs": 0, "blocked_inversions": 0}
    if any(p not in chunks or p not in prior for p in prefix):
        raise ValueError("Incomplete guard features")
    if any(not math.isfinite(prior[p]) for p in prefix):
        raise ValueError("Non-finite prior")
    peaks = {p: evidence_score(chunks[p]) for p in prefix}
    supporting = {p: sorted(chunks[p], reverse=True)[1] if len(chunks[p]) > 1 else None for p in prefix}
    edges = []
    if margin is not None:
        for i, a in enumerate(prefix):
            for b in prefix[i + 1:]:
                strength = supporting[a] if corroboration else peaks[a]
                if (strength is not None and strength - peaks[b] >= margin
                        and prior[a] - prior[b] >= prior_margin):
                    edges.append((a, b))
    proposed_index = {p: i for i, p in enumerate(proposal)}
    outgoing = {p: [] for p in prefix}
    indegree = dict.fromkeys(prefix, 0)
    for a, b in edges:
        outgoing[a].append(b)
        indegree[b] += 1
    ready = [(proposed_index[p], p) for p in prefix if indegree[p] == 0]
    heapq.heapify(ready)
    ranked = []
    while ready:
        _, p = heapq.heappop(ready)
        ranked.append(p)
        for q in outgoing[p]:
            indegree[q] -= 1
            if indegree[q] == 0:
                heapq.heappush(ready, (proposed_index[q], q))
    if len(ranked) != len(prefix):
        raise AssertionError("Baseline-oriented precedence graph must be acyclic")
    return ranked + base[head:], {
        "abstained": False, "protected_pairs": len(edges),
        "blocked_inversions": sum(proposed_index[a] > proposed_index[b] for a, b in edges)}
