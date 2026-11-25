"""Bounded static source lookups and guarded, pair-only ranking updates."""
import ast
import hashlib
import re
from pathlib import Path


def terms(text):
    return {s.lower() for s in re.findall(r'[A-Za-z_][A-Za-z_0-9]{2,}', text)}


def checked_lines(root, path, checksum):
    file = (root / path).resolve()
    if Path(path).is_absolute() or not file.is_relative_to(root.resolve()):
        raise ValueError("Source path outside snapshot")
    raw = file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("Source no longer matches snapshot")
    return raw.decode("utf-8", errors="replace").splitlines()


def menu(lines, query):
    """Return addressable definitions and reference sites, not their contents."""
    words = terms(query)
    pool = []
    try:
        tree = ast.parse("\n".join(lines))
    except (SyntaxError, ValueError):
        tree = None
    if tree:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                lo, hi = node.lineno, node.end_lineno
                name = node.name
                local = set.union(set(), *(terms(s) for s in lines[lo-1:min(hi,lo+65)]))
                score = 15*(name.lower() in words) + len(words & local)
                pool.append((score, {"action":"definition", "symbol":name,"lo":lo,"hi":hi,"anchor":lo}))
    matches = []
    for i, line in enumerate(lines, 1):
        overlap = terms(line) & words
        if overlap:
            score = sum(min(12,len(w)) for w in overlap)
            matches.append((score,i,sorted(overlap, key=lambda w:(-len(w),w))[0]))
    selected = [v for _,v in sorted(pool,key=lambda a:(-a[0],a[1]["lo"]))[:8]]
    sites = []
    for _,i,word in sorted(matches,key=lambda v:(-v[0],v[1])):
        if all(abs(i-j)>6 for j in sites):
            sites.append(i)
            selected.append({"action":"reference_window","symbol":word,"lo":max(1,i-12),
                             "hi":min(len(lines),i+12),"anchor":i})
            if len(sites)==5:
                break
    if not selected:
        selected.append({"action":"source_start","symbol":"module","lo":1,"hi":max(1,len(lines)),"anchor":1})
    fixed = next((i for i,v in enumerate(selected) if v["action"]=="reference_window"),0)
    return selected,fixed


def excerpt(lines, action, tokenizer, original, cap=384):
    lo, hi, center = action["lo"], min(action["hi"],len(lines)), action["anchor"]
    # Contiguous, complete source lines; never fabricate or truncate a statement mid-line.
    chosen = {}
    for n in sorted(range(lo,hi+1), key=lambda n:(abs(n-center), n)):
        trial = {**chosen,n:lines[n-1]}
        text = "\n".join(f"{i}: {s}" for i,s in sorted(trial.items()))
        if len(tokenizer.encode(text,add_special_tokens=False)) > cap:
            break
        chosen = trial
    old = {s.strip() for s in original.splitlines()}
    return {"text":"\n".join(f"{i}: {s}" for i,s in sorted(chosen.items())),
            "lines":[i for i in sorted(chosen) if chosen[i].strip()],
            "novel":[i for i in sorted(chosen) if chosen[i].strip() and chosen[i].strip() not in old]}


def apply_verdict(ranking, paths, evidence, verdict):
    pref = verdict["prefer"]
    if pref == "abstain":
        return ranking, "abstained"
    refs = [verdict["line_A"],verdict["line_B"]]
    if pref not in ("A","B") or any(n not in e["lines"] for n,e in zip(refs,evidence)):
        return ranking, "invalid_citation"
    if not any(n in e["novel"] for n,e in zip(refs,evidence)):
        return ranking, "no_new_cited_line"
    best = paths["AB".index(pref)]
    if best == ranking[0]:
        return ranking, "confirmed"
    return [best]+[f for f in ranking if f!=best], "promoted"
