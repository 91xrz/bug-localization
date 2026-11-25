"""Acquire original GHRB issue reports and prepare auditable retrieval rows.

The official checkout is read-only. No upstream code, tests or patches run.
This prepares data; it does not implement Java structural AB or run AB/C.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def github_identity(url, kind):
    parsed = urlsplit(url)
    match = re.fullmatch(r"/([\w.-]+)/([\w.-]+)/" + kind + r"/(\d+)/?", parsed.path)
    if parsed.scheme != "https" or parsed.netloc != "github.com" or not match:
        raise ValueError("Unexpected GitHub URL: " + url)
    owner, repo, number = match.groups()
    return owner + "/" + repo, int(number)


def patch_labels(text):
    """Keep pre-image paths; retain additions as explicit unreachable labels."""
    labels, entries = [], []
    # Only parse file headers, never ---/+++ content inside a hunk.
    for block in re.split(r"(?m)^diff --git ", text)[1:]:
        header = block.split("\n@@", 1)[0]
        old = re.findall(r"(?m)^--- (.+)$", header)
        new = re.findall(r"(?m)^\+\+\+ (.+)$", header)
        if len(old) != 1 or len(new) != 1:
            raise ValueError("Unsupported binary/rename-only/malformed patch header")
        before, after = old[0], new[0]
        chosen = after if before == "/dev/null" else before
        if not chosen.startswith(("a/", "b/")):
            raise ValueError("Unsupported quoted or absolute patch path")
        path = chosen[2:]
        if "\\" in path or any(p in ("", ".", "..") for p in path.split("/")) or PurePosixPath(path).is_absolute():
            raise ValueError("Unsafe patch path")
        if path in labels:
            raise ValueError("Duplicate production file in patch")
        labels.append(path)
        entries.append({"file_path": path, "old_path": before, "new_path": after,
                        "added_file": before == "/dev/null", "deleted_file": after == "/dev/null"})
    if not labels:
        raise ValueError("No production patch labels")
    return labels, entries


def extract_issue(html, requested_url, final_url=None):
    repository, number = github_identity(requested_url, "issues")
    canonical_repository, canonical_number = github_identity(final_url or requested_url, "issues")
    if canonical_number != number:
        raise ValueError("Issue redirected to a different number")
    for raw in re.findall(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>', html, re.S):
        value = json.loads(raw)
        try:
            issue = value["payload"]["issueViewerRoute"]["data"]["repository"]["issue"]
        except (KeyError, TypeError):
            continue
        if issue["number"] != number or issue["repository"]["nameWithOwner"].lower() != canonical_repository.lower():
            raise ValueError("Issue page identity mismatch")
        if not isinstance(issue.get("body"), str) or not isinstance(issue.get("title"), str):
            raise ValueError("Raw Markdown issue body/title unavailable")
        return {"issue_url": requested_url, "issue_number": number, "issue_repository": repository,
                "title": issue["title"], "body": issue["body"],
                "created_at": issue.get("createdAt"), "last_edited_at": issue.get("lastEditedAt"),
                "canonical_issue_repository": canonical_repository,
                "extraction": "GitHub issueViewerRoute original issue body; no comments/PR body"}
    raise ValueError("No verified raw issue body in page")


def fetch_issue(url, cache):
    key = digest(url.encode())
    record_path = cache / "issues" / (key + ".json")
    html_path = cache / "pages" / (key + ".html")
    if record_path.exists():
        result = read(record_path)
        raw = html_path.read_bytes()
        if result["issue_url"] != url or digest(raw) != result["page_sha256"]:
            raise ValueError("Changed issue cache")
        parsed = extract_issue(raw.decode("utf-8"), url, result["final_url"])
        if any(result.get(k, result.get("issue_repository") if k == "canonical_issue_repository" else None) != v for k, v in parsed.items()):
            raise ValueError("Cached issue differs from raw page")
        return result
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (bug-localization research dataset preparation)"})
    with urlopen(request, timeout=45) as response:
        raw = response.read(20_000_001)
        if len(raw) > 20_000_000:
            raise ValueError("Issue page exceeds 20 MB")
        final_url = response.url
    result = extract_issue(raw.decode("utf-8"), url, final_url)
    result.update(page_sha256=digest(raw), fetched_at=datetime.now(timezone.utc).isoformat(), final_url=final_url)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_bytes(raw)
    save(record_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "datasets/GHRB")
    parser.add_argument("--cache", type=Path, default=ROOT / "datasets/GHRB_acquisition")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/ghrb_formatted")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    raw_root, cache, output = args.raw_root.resolve(), args.cache.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("Output exists; choose a new version")
    if not 1 <= args.workers <= 4:
        raise ValueError("Use 1 to 4 download workers")
    commit = subprocess.check_output(["git", "-C", str(raw_root), "rev-parse", "HEAD"], text=True).strip()
    rows, input_hashes = {}, {}
    deprecated = read(raw_root / "debug/deprecated_bugs.json")
    for p in sorted((raw_root / "verified_bug").glob("*.json")):
        content = p.read_bytes()
        input_hashes[str(p.relative_to(raw_root))] = digest(content)
        for key, value in json.loads(content).items():
            if key in rows or key in deprecated or value["bug_id"] != key:
                raise ValueError("Duplicate or deprecated active bug")
            rows[key] = value
    urls = sorted({r["issue"]["url"] for r in rows.values()})
    manifest = {"origin": "https://github.com/coinse/GHRB", "revision": commit,
                "raw_root": str(raw_root), "reports_expected": len(rows), "unique_issues": len(urls),
                "preparation_script_sha256": digest(Path(__file__).read_bytes()),
                "source_hashes": input_hashes, "status": "fetching_reports",
                "query_policy": "Original issue title + raw Markdown body only; never patches, PRs or comments",
                "snapshot_policy": "Unmodified upstream buggy_commit; do not apply GHRB's added test patch",
                "label_policy": "All production-diff paths; preserve additions and non-Java files for explicit coverage audit"}
    save(cache / "manifest.json", manifest)
    issues, failures = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch_issue, url, cache): url for url in urls}
        for i, future in enumerate(as_completed(pending), 1):
            url = pending[future]
            try:
                issues[url] = future.result()
            except Exception as error:
                failures[url] = f"{type(error).__name__}: {error}"
            save(cache / "progress.json", {"completed": i, "expected": len(urls), "success": len(issues), "failures": failures})
            if i % 10 == 0 or i == len(urls):
                print(f"Reports {i}/{len(urls)}, success={len(issues)}, failures={len(failures)}", flush=True)
    if failures:
        manifest.update(status="incomplete_reports", failures=failures)
        save(cache / "manifest.json", manifest)
        raise RuntimeError("Some original reports are unavailable; cached successes preserved, no partial evaluation split exported")
    formatted, audits = [], []
    issue_groups = defaultdict(list)
    for bug_id, row in rows.items():
        repo, pr_number = github_identity(row["PR_url"], "pull")
        if not re.fullmatch("[0-9a-f]{40}", row["buggy_commit"]):
            raise ValueError("Invalid pinned commit")
        report = issues[row["issue"]["url"]]
        patch = raw_root / "data/prod_diff" / (bug_id + ".diff")
        patch_raw = patch.read_bytes()
        labels, entries = patch_labels(patch_raw.decode("utf-8"))
        input_hashes[str(patch.relative_to(raw_root))] = digest(patch_raw)
        body = (report["title"] + "\n\n" + report["body"]).strip()
        if not body:
            raise ValueError("Empty report")
        # The current batch selector requires (repo, issue_id) to be unique.
        # Use the explicit GHRB bug key, preserving the real issue number separately.
        formatted.append({"repo_name": repo, "issue_id": bug_id, "ghrb_bug_id": bug_id,
            "original_issue_number": report["issue_number"], "issue_repository": report["issue_repository"],
            "canonical_issue_repository": report.get("canonical_issue_repository", report["issue_repository"]),
            "issue_url": report["issue_url"], "pull_url": row["PR_url"], "pr_number": pr_number,
            "before_fix_sha": row["buggy_commit"], "merge_commit": row["merge_commit"],
            "title": report["title"], "body": body, "original_issue_body": report["body"],
            "updated_files": repr(labels), "source_revision_role": "GHRB buggy_commit without test patch",
            "dataset_revision": commit, "report_fetched_at": report["fetched_at"],
            "report_page_sha256": report["page_sha256"], "production_diff_sha256": digest(patch_raw),
            "label_mapping_json": json.dumps(entries, ensure_ascii=False)})
        issue_groups[report["issue_url"]].append(bug_id)
        audits.append({"bug_id": bug_id, "repository": repo, "gold_count": len(labels),
                       "added_labels": [e["file_path"] for e in entries if e["added_file"]],
                       "non_java_labels": [p for p in labels if not p.endswith(".java")],
                       "cross_repository_issue": repo.lower() != report["issue_repository"].lower(),
                       "snapshot_verified": False})
    from datasets import Dataset, DatasetDict
    DatasetDict({"test": Dataset.from_list(formatted)}).save_to_disk(str(output))
    manifest.update(status="reports_and_formatted_data_ready", projects=dict(Counter(r["repo_name"] for r in formatted)),
        formatted_root=str(output), reports_downloaded=len(issues), split_semantics="test is evaluation corpus, NOT a frozen train/test split",
        duplicate_issue_groups={u: ids for u, ids in issue_groups.items() if len(ids) > 1},
        added_file_labels=sum(len(a["added_labels"]) for a in audits),
        non_java_labels=sum(len(a["non_java_labels"]) for a in audits),
        cross_repository_issue_bugs=[a["bug_id"] for a in audits if a["cross_repository_issue"]],
        issues_created_before_2021_09=sum(r["issue"]["createdAt"] < "2021-09-01" for r in rows.values()),
        all_snapshots_downloaded=False, historical_issue_revisions_recovered=False,
        upstream_tests_executed=False, ab_c_inference_executed=False)
    save(cache / "manifest.json", manifest)
    save(output / "conversion_report.json", manifest | {"cases": audits})
    save(cache / "SHA256SUMS.json", {str(p.relative_to(cache)): digest(p.read_bytes())
                                   for folder in ("issues", "pages") for p in sorted((cache / folder).iterdir())})
    assert all(digest((raw_root / p).read_bytes()) == h for p, h in input_hashes.items())
    print(json.dumps({k: manifest[k] for k in ("status", "reports_expected", "reports_downloaded", "added_file_labels", "non_java_labels")}), flush=True)


if __name__ == "__main__":
    main()
