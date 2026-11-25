"""Acquire pinned GHRB archives and materialize Java candidates for IR experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bug_localization.snapshot import sha256_file, snapshot_inventory


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def fetch(repository, commit, canonical, output):
    folder = output / repository / commit
    source, manifest_path = folder / "source", folder / "snapshot.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_bytes())
        if (m["repository"], m["commit"], m["canonical_repository"]) != (repository, commit, canonical):
            raise ValueError("Snapshot identity changed")
        if snapshot_inventory(source) != m["files"] or sha256_file(folder / "snapshot.zip") != m["archive_sha256"]:
            raise ValueError("Snapshot content changed")
        return {k: m[k] for k in ("repository", "commit", "canonical_repository", "archive_bytes", "java_files")}
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "snapshot.zip"
    reused = ROOT / "repos" / repository / commit / "snapshot.zip"
    url = f"https://codeload.github.com/{canonical}/zip/{commit}"
    if not archive.exists():
        temporary = folder / "snapshot.zip.part"
        if reused.exists():
            shutil.copyfile(reused, temporary)
        else:
            with urlopen(Request(url, headers={"User-Agent": "bug-localization-GHRB-preparation"}), timeout=60) as response, temporary.open("wb") as handle:
                while data := response.read(1024*1024):
                    handle.write(data)
                    if handle.tell() > 1_000_000_000:
                        raise ValueError("Archive exceeds the declared 1 GB limit")
        temporary.replace(archive)
    if source.exists():
        raise ValueError("Incomplete extracted snapshot; inspect before retry")
    files, skipped = {}, []
    with zipfile.ZipFile(archive) as z:
        roots = {PurePosixPath(info.filename).parts[0] for info in z.infolist()}
        allowed = {repository.split("/")[1] + "-" + commit, canonical.split("/")[1] + "-" + commit}
        if len(roots) != 1 or not roots <= allowed:
            raise ValueError("Archive root differs from verified repository/commit")
        selected = []
        for info in z.infolist():
            parts = PurePosixPath(info.filename).parts[1:]
            if "\\" in info.filename or ".." in parts or PurePosixPath(info.filename).is_absolute():
                raise ValueError("Unsafe archive path")
            if info.is_dir() or not parts:
                continue
            path = "/".join(parts)
            reason = None
            if not path.endswith(".java"):
                continue  # Entire original archive is retained separately.
            if info.file_size > 2097152:
                reason = "max_file_bytes"
            elif stat.S_ISLNK(info.external_attr >> 16):
                reason = "symlink"
            elif any(p in {".git", ".venv", "__pycache__", "node_modules"} for p in parts):
                reason = "excluded_directory"
            if reason:
                skipped.append({"path": path, "reason": reason})
            else:
                selected.append((path, info))
        if sum(info.file_size for _, info in selected) > 2_000_000_000:
            raise ValueError("Java source exceeds declared 2 GB limit")
        if len({p for p, _ in selected}) != len(selected) or not selected:
            raise ValueError("Duplicate or empty Java inventory")
        source.mkdir()
        for path, info in selected:
            raw = z.read(info)  # zipfile verifies each selected member CRC.
            dest = source / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            files[path] = hashlib.sha256(raw).hexdigest()
    m = {"repository": repository, "canonical_repository": canonical, "commit": commit,
         "source_url": url, "source": str(source), "archive_bytes": archive.stat().st_size,
         "archive_sha256": sha256_file(archive), "files": files, "java_files": len(files), "skipped_java": skipped,
         "materialization": "Only Java candidates extracted; full upstream ZIP retained; no patches applied"}
    save(manifest_path, m)
    return {k: m[k] for k in ("repository", "commit", "canonical_repository", "archive_bytes", "java_files")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/GHRB_sources")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use 1-4 workers")
    from datasets import load_from_disk
    rows = list(load_from_disk(str(ROOT / "datasets/ghrb_formatted"))["test"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    aliases_path = output / "repository_aliases.json"
    if aliases_path.exists():
        aliases = json.loads(aliases_path.read_bytes())
    else:
        aliases = {}
        for repo in sorted({r["repo_name"] for r in rows}):
            with urlopen(Request("https://github.com/"+repo, headers={"User-Agent": "Mozilla/5.0"}), timeout=45) as response:
                parsed = urlsplit(response.url)
            if parsed.netloc != "github.com" or not re.fullmatch(r"/[\w.-]+/[\w.-]+/?", parsed.path):
                raise ValueError("Unexpected repository redirect")
            aliases[repo] = parsed.path.strip("/")
        save(aliases_path, aliases)
    pairs = sorted({(r["repo_name"], r["before_fix_sha"]) for r in rows})
    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch, repo, sha, aliases[repo], output): (repo, sha) for repo, sha in pairs}
        for i, future in enumerate(as_completed(pending), 1):
            key = "/".join(pending[future])
            try:
                results[key] = future.result()
            except Exception as error:
                errors[key] = f"{type(error).__name__}: {error}"
            save(output / "status.json", {"completed": i, "expected": len(pairs), "success": len(results), "errors": errors})
            print(f"Snapshots {i}/{len(pairs)} success={len(results)} errors={len(errors)} {key}", flush=True)
    save(output / "acquisition.json", {"snapshots": results, "errors": errors, "reports": len(rows),
                                      "status": "passed" if not errors else "incomplete"})
    if errors:
        raise RuntimeError("Snapshot acquisition incomplete; no reports removed")


if __name__ == "__main__":
    main()
