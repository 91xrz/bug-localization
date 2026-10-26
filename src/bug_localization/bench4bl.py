"""Bench4BL release-based input adapter. Never checks out or executes subject code."""

import ast
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import uuid
import xml.etree.ElementTree as ET

from .snapshot import sha256_file, snapshot_inventory

PROTOCOL = 'bench4bl_author_release_groups_v1'
GROUPS = ('Apache', 'Commons', 'JBoss', 'Spring', 'Wildfly')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def literal(path, subject):
    value = ast.literal_eval(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict) or subject not in value:
        raise ValueError('Invalid author metadata: ' + str(path))
    return value[subject]


def git(repo, *args):
    result = subprocess.run(['git', '--no-replace-objects', '-c', 'core.fsmonitor=false', '-C', str(repo), *args],
                            stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
    if result.returncode:
        raise ValueError('Read-only Git operation failed: ' + result.stderr.decode('utf-8', errors='replace')[:500])
    return result.stdout


def release_commit(repo, tag):
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError('Missing author release tag')
    # Resolve only an exact tag, never HEAD, a branch, or a guessed nearby version.
    commit = git(repo, 'rev-parse', '--verify', '--end-of-options', 'refs/tags/' + tag.strip() + '^{commit}').decode().strip()
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Invalid release commit')
    return commit


def java_tree(repo, commit):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Invalid source revision')
    entries = []
    for record in git(repo, 'ls-tree', '-r', '-z', '--full-tree', commit).split(b'\0'):
        if not record:
            continue
        header, raw_path = record.split(b'\t', 1)
        mode, kind, blob = header.decode('ascii').split()
        path = raw_path.decode('utf-8')
        if not path.lower().endswith('.java') or kind != 'blob' or mode not in ('100644', '100755'):
            continue
        pure = PurePosixPath(path)
        if pure.is_absolute() or '..' in pure.parts or '\\' in path or ':' in path:
            raise ValueError('Unsafe Git tree path')
        entries.append({'path': path, 'blob': blob, 'mode': mode})
    if not entries:
        raise ValueError('No regular Java files in the release tree')
    return sorted(entries, key=lambda e: e['path'])


def map_labels(labels, entries):
    """Exact path or unique qualified-name suffix; ambiguous/missing labels stay gold."""
    canonical = {}
    for label in labels:
        text = label.strip().replace('\\', '/')
        if '/' not in text and text.endswith('.java'):
            text = text[:-5].replace('.', '/') + '.java'
        canonical[label] = text
    matches = defaultdict(list)
    wanted = set(canonical.values())
    paths = {e['path'] for e in entries}
    for path in sorted(paths):
        parts = path.split('/')
        for i in range(len(parts)):
            suffix = '/'.join(parts[i:])
            if suffix in wanted:
                matches[suffix].append(path)
    gold, audit = [], []
    for label in labels:
        text = canonical[label]
        candidates = [text] if text in paths else matches[text]
        if len(candidates) == 1:
            target, status = candidates[0], 'resolved'
        else:
            # Preserve the original denominator without guessing a file or dropping its label.
            target = '__bench4bl_unresolved_gold__/' + fingerprint(label)[:24] + '.java'
            if target in paths:
                raise ValueError('Unresolved-gold marker collides with a real candidate')
            status = 'ambiguous' if candidates else 'not_in_release'
        gold.append(target)
        audit.append({'original': label, 'mapped': target, 'status': status, 'candidates': candidates})
    return list(dict.fromkeys(gold)), audit


def read_reports(path):
    reports = {}
    for node in ET.parse(path).getroot().findall('bug'):
        number = int(node.attrib['id'])
        if number in reports:
            raise ValueError('Duplicate XML report id: ' + str(number))
        info = node.find('buginformation')
        if info is None:
            raise ValueError('Missing buginformation')
        text = lambda key: ''.join(info.find(key).itertext()) if info.find(key) is not None else ''
        labels = list(dict.fromkeys(''.join(f.itertext()).strip() for f in node.findall('fixedFiles/file') if ''.join(f.itertext()).strip()))
        reports[number] = {'id': number, 'title': text('summary'), 'body': text('description'),
                           'reported_version': text('version'), 'reported_fixed_version': text('fixedVersion'),
                           'opened_at': node.get('opendate', ''), 'fixed_at': node.get('fixdate', ''), 'labels': labels}
    return reports


def prepare(project, raw_root, output, subjects=None):
    """Build all curated modern records; no source snapshots or embeddings are generated."""
    from datasets import Dataset, DatasetDict

    raw_root, output = raw_root.resolve(), output.resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a new --output to preserve prior experiments')
    rows, registry, audits = [], {}, []
    selected_subjects = set(subjects) if subjects else None
    seen_subjects = set()
    for group in GROUPS:
        for base in sorted((raw_root / 'data' / group).iterdir()):
            subject = base.name
            label = group + '/' + subject
            if selected_subjects is not None and label not in selected_subjects:
                continue
            seen_subjects.add(label)
            versions = literal(base / 'versions.txt', subject)
            groups = literal(base / 'bugs.txt', subject)
            reports = read_reports(base / 'bugrepo/repository.xml')
            if not isinstance(versions, dict) or not isinstance(groups.get('all'), list):
                raise ValueError('Missing author report/release metadata: ' + label)
            assignments = {}
            for version, tag in versions.items():
                key = subject + '_' + re.sub('[^a-zA-Z0-9]', '_', version)
                for number in groups.get(key, []):
                    if number in assignments:
                        raise ValueError('Multiple author version assignments: ' + label + '#' + str(number))
                    assignments[number] = (version, tag, key)
            if set(assignments) != set(groups['all']):
                raise ValueError('Version groups do not exactly cover author all-list: ' + label)
            repository = 'Bench4BL/' + group + '_' + subject
            tree_cache, label_cache = {}, {}
            counters = {'subject': label, 'reports': 0, 'resolved_labels': 0, 'unresolved_labels': 0,
                        'reports_with_unresolved_labels': 0, 'metadata_sha256': {name: sha256_file(base / name)
                         for name in ('versions.txt', 'bugs.txt', 'answers.txt', 'bugrepo/repository.xml')}}
            for number in groups['all']:
                if number not in reports or not reports[number]['labels']:
                    raise ValueError('Curated report has no XML or labels: ' + label + '#' + str(number))
                report = reports[number]
                version, tag, group_key = assignments[number]
                if tag not in tree_cache:
                    commit = release_commit(base / 'gitrepo', tag)
                    entries = java_tree(base / 'gitrepo', commit)
                    tree_cache[tag] = (commit, entries)
                    needed = list(dict.fromkeys(f for bid in groups['all'] if assignments[bid][1] == tag
                                                for f in reports[bid]['labels']))
                    _, mapped = map_labels(needed, entries)
                    label_cache[tag] = {a['original']: a for a in mapped}
                    source_key = repository + '@' + commit
                    registry.setdefault(source_key, {'repository': repository, 'commit': commit, 'subject': label,
                         'tag': tag, 'release_version': version, 'java_tree_sha256': fingerprint(entries),
                         'java_files': len(entries), 'protocol': PROTOCOL})
                commit, entries = tree_cache[tag]
                mappings = [label_cache[tag][f] for f in report['labels']]
                gold = list(dict.fromkeys(m['mapped'] for m in mappings))
                unresolved = sum(m['status'] != 'resolved' for m in mappings)
                counters['reports'] += 1
                counters['resolved_labels'] += len(mappings) - unresolved
                counters['unresolved_labels'] += unresolved
                counters['reports_with_unresolved_labels'] += bool(unresolved)
                rows.append({'repo_name': repository, 'issue_id': number, 'title': report['title'], 'body': report['body'],
                     'updated_files': repr(gold),
                     # Compatibility alias only. The explicit role below travels with the data/output.
                     'before_fix_sha': commit, 'source_revision': commit, 'snapshot_protocol': PROTOCOL,
                     'source_revision_role': 'author_release_tag_not_per_bug_pre_fix',
                     'bench4bl_subject': label, 'release_version': version, 'release_tag': tag,
                     'author_version_group': group_key, 'reported_version': report['reported_version'],
                     'reported_fixed_version': report['reported_fixed_version'],
                     'opened_at': report['opened_at'], 'fixed_at': report['fixed_at'],
                     'original_updated_files': repr(report['labels']),
                     'label_mapping_json': json.dumps(mappings, ensure_ascii=False), 'unresolved_gold_count': unresolved})
            audits.append(counters)
            print(f"{label}: {counters['reports']} reports; unresolved labels={counters['unresolved_labels']}", flush=True)
    if selected_subjects and selected_subjects != seen_subjects:
        raise ValueError('Unsupported/missing subjects: ' + ', '.join(sorted(selected_subjects - seen_subjects)))
    if not rows:
        raise ValueError('No selected reports')
    try:
        root_record = raw_root.relative_to(project.resolve()).as_posix()
    except ValueError:
        root_record = str(raw_root)
    staging = output.with_name(output.name + '.preparing-' + uuid.uuid4().hex)
    DatasetDict({'test': Dataset.from_list(rows)}).save_to_disk(str(staging))
    registry_data = {'protocol': PROTOCOL, 'raw_root': root_record, 'sources': registry}
    (staging / 'bench4bl_sources.json').write_text(json.dumps(registry_data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = {'protocol': PROTOCOL, 'reports': len(rows), 'subjects': len(audits), 'release_sources': len(registry),
        'query_fields': ['title', 'body'], 'split_semantics': 'evaluation corpus, not a newly invented train/test split',
        'compatibility_before_fix_sha': 'Contains author release commit, NOT per-bug pre-fix commit',
        'label_policy': 'Keep every author-selected report and gold label; unresolved labels stay as explicit missing-gold markers',
        'xml_only_reports_not_in_author_all': 'Not selected; the supplied bugs.txt all-list is authoritative',
        'legacy_previous': 'Five Previous subjects lack bugs.txt and/or have inconsistent release metadata; not silently guessed',
        'implementation_sha256': sha256_file(Path(__file__)), 'projects': audits}
    (staging / 'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output)
    return report


def local_snapshot(project, dataset, repository, commit):
    """Materialize regular Java blobs from the bundled Git objects, lazily and offline."""
    if not re.fullmatch('[A-Za-z0-9_.-]+', dataset) or dataset in ('.', '..'):
        raise ValueError('Invalid source dataset')
    if not re.fullmatch('[0-9a-f]{40}', commit or ''):
        raise ValueError('Bench4BL source revision must be an exact release SHA')
    registry_path = project / 'datasets' / dataset / 'bench4bl_sources.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    if registry['protocol'] != PROTOCOL:
        raise ValueError('Unsupported Bench4BL source protocol')
    entry = registry['sources'].get(repository + '@' + commit)
    if not entry or entry['repository'] != repository or entry['commit'] != commit:
        raise ValueError('Source is not registered by the converted dataset')
    group, subject = entry['subject'].split('/')
    if group not in GROUPS or not re.fullmatch('[A-Za-z0-9_]+', subject):
        raise ValueError('Invalid registered subject')
    raw_root = Path(os.environ.get('BENCH4BL_ROOT', registry['raw_root']))
    raw_root = raw_root if raw_root.is_absolute() else project / raw_root
    repo = raw_root / 'data' / group / subject / 'gitrepo'
    if release_commit(repo, entry['tag']) != commit:
        raise ValueError('Author release tag changed since conversion')
    entries = java_tree(repo, commit)
    if fingerprint(entries) != entry['java_tree_sha256']:
        raise ValueError('Java source tree differs from converted corpus')
    base = project / 'repos/bench4bl' / group / subject / commit
    source, manifest_path = base / 'source', base / 'snapshot.json'
    provenance = {'protocol': PROTOCOL, 'subject': entry['subject'], 'release_tag': entry['tag'],
                  'release_version': entry['release_version'], 'release_commit': commit,
                  'revision_role': 'author_release_tag_not_per_bug_pre_fix',
                  'java_tree_sha256': entry['java_tree_sha256']}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (manifest.get('repository') != repository or manifest.get('commit') != commit
                or manifest.get('bench4bl_source') != provenance or snapshot_inventory(source) != manifest['files']):
            raise ValueError('Cached Bench4BL snapshot identity or contents changed')
        return source, manifest
    if source.exists():
        raise ValueError('Unverified Bench4BL snapshot exists; review it before retrying')
    base.mkdir(parents=True, exist_ok=True)
    staging = base / ('source.preparing-' + uuid.uuid4().hex)
    staging.mkdir()
    paths = [e['path'] for e in entries]
    if os.name == 'nt' and len({p.casefold() for p in paths}) != len(paths):
        raise ValueError('Case-colliding source paths require a case-sensitive filesystem')
    command = ['git', '--no-replace-objects', '-c', 'core.fsmonitor=false', '-C', str(repo), 'cat-file', '--batch']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for entry_file in entries:
            process.stdin.write((entry_file['blob'] + '\n').encode('ascii'))
            process.stdin.flush()
            header = process.stdout.readline().decode('ascii').split()
            if len(header) != 3 or header[:2] != [entry_file['blob'], 'blob']:
                raise ValueError('Unexpected Git object response')
            size = int(header[2])
            raw = process.stdout.read(size)
            if len(raw) != size or process.stdout.read(1) != b'\n':
                raise ValueError('Incomplete source blob')
            if hashlib.sha1(b'blob ' + str(size).encode() + b'\0' + raw).hexdigest() != entry_file['blob']:
                raise ValueError('Git blob checksum mismatch')
            path = staging / entry_file['path']
            if not path.resolve().is_relative_to(staging.resolve()):
                raise ValueError('Source path escapes snapshot')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        process.stdin.close()
        if process.wait(timeout=30):
            raise ValueError('Git object reader failed')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if not process.stdin.closed:
            process.stdin.close()
        process.stdout.close()
        process.stderr.close()
    staging.rename(source)
    manifest = {'repository': repository, 'commit': commit, 'source_url': 'https://github.com/exatoa/Bench4BL',
                'source_provider': 'bench4bl_release', 'bench4bl_source': provenance,
                'archive_sha256': None, 'files': snapshot_inventory(source)}
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(manifest_path)
    return source, manifest
