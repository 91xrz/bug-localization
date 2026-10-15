"""Verify and unpack a manually downloaded Windows distribution into .runtime."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile


PROJECT = Path(__file__).resolve().parents[2]


def main():
    if sys.platform != 'win32':
        raise SystemExit('This installer is for Windows x86_64.')
    config = json.loads((PROJECT / 'configs/elasticsearch/distribution.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Path to the ZIP downloaded by the user')
    args = parser.parse_args()
    runtime = PROJECT / '.runtime/elasticsearch'
    archive = runtime / config['url'].rsplit('/', 1)[-1]
    es_home = runtime / ('elasticsearch-' + config['version'])
    marker = es_home / '.installation-verified.json'
    supplied = args.archive.resolve() if args.archive else archive
    installed = False
    if marker.is_file():
        previous = json.loads(marker.read_text(encoding='utf-8'))
        installed = all(previous.get(key) == config[key] for key in ('version', 'sha512'))
        installed = installed and all((es_home / name).is_file() for name in
                                      ('bin/elasticsearch.bat', 'jdk/bin/java.exe', 'config/jvm.options', 'config/log4j2.properties'))
    if not supplied.is_file() and (args.archive is not None or not installed):
        raise SystemExit('Please download the Windows ZIP yourself first:\n' + config['url']
                         + '\nThen rerun with --archive <local ZIP path>. No software is downloaded by this script.')
    if supplied.is_file():
        with supplied.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha512').hexdigest()
        if actual != config['sha512']:
            raise ValueError('The downloaded archive does not match the official SHA512')
    runtime.mkdir(parents=True, exist_ok=True)
    if supplied.is_file() and supplied.resolve() != archive.resolve():
        if archive.exists():
            with archive.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha512').hexdigest() != actual:
                    raise ValueError('A different archive already exists in .runtime; refusing to overwrite it')
        else:
            shutil.copy2(supplied, archive)
    if not installed:
        if es_home.exists():
            raise ValueError('Unverified installation directory already exists: ' + str(es_home))
        with zipfile.ZipFile(archive) as handle:
            if sum(info.file_size for info in handle.infolist()) > 6_000_000_000:
                raise ValueError('Uncompressed archive is too large')
            for info in handle.infolist():
                parts = PurePosixPath(info.filename).parts
                if not parts or parts[0] != es_home.name or '\\' in info.filename:
                    raise ValueError('Unexpected archive path: ' + info.filename)
                if not (runtime / info.filename).resolve().is_relative_to(es_home.resolve()):
                    raise ValueError('Archive path escapes the installation directory')
            handle.extractall(runtime)
        marker.write_text(json.dumps({'version': config['version'], 'source': config['url'], 'sha512': actual}, indent=2), encoding='utf-8')
    generated = runtime / 'config'
    generated.mkdir(exist_ok=True)
    for name in ('jvm.options', 'log4j2.properties'):
        shutil.copy2(es_home / 'config' / name, generated / name)
    settings = (PROJECT / 'configs/elasticsearch/elasticsearch.yml').read_text(encoding='utf-8')
    settings += '\npath.data: ' + json.dumps(str(runtime / 'data').replace('\\', '/'))
    settings += '\npath.logs: ' + json.dumps(str(PROJECT / 'results/elasticsearch/server').replace('\\', '/')) + '\n'
    (generated / 'elasticsearch.yml').write_text(settings, encoding='utf-8')
    (generated / 'jvm.options.d').mkdir(exist_ok=True)
    (generated / 'jvm.options.d/heap.options').write_text('-Xms1g\n-Xmx1g\n', encoding='utf-8')
    (PROJECT / 'results/elasticsearch').mkdir(parents=True, exist_ok=True)
    result = {'version': config['version'], 'archive_available': archive.is_file(),
              'archive_bytes': archive.stat().st_size if archive.is_file() else None,
              'verification_source': 'archive_sha512' if supplied.is_file() else 'previous_verified_installation',
              'official_sha512': config['sha512'], 'es_home': str(es_home), 'configuration': str(generated)}
    (PROJECT / 'results/elasticsearch/installation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
