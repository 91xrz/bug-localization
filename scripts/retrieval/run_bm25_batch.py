"""Evaluate a small deterministic BM25 batch and export JSON/CSV summaries."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))

from bug_localization.batch import run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/retrieval/bm25_batch.json')
    parser.add_argument('--limit', type=int, help='Override the number of samples (1-50)')
    parser.add_argument('--repository', action='append', help='Repository filter; repeat for multiple repositories')
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else PROJECT / args.config
    batch = json.loads(path.read_text(encoding='utf-8'))
    base_path = Path(batch['base_config'])
    base_path = base_path if base_path.is_absolute() else PROJECT / base_path
    base = json.loads(base_path.read_text(encoding='utf-8'))
    _, summary = run_batch(PROJECT, base, args.repository if args.repository is not None else batch['repositories'],
                           args.limit if args.limit is not None else batch['limit'])
    if summary['failed_samples']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
