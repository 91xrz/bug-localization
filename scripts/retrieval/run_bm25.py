"""Run single-report or small-batch BM25 lexical comparison experiments."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))

from bug_localization.bm25 import run
from bug_localization.batch import run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/retrieval/bm25.json')
    parser.add_argument('--batch', action='store_true', help='Use a BM25 or comparison config for a batch')
    parser.add_argument('--limit', type=int, help='Batch sample count (1-50); defaults to config limit or 5')
    parser.add_argument('--repository', action='append', help='Batch repository filter; repeat as needed')
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT / args.config
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if args.batch:
        base = config
        if 'bm25_config' in config:
            base = json.loads((PROJECT / config['bm25_config']).read_text(encoding='utf-8'))
        repositories = args.repository if args.repository is not None else config.get('repositories', [base['repository']])
        limit = args.limit if args.limit is not None else config.get('limit', 5)
        _, summary = run_batch(PROJECT, base, repositories, limit)
        if summary['failed_samples']:
            raise SystemExit(1)
    else:
        if args.limit is not None or args.repository is not None or 'bm25_config' in config:
            parser.error('Batch options and comparison configs require --batch')
        run(PROJECT, config)


if __name__ == '__main__':
    main()
