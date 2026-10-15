"""Compare BM25 and dynamic/method/fixed semantic retrieval on pinned reports; no RRF."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.comparison import run_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/retrieval/comparison.json')
    parser.add_argument('--limit', type=int, help='Override sample count (1-50)')
    parser.add_argument('--repository', action='append', help='Repository filter; repeat to add repositories')
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else PROJECT / args.config
    config = json.loads(path.read_text(encoding='utf-8'))
    if args.limit is not None:
        config['limit'] = args.limit
    if args.repository:
        config['repositories'] = args.repository
    _, report = run_comparison(PROJECT, config)
    if report['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
