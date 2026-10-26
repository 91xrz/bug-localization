"""Fuse paired saved BM25 and semantic file rankings with equal-weight RRF."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.hybrid import run_hybrid
from bug_localization.vector import RETRIEVAL_ROUTES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bm25-run', type=Path, required=True, help='Saved BM25 batch directory')
    parser.add_argument('--semantic-run', action='append', required=True, metavar='LABEL=PATH',
                        help='Repeat for independent semantic controls, e.g. ast64=results/vector/batches/...')
    parser.add_argument('--route', choices=RETRIEVAL_ROUTES, default='blagent_ast')
    parser.add_argument('--rrf-constant', type=int, default=60)
    args = parser.parse_args()
    folders = {}
    for value in args.semantic_run:
        label, separator, path = value.partition('=')
        if not separator or not path or label in folders:
            parser.error('--semantic-run requires a unique LABEL=PATH')
        folder = Path(path)
        folders[label] = folder if folder.is_absolute() else PROJECT / folder
    lexical = args.bm25_run if args.bm25_run.is_absolute() else PROJECT / args.bm25_run
    _, summary = run_hybrid(PROJECT, lexical, folders, args.route, args.rrf_constant)
    print(json.dumps({name: values['metrics'] for name, values in summary['routes'].items()}, indent=2))


if __name__ == '__main__':
    main()
