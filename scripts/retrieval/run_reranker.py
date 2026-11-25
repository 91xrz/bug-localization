"""Rerank a saved semantic batch and compare the exact same reports/candidates."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))

from bug_localization.reranking import run_reranking
from bug_localization.vector import RETRIEVAL_ROUTES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantic-run', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/retrieval/reranker.json')
    parser.add_argument('--routes', nargs='+', choices=RETRIEVAL_ROUTES, default=['blagent_ast'])
    parser.add_argument('--limit', type=int, help='First N source samples, for smoke checks')
    parser.add_argument('--device', help='cpu, cuda, or cuda:0 etc.; honors CUDA_VISIBLE_DEVICES')
    parser.add_argument('--top-n-files', type=int)
    parser.add_argument('--chunks-per-file', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--resume', type=Path, help='Continue an existing run with exactly the same selection/settings')
    args = parser.parse_args()
    config = json.loads((PROJECT / args.config).read_text(encoding='utf-8'))
    for key in ('device', 'top_n_files', 'chunks_per_file', 'batch_size'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    folder, summary = run_reranking(PROJECT, args.semantic_run, config, args.routes, args.limit, args.resume)
    print('Reranking results: ' + str(folder), flush=True)
    if summary['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
