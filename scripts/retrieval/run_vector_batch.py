"""Evaluate dynamic/method/fixed/BLagent AST Qwen retrieval and file aggregations at @10."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))

from bug_localization.vector import RETRIEVAL_ROUTES, run_semantic_batch
from bug_localization.aggregation import AGGREGATIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/retrieval/vector.json')
    parser.add_argument('--selection', help='Pin an existing BM25 selection.json; cannot combine with limit/repository')
    parser.add_argument('--limit', type=int, help='First matching samples in dataset order (1-50)')
    parser.add_argument('--repository', action='append', help='Repeat to select multiple repositories')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--aggregation', choices=AGGREGATIONS, help='Primary file ranking; all three aggregations are evaluated')
    parser.add_argument('--temperature', type=float, help='Normalized LogSumExp temperature (default 0.1)')
    parser.add_argument('--routes', nargs='+', choices=RETRIEVAL_ROUTES, help='Chunking routes to evaluate')
    parser.add_argument('--ast-max-chars', type=int, help='BLagent AST sibling-packing character budget (default 1500)')
    fusion = parser.add_mutually_exclusive_group()
    fusion.add_argument('--fusion', dest='enable_fusion', action='store_true', default=None,
                        help='Also evaluate the two-route and three-route RRF controls')
    fusion.add_argument('--no-fusion', dest='enable_fusion', action='store_false',
                        help='Evaluate only the three independent semantic routes')
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else PROJECT / args.config
    config = json.loads(path.read_text(encoding='utf-8'))
    if args.selection and (args.limit is not None or args.repository):
        parser.error('--selection cannot be combined with --limit or --repository')
    if args.selection:
        config['selection_manifest'] = args.selection
    if args.limit is not None or args.repository:
        config['selection_manifest'] = None
    if args.limit is not None:
        config['limit'] = args.limit
    if args.repository:
        config['repositories'] = args.repository
    if args.device:
        config['device'] = args.device
    if args.aggregation:
        config['file_aggregation'] = args.aggregation
    if args.temperature is not None:
        config['aggregation_temperature'] = args.temperature
    if args.routes:
        config['chunking_routes'] = args.routes
    if args.ast_max_chars is not None:
        config['ast_max_chars'] = args.ast_max_chars
    if args.enable_fusion is not None:
        config['enable_fusion'] = args.enable_fusion
    _, summary = run_semantic_batch(PROJECT, config)
    if any(route['failed_samples'] for route in summary['routes'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
