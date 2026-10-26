"""Compare rank-free per-route file votes using an existing hybrid result batch."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.voting import run_voting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-run', type=Path, required=True, help='Completed results/hybrid/... directory')
    parser.add_argument('--semantic-label', action='append', help='Repeat labels from the input, e.g. ast96')
    parser.add_argument('--depths', nargs='+', default=['all', '10'], help='Candidate depths: all or positive integers')
    args = parser.parse_args()
    try:
        depths = [None if value == 'all' else int(value) for value in args.depths]
        if any(value is not None and value < 1 for value in depths):
            raise ValueError()
    except ValueError:
        parser.error('--depths must contain all or positive integers')
    source = args.input_run if args.input_run.is_absolute() else PROJECT / args.input_run
    _, summary = run_voting(PROJECT, source, args.semantic_label, depths)
    print(json.dumps({'path_tie_break': {k: v['metrics'] for k, v in summary['routes'].items()},
                      'uniform_tie_expectation': summary['uniform_tie_expectation_routes'],
                      'matched_depth_rrf': {k: v['metrics'] for k, v in summary['matched_depth_rrf_routes'].items()}}, indent=2))


if __name__ == '__main__':
    main()
