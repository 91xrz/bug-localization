"""Evaluate the pinned BLAZE released encoder (shared-protocol adaptation)."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.blaze_experiment import run_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/retrieval/blaze.json')
    parser.add_argument('--selection', help='Existing frozen selection manifest (up to 5000 reports)')
    parser.add_argument('--limit', type=int, help='First matching reports in dataset order, 1-50')
    parser.add_argument('--repository', action='append')
    parser.add_argument('--device', help='cpu (default), cuda, or explicit cuda:N')
    parser.add_argument('--output', type=Path, help='New output directory; existing runs cannot be overwritten')
    args = parser.parse_args()
    config = json.loads((PROJECT / args.config).read_text(encoding='utf-8'))
    if args.selection:
        config['selection_manifest'] = args.selection
    if config.get('selection_manifest') and (args.limit is not None or args.repository):
        parser.error('A frozen selection cannot be combined with limit/repository overrides')
    if args.limit is not None:
        config['limit'] = args.limit
    if args.repository:
        config['repositories'] = args.repository
    if args.device:
        config['device'] = args.device
    _, summary = run_batch(PROJECT, config, args.output)
    if any(route['failed_samples'] for route in summary['routes'].values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
