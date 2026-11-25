"""Run a minimal RRL author-agent test with the local 7B model adaptation."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/retrieval/rrl.json')
    parser.add_argument('--selection')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--repository', action='append')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = json.loads((PROJECT / args.config).read_text())
    if args.selection:
        config['selection_manifest'] = args.selection
    if config.get('selection_manifest') and (args.limit is not None or args.repository):
        parser.error('A frozen selection cannot be combined with limit/repository overrides')
    if args.limit is not None:
        config['limit'] = args.limit
    if args.repository:
        config['repositories'] = args.repository
    sys.path[:0] = [str(PROJECT / config['dependency_path']), str(PROJECT / 'src')]
    from bug_localization.rrl import run
    _, summary = run(PROJECT, config, args.output)
    if summary['failed_samples']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
