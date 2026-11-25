"""Run BLAgent author-agent adaptation on frozen project samples."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / p) for p in ('.runtime/blagent-deps', '.runtime/rrl-deps', '.runtime/igsil-deps', 'src')]
from bug_localization.blagent import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/retrieval/blagent.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = json.loads((PROJECT / args.config).read_text())
    result = run(PROJECT, config, args.output, check_only=args.check_only)
    if args.check_only:
        print(json.dumps(result, indent=2))
    else:
        folder, summary = result
        print('Results:', folder)
        print(json.dumps(summary, indent=2))
        if summary['failed_samples'] or summary['pending_samples']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
