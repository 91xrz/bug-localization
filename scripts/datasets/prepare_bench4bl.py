"""Convert original Bench4BL modern subjects into the existing HF input schema."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.bench4bl import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=PROJECT / 'datasets/Bench4BL')
    parser.add_argument('--output', type=Path, default=PROJECT / 'datasets/bench4bl_formatted')
    parser.add_argument('--subjects', nargs='+', help='Optional exact Group/Subject list; defaults to all 46 modern subjects')
    args = parser.parse_args()
    result = prepare(PROJECT, PROJECT / args.raw_root, PROJECT / args.output, args.subjects)
    print(json.dumps({k: v for k, v in result.items() if k != 'projects'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
