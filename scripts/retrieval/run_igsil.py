"""Run IGSIL author file-level localization on pinned project samples."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT / '.runtime/igsil-deps'), str(PROJECT / '.runtime/rrl-deps'), str(PROJECT / 'src')]
from bug_localization.igsil import run

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/retrieval/igsil.json')
    parser.add_argument('--selection')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = json.loads((PROJECT / args.config).read_text())
    if args.selection:
        config['selection_manifest'] = args.selection
    _, summary = run(PROJECT, config, args.output)
    if summary['failed_samples']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
