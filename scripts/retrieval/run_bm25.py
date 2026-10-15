"""Run the configured end-to-end lexical retrieval example."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))

from bug_localization.bm25 import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/retrieval/bm25.json')
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT / args.config
    config = json.loads(config_path.read_text(encoding='utf-8'))
    run(PROJECT, config)


if __name__ == '__main__':
    main()
