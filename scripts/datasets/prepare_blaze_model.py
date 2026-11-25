"""Prepare and verify the pinned public BLAZE encoder and CodeSage source."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.blaze_model import prepare_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=PROJECT / 'models/BLAZE-release')
    parser.add_argument('--reuse-weights', type=Path, default=PROJECT / 'models/BlazeModel/model.safetensors')
    args = parser.parse_args()
    identity = prepare_model(args.destination, args.reuse_weights)
    print(json.dumps(identity, indent=2))


if __name__ == '__main__':
    main()
