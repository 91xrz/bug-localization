"""Extract bundled LCA repositories, audit base commits and prepare offline inputs."""
import argparse
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.lca import prepare, save

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=PROJECT / 'datasets/LCA_bug_localization')
    parser.add_argument('--languages', nargs='+', choices=['java', 'py', 'kt'], default=['java', 'py', 'kt'])
    parser.add_argument('--no-test-snapshots', action='store_true')
    args = parser.parse_args()
    try:
        result = prepare(PROJECT, args.raw_root.resolve(), args.languages, not args.no_test_snapshots)
    except Exception as error:
        progress = args.raw_root / 'metadata/preparation-status.json'
        state = json.loads(progress.read_text()) if progress.exists() else {}
        state.update({'status': 'failed', 'error': str(error)})
        save(progress, state)
        raise
    print(json.dumps(result, indent=2))
