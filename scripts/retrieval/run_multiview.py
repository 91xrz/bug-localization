"""Generate report views once, or evaluate cached views with existing BM25/AST retrieval."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.multiview import prepare_queries, run_multiview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/experiments/report_multiview.json')
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument('--prepare', action='store_true', help='Generate/validate/cache queries; no ES or embeddings')
    phase.add_argument('--run', action='store_true', help='Use cached queries only; LLM server may be stopped')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--selection', type=str)
    args = parser.parse_args()
    if args.limit is not None and args.selection:
        parser.error('--limit and --selection are mutually exclusive')
    config = json.loads((PROJECT / args.config).read_text(encoding='utf-8'))
    if args.limit is not None:
        config.update(limit=args.limit, selection_manifest=None)
    if args.selection:
        config['selection_manifest'] = args.selection
    folder, summary = (prepare_queries if args.prepare else run_multiview)(PROJECT, config)
    print('Results:', folder)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
