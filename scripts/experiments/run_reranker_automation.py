"""Resume reranking and audit automatically; intended for detached tmux runs."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import write_json
from bug_localization.reranking import run_reranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    args = parser.parse_args()
    folder = (PROJECT / args.run).resolve()
    effective = json.loads((folder / 'effective_config.json').read_text())
    status = {'pid': os.getpid(), 'run': str(folder), 'tmux_session': 'irbl-reranker-180'}

    def update(stage, **extra):
        status.update(stage=stage, updated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(), **extra)
        write_json(folder / 'automation_status.json', status)

    try:
        update('reranking')
        _, summary = run_reranking(PROJECT, Path(effective['source_run']), effective['reranker'],
                                   effective['routes'], resume_folder=folder)
        if summary['status'] != 'completed':
            raise RuntimeError('Reranking contains failures; inspect samples.json')
        update('auditing')
        subprocess.run([sys.executable, str(PROJECT / 'scripts/analysis/analyze_reranking.py'),
                        '--run', str(folder), '--selection', str(PROJECT / args.selection)], check=True, cwd=PROJECT)
        write_json(folder / 'final_summary.json', {
            'summary': summary,
            'projects': json.loads((folder / 'project_summary.json').read_text()),
            'verification': json.loads((folder / 'verification.json').read_text()),
        })
        update('completed', exit_code=0)
    except BaseException as error:
        update('failed', exit_code=1, error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
