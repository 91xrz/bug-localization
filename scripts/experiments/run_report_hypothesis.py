"""One-command generation, retrieval, audit, and final summary; detailed output stays in logs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from bug_localization.batch import write_json, aggregate_results


def execute(command, log, environment):
    with log.open('w') as output:
        result = subprocess.run(command, cwd=PROJECT, env=environment, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Stage failed with exit code {result.returncode}; see {log}')


def result_folder(log):
    for line in log.read_text().splitlines():
        if line.startswith('Results: '):
            return Path(line[len('Results: '):])
    raise ValueError('No result directory reported in ' + str(log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/experiments/report_hypothesis_ast64.json')
    parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '1'), help='Physical GPU IDs, default 1')
    parser.add_argument('--resume-retrieval', type=Path, help='Wait for an already-running batch and audit it; no repeat retrieval')
    parser.add_argument('--wait-timeout', type=int, default=7200, help='Maximum seconds to wait for an attached batch')
    args = parser.parse_args()
    if args.wait_timeout < 1:
        parser.error('--wait-timeout must be positive')
    logs = PROJECT / 'results/report_hypothesis/automation' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    logs.mkdir(parents=True)
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu}
    write_json(logs / 'status.json', {'status': 'running', 'pid': os.getpid(), 'config': str(args.config)})
    try:
        if args.resume_retrieval:
            folder = (PROJECT / args.resume_retrieval).resolve()
            deadline = time.monotonic() + args.wait_timeout
            while True:
                status = json.loads((folder / 'status.json').read_text())
                if status['status'] == 'completed':
                    break
                if status['status'] == 'failed':
                    raise RuntimeError(f'Retrieval failed; see {folder / "status.json"}')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Attached retrieval did not finish within --wait-timeout')
                time.sleep(5)
        else:
            config_path = args.config.resolve()
            config = json.loads(config_path.read_text())
            if config.get('enrichment_config'):
                prepare_command = [sys.executable, '-B', '-u', str(PROJECT / 'scripts/retrieval/run_report_enrichment.py'),
                                   '--config', str(PROJECT / config['enrichment_config']), '--prepare']
                write_json(logs / 'status.json', {'status': 'running', 'phase': 'enrichment', 'pid': os.getpid()})
                execute(prepare_command, logs / 'enrichment.log', environment)
                config['enrichment_source'] = str(result_folder(logs / 'enrichment.log'))
                config_path = logs / 'resolved_config.json'
                write_json(config_path, config)
            runner = [sys.executable, '-B', '-u', str(PROJECT / 'scripts/retrieval/run_report_enrichment.py'),
                      '--config', str(config_path)]
            write_json(logs / 'status.json', {'status': 'running', 'phase': 'hypothesis_generation', 'pid': os.getpid()})
            execute(runner + ['--prepare'], logs / 'prepare.log', environment)
            # Separate processes ensure the generator releases its GPU before embedding starts.
            write_json(logs / 'status.json', {'status': 'running', 'phase': 'retrieval', 'pid': os.getpid()})
            execute(runner + ['--run'], logs / 'retrieval.log', environment)
            folder = result_folder(logs / 'retrieval.log')
        write_json(logs / 'status.json', {'status': 'running', 'phase': 'audit', 'pid': os.getpid(), 'result_directory': str(folder)})
        execute([sys.executable, '-B', str(PROJECT / 'scripts/analysis/analyze_report_hypothesis.py'), str(folder)],
                logs / 'audit.log', environment)
        verification = json.loads((folder / 'verification.json').read_text())
        audit = json.loads((folder / 'hypothesis_verification.json').read_text())
        rows = verification['comparison']
        direct = [r for r in rows if r['query'] in ('context', 'functionality', 'logic',
                  'context_functionality', 'functionality_logic', 'hypothesis_all')]
        fusion = [r for r in rows if r['query'].startswith('rrf_')]
        samples = json.loads((folder / 'samples.json').read_text())
        repositories = sorted({r['repository'] for r in samples['q0']})
        project_summary = {repo: {name: aggregate_results([r for r in records if r['repository'] == repo], 10)
                            for name, records in samples.items()} for repo in repositories}
        write_json(folder / 'project_summary.json', project_summary)
        summary = {'status': 'completed', 'result_directory': str(folder),
            'samples': len(audit['cases']), 'routes': len(rows), 'projects': repositories,
            'fixed_parameter_validation': len(fusion) == 1 and len(direct) == 1,
            'baseline': next(r for r in rows if r['query'] == 'q0'),
            'enrichment_control': next(r for r in rows if r['query'] == 'OB_EB_S2R'),
            'best_direct_hypothesis': max(direct, key=lambda r: r['MAP@10']),
            'best_fusion': max(fusion, key=lambda r: r['MAP@10']),
            'verification_passed': audit['verified'] and verification['metrics_and_fusion_verified'],
            'historical_control_summary': audit['historical_control_summary'],
            'scope': verification['scope'], 'logs': str(logs)}
        write_json(folder / 'final_summary.json', summary)
        write_json(logs / 'status.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except BaseException as error:
        write_json(logs / 'status.json', {'status': 'failed', 'error': str(error)})
        print(f'Automation failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
