"""Audited postprocessing recovery; never rerun inference or alter frozen records."""
import argparse
from pathlib import Path
import subprocess
import sys
import traceback
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_c3_supplement import freeze, audit_cells, HoldoutSources, Sources, summarize_cost, now
from bug_localization.batch import write_json, aggregate_results
from bug_localization.reranking import read_json
from bug_localization.snapshot import sha256_file
from bug_localization.paired_statistics import analyze


def complete(folder, mode):
    spec = freeze(folder, mode)
    methods = ['A3'] if mode == 'factorial' else ['A', 'C3']
    source = Sources(spec['config']) if mode == 'factorial' else HoldoutSources(folder, spec['config'])
    record_hashes = {m: sha256_file(folder / m / 'samples.json') for m in methods}
    write_json(folder / 'status.json', {'status': 'running', 'stage': 'postprocessing_recovery', 'updated_at': now()})
    audit_cells(folder, spec)
    summaries = {}
    for method in methods:
        records = read_json(folder / method / 'samples.json')
        normalized = []
        for sample, record in zip(source.samples, records):
            if record['status'] == 'ok':
                case = source.load(sample) if mode == 'factorial' else source.load(sample, method == 'A')
                record = {**record, 'missing_gold': sorted(set(case['gold']) - case['inventory'])}
            normalized.append(record)
        summaries[method] = aggregate_results(normalized, 10)
        write_json(folder / method / 'summary.json', summaries[method])
        write_json(folder / method / 'samples_with_coverage.json', normalized)
    write_json(folder / 'summaries.json', summaries)
    if mode == 'factorial':
        subprocess.run([sys.executable, str(PROJECT / 'scripts/analysis/analyze_c3_supplement.py'),
            '--factorial', str(folder), '--output', str(folder / 'statistics')], check=True)
    else:
        methods = {m: read_json(folder / m / 'samples.json') for m in methods}
        write_json(folder / 'paired_statistics.json', analyze(methods, {'C3_vs_A': {'C3': 1, 'A': -1}}))
        summarize_cost(folder, spec['config'])
    if record_hashes != {m: sha256_file(folder / m / 'samples.json') for m in record_hashes}:
        raise ValueError('Original records modified during recovery')
    write_json(folder / 'postprocessing_recovery.json', {'completed_at': now(),
        'reason': 'Inference completed but original aggregation raised KeyError missing_gold. Reconstructed coverage from saved labels and source inventory; no inference rerun or metric alteration.',
        'original_records_sha256': record_hashes, 'recovery_implementation_sha256': sha256_file(Path(__file__)),
        'original_protocol_sha256': sha256_file(folder / 'protocol.json'), 'independent_audit_completed': True})
    write_json(folder / 'status.json', {'status': 'completed', 'updated_at': now(),
        'postprocessing_recovered': True, 'failed_samples_by_method': {m: s['failed_samples'] for m, s in summaries.items()}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['factorial', 'holdout'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    try:
        complete(a.output.resolve(), a.mode)
    except BaseException as e:
        write_json(a.output / 'status.json', {'status': 'failed', 'stage': 'postprocessing_recovery',
            'error': str(e), 'traceback': traceback.format_exc(), 'updated_at': now()})
        raise

if __name__ == '__main__':
    main()
