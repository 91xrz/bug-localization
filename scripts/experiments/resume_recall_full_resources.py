"""Resume frozen full-recall checkpoints with a GPU-1 admission amendment.

Only the GPU-1 free-memory gate changes; inference and scoring remain frozen.
"""
import argparse
import fcntl
from pathlib import Path
import sys
import run_recall_full as experiment


def worker_command(command, folder, minimum):
    if '--stage' in command and command[command.index('--stage') + 1] == 'generate1':
        return [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                '--output', str(folder), '--minimum-free-mib', str(minimum), '--worker']
    return command


def worker(folder, minimum):
    original_wait = experiment.wait_gpu

    def amended_wait(output, stage, devices, threshold, config, **extra):
        if stage != 'generate1' or devices != [1]:
            raise ValueError('Resource amendment is restricted to GPU 1 generation')
        return original_wait(output, stage, devices, minimum, config, **extra)

    experiment.wait_gpu = amended_wait
    with (folder / 'generate1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            experiment.generate(folder, 1)
        except Exception as error:
            experiment.state(folder, 'generate1', 'failed', error_type=type(error).__name__, error=str(error))
            raise


def launch(folder, minimum):
    with (folder / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, _, _, _ = experiment.context(folder)
        event = {'recorded_at': experiment.now(), 'reason': 'User requested use of idle GPU 1',
            'old_minimum_free_mib': protocol['config']['generation_min_free_mib'],
            'new_gpu1_minimum_free_mib': minimum,
            'scope': 'GPU admission only; all numerical parameters and implementation remain frozen',
            'original_protocol_sha256': experiment.sha256_file(folder / 'protocol.json'),
            'wrapper_sha256': experiment.sha256_file(Path(__file__)),
            'resume_policy': 'Resume all stages using original verified per-report checkpoints',
            'previous_launcher': experiment.read(folder / 'status/launcher.json')}
        event_path = folder / 'resource_events' / (experiment.now().replace(':', '-') + '.json')
        experiment.save(event_path, event)
        original_popen = experiment.subprocess.Popen

        def amended_popen(command, *args, **kwargs):
            return original_popen(worker_command(command, folder, minimum), *args, **kwargs)

        experiment.subprocess.Popen = amended_popen
        try:
            experiment.launch(folder)
        except Exception as error:
            experiment.state(folder, 'launcher', 'failed', error_type=type(error).__name__, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-free-mib', type=int, default=18500)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if not 18500 <= args.minimum_free_mib <= 19000:
        parser.error('Admission threshold must remain between 18500 and 19000 MiB')
    (worker if args.worker else launch)(args.output.resolve(), args.minimum_free_mib)


if __name__ == '__main__':
    main()
