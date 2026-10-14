#!/usr/bin/env python3
"""Sequential, locked tmux workload: final VIA models, external test, curve, report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from configure_tensorflow_gpu import gpu_environment
from src.study_io import write_json, sha256, source_identity, committed_bytes


def now():
    return datetime.now(timezone.utc).isoformat()


def run_stage(name, arguments, progress, environment):
    log = ROOT/'logs/stage2'/f'{name}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    progress.update(state='running', current_stage=name, updated_utc=now())
    progress['stages'][name] = {'state': 'running', 'started_utc': now(), 'log': str(log.relative_to(ROOT))}
    write_json(ROOT/'results/stage2_verification/progress.json', progress)
    print(f'[{now()}] Starting {name}; log={log.relative_to(ROOT)}', flush=True)
    with log.open('a') as stream:
        process = subprocess.Popen([sys.executable, '-u', *arguments], cwd=ROOT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        progress['stages'][name]['pid'] = process.pid
        write_json(ROOT/'results/stage2_verification/progress.json', progress)
        for line in process.stdout:
            stream.write(line); stream.flush(); print(line, end='', flush=True)
        code = process.wait()
    progress['stages'][name].update(state='completed' if code == 0 else 'failed',
                                    returncode=code, ended_utc=now(), log_sha256=sha256(log))
    history_path = ROOT/'results/stage2_verification/execution_history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append({'stage': name, **progress['stages'][name]})
    write_json(history_path, history)
    progress.update(state='running' if code == 0 else 'failed', updated_utc=now())
    write_json(ROOT/'results/stage2_verification/progress.json', progress)
    if code:
        raise SystemExit(f'{name} failed ({code}); resume only after inspecting {log}')
    print(f'[{now()}] Completed {name}', flush=True)


def recover_finished_checkpoint(name, progress):
    """Restart only an OOM during reload verification after fitting has stopped.

    Scientific sources, checkpoints, final freezes and principal predictions are
    untouched. The native training function resumes its committed stopped state;
    it performs no extra epochs. Its generic OOM fallback also halves the batch
    after fitting: restore that bookkeeping to the batch of the stopped epoch.
    """
    if name not in ('final_models', 'learning_curve'):
        return False
    log = ROOT/'logs/stage2'/f'{name}.log'
    tail = log.read_text()[-12000:]
    if 'expected = model(check_batch, training=False).numpy()' not in tail or 'ResourceExhaustedError' not in tail:
        return False
    if source_identity() != progress['identity']['sources']:
        return False
    directory = ROOT/('results/external_test/models' if name == 'final_models' else 'results/learning_curve')
    candidates = []
    for path in directory.rglob('manifest.json'):
        manifest = json.loads(path.read_text())
        if manifest.get('state') != 'running' or not manifest.get('checkpoint') or (path.parent/'complete.json').exists():
            continue
        checkpoint = path.parent/manifest['checkpoint']/'state.json'
        state = json.loads(checkpoint.read_text())
        if state.get('stopped'):
            candidates.append((path, manifest, checkpoint, state))
    if len(candidates) != 1:
        return False
    path, manifest, checkpoint, state = candidates[0]
    failed_reloads = [r for r in manifest.get('oom_reductions', []) if r['epoch'] == state['epoch']+1]
    if not failed_reloads:
        return False
    record = {'utc': now(), 'stage': name, 'job': str(path.parent.relative_to(ROOT)),
              'reason': 'GPU OOM in eight-image native reload check after early stopping',
              'recovery': 'fresh subprocess, stopped checkpoint, no extra epochs',
              'checkpoint_sha256': sha256(checkpoint), 'stopped_epoch': state['epoch'],
              'previous_manifest': manifest.copy(), 'previous_manifest_sha256': sha256(path),
              'restored_training_batch_size': failed_reloads[0]['from'], 'failed_log_sha256': sha256(log),
              'sources_unchanged': True}
    for filename in ('final_models.json', 'predictions.csv', 'first_evaluation.json'):
        artifact = ROOT/'results/external_test'/filename
        if artifact.exists():
            record[filename+'_sha256'] = sha256(artifact)
    write_json(path.parent/'reload_recovery.json', record)
    manifest['batch_size'] = failed_reloads[0]['from']
    write_json(path, manifest)
    progress.setdefault('recoveries', []).append(record)
    write_json(ROOT/'results/stage2_verification/progress.json', progress)
    print(f'Reload-only OOM: restart {name} from {path.parent.name}, '
          f'stopped epoch {state["epoch"]}; restore training batch {manifest["batch_size"]}', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    args = parser.parse_args()
    lock_path = ROOT/'cache/stage2_external.lock'; lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Another stage-two runner is already active')
        registration = committed_bytes(ROOT/'configs/external_test.json')
        committed_bytes(ROOT/'docs/EXTERNAL_TEST_PROTOCOL.md')
        environment = gpu_environment() if args.device == 'gpu' else dict(__import__('os').environ)
        environment['PYTHONHASHSEED'] = '1980'
        environment['TF_DETERMINISTIC_OPS'] = '1'
        if args.device == 'cpu': environment['CUDA_VISIBLE_DEVICES'] = '-1'
        identity = {'config_sha256': registration['sha256'], 'sources': source_identity(), 'device': args.device}
        path = ROOT/'results/stage2_verification/progress.json'
        previous = json.loads(path.read_text()) if path.exists() else {}
        if previous and previous['identity'] != identity:
            raise SystemExit('Stage-two code/config/device changed; inspect provenance before resuming')
        progress = previous or {'started_utc': now(), 'identity': identity, 'stages': {}}
        stages = [('final_models', ['src/external_test.py', '--stage', 'final-models', '--device', args.device]),
                  ('external_evaluation', ['src/external_test.py', '--stage', 'evaluate', '--device', args.device]),
                  ('external_analysis', ['scripts/analyze_external_test.py']),
                  ('learning_curve', ['src/external_test.py', '--stage', 'learning-curve', '--device', args.device]),
                  ('stage2_report', ['scripts/build_stage2_report.py'])]
        for name, arguments in stages:
            # Completed model/evaluation stages verify frozen artifacts through their own gates.
            # A report may always be rebuilt from fixed results, without new predictions.
            if name in ('external_analysis', 'learning_curve') and progress['stages'].get(name, {}).get('state') == 'completed':
                print(f'Reusing completed {name}', flush=True)
                continue
            for attempt in range(6):
                try:
                    run_stage(name, arguments, progress, environment)
                    break
                except SystemExit:
                    if attempt == 5 or not recover_finished_checkpoint(name, progress):
                        raise
        progress.update(state='completed', current_stage=None, completed_utc=now(), updated_utc=now())
        write_json(path, progress)
        print('Stage two completed; inspect docs/STAGE2_REPORT.md before committing the measured results', flush=True)


if __name__ == '__main__': main()
