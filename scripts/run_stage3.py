#!/usr/bin/env python3
"""Durable sequential CPU pipeline, final tests and protected-file verification."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))


def main():
    lock_path=ROOT/'cache/stage3.lock'; lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('w') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('A stage-three runner is already active')
        environment=dict(os.environ); environment['PYTHONHASHSEED']='1980'
        environment['CUDA_VISIBLE_DEVICES']='-1'
        environment['OMP_NUM_THREADS']='2'; environment['OPENBLAS_NUM_THREADS']='2'
        log_path=ROOT/'logs/stage3.log'; log_path.parent.mkdir(exist_ok=True)
        with log_path.open('a') as log:
            process=subprocess.Popen([sys.executable,'-u','scripts/analyze_stage3.py','--section','all'],cwd=ROOT,
                env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout: print(line,end='',flush=True); log.write(line); log.flush()
            if process.wait(): raise SystemExit('Stage-three analysis stopped; inspect logs/stage3.log')
        test_environment=environment.copy()
        test_environment['PYTHONPATH']=str(ROOT/'cache/scene-groups-venv/lib/python3.12/site-packages')
        verification=ROOT/'results/stage3/verification'; verification.mkdir(parents=True,exist_ok=True)
        with (verification/'pytest.log').open('w') as log:
            subprocess.run([sys.executable,'-m','pytest','-q','--disable-warnings'],cwd=ROOT,env=test_environment,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        print((verification/'pytest.log').read_text().splitlines()[-1],flush=True)
        from src.stage3_io import check_protected
        check_protected(ROOT/'results/stage3')
        from scripts.build_stage3_report import build
        build(mode='handoff')
        print('Stage three complete. Author-only names/categories remain blank. Stop here.',flush=True)


if __name__=='__main__': main()
