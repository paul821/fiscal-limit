import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["estimates", "robustness", "validation", "describe", "oos2026", "paper_figs"]

args = sys.argv[1:]
env = dict(os.environ)
if "--smoke" in args:
    env.update(SILO1_B="99", SILO1_REPS="20")
start = STAGES.index(args[args.index("--from") + 1]) if "--from" in args else 0
log_dir = os.path.join(HERE, "rerun_logs"); os.makedirs(log_dir, exist_ok=True)
t_all = time.time()
for name in STAGES[start:]:
    t = time.time()
    log = os.path.join(log_dir, f"{name}.log")
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, os.path.join(HERE, f"silo1_{name}.py")], cwd=HERE, env=env,
                           stdout=fh, stderr=subprocess.STDOUT)
    print(f"{name:14s} {'ok' if r.returncode == 0 else 'FAILED'}  {time.time() - t:6.0f}s  (log: rerun_logs/{name}.log)", flush=True)
    if r.returncode != 0:
        print(open(log).read()[-2000:])
        sys.exit(1)
print(f"pipeline done in {time.time() - t_all:.0f}s" + ("  [SMOKE RUN: B and panel counts reduced; do not use these outputs]" if "--smoke" in args else ""))
