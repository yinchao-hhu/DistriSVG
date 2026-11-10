"""Run official adapters sequentially and retain every exit code/log."""
from pathlib import Path
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "04_数值实验/results/distrisvg_completion_2026-09-25/official_competitors"
RUNTIME = ROOT / "08_其他/运行环境/distrisvg_2026_09_25"


def main():
    env = os.environ.copy()
    env.update(PATH=str(RUNTIME / "r/bin") + ":" + env["PATH"], OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", RGL_USE_NULL="TRUE")
    jobs = [("CTSV", "run_official_ctsv_benchmark.R", [str(OUT / "source_data"), "0,1,2", "complete,missing"]),
            ("STANCE", "run_stance_compact.R", [str(RUNTIME / "sources/STANCE"), str(OUT / "source_data"), "0,1,2"]),
            ("CELINA", "run_celina_compact.R", [str(RUNTIME / "sources/CELINA"), str(OUT / "source_data"), "0,1,2"]),
            ("spVC", "run_spvc_compact.R", [str(RUNTIME / "sources/spVC/R"), str(OUT / "source_data"), "0,1,2"])]
    status_file = OUT / "run_status.json"
    results = json.loads(status_file.read_text()) if status_file.exists() else {}
    for name, adapter, args in jobs:
        if results.get(name, {}).get("exit_code") == 0:
            continue
        started = time.time()
        command = [str(RUNTIME / "r/bin/Rscript"), "--vanilla", str(OUT / "adapters" / adapter), *args]
        with (OUT / (name + ".log")).open("a") as log:
            log.write("\nExecution started: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            log.flush()
            completed = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        results[name] = dict(exit_code=completed.returncode, elapsed_seconds=time.time()-started, command=command)
        status_file.write_text(json.dumps(results, indent=2) + "\n")
        print(name, results[name]["exit_code"], flush=True)


if __name__ == "__main__":
    main()
