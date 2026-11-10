"""Freeze spVC's official random boundary branch and verify an identical replay."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/"04_数值实验/results/distrisvg_completion_2026-09-25/official_competitors"
RUNTIME=ROOT/"08_其他/运行环境/distrisvg_2026_09_25"


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=BASE/"spvc_seeded_audit"; out.mkdir(exist_ok=True)
    original=BASE/"adapters/run_spvc_compact.R"
    target=out/"run_spvc_seeded.R"
    text=original.read_text()
    marker='  started <- proc.time()[["elapsed"]]\n  fit <- test.spVC('
    assert text.count(marker)==1
    text="options(warn=1)\n"+text.replace(marker,'  set.seed(20260925L + repeat_id)\n'+marker)
    if target.exists(): assert target.read_text()==text
    else: target.write_text(text)
    source=RUNTIME/"sources/spVC/R/testStat.R"
    manifest=dict(original_adapter_sha256=digest(original),seeded_adapter_sha256=digest(target),
        seed_rule="20260925 + repeat_id, immediately before each test.spVC call",
        official_testStat_sha256=digest(source),
        note="Official pval==0.5 or pval==1 runif(1) branch is retained, not repaired or removed.")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    env=os.environ.copy()
    env.update(PATH=str(RUNTIME/"r/bin")+":"+env["PATH"],OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",RGL_USE_NULL="TRUE")
    records=[]
    for run in ("first","replay"):
        directory=out/run; directory.mkdir(exist_ok=True)
        for pattern in ("official_method_gene_panel.csv","repeat*_complete_counts.csv",
                        "repeat*_complete_coordinates.csv","repeat*_complete_omega.csv",
                        "repeat*_complete_truth.csv"):
            for path in (BASE/"source_data").glob(pattern): shutil.copy2(path,directory/path.name)
        status=directory/"run_status.json"
        if status.exists() and json.loads(status.read_text())["exit_code"]==0:
            records.append(json.loads(status.read_text())); continue
        command=[str(RUNTIME/"r/bin/Rscript"),"--vanilla",str(target),
                 str(RUNTIME/"sources/spVC/R"),str(directory),"0,1,2"]
        start=time.time()
        with (directory/"run.log").open("w") as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record=dict(run=run,command=command,exit_code=result.returncode,elapsed_seconds=time.time()-start)
        status.write_text(json.dumps(record,indent=2)+"\n"); result.check_returncode()
        records.append(record)
        print(run,"completed",flush=True)
    comparisons=[]
    for rep in range(3):
        name=f"repeat{rep}_complete_spvc_pairs.csv"
        first=pd.read_csv(out/"first"/name).set_index(["gene_id","cell_type"])
        replay=pd.read_csv(out/"replay"/name).set_index(["gene_id","cell_type"]).loc[first.index]
        error=float(np.max(np.abs(first.pair_p-replay.pair_p)))
        assert np.allclose(first.pair_p,replay.pair_p,rtol=1e-10,atol=1e-14),error
        comparisons.append(dict(repeat=rep,max_pair_p_difference=error,selected_pairs=int((first.pair_bh_q<.05).sum())))
    pd.DataFrame(comparisons).to_csv(out/"output_comparison.csv",index=False)
    (out/"run_status.json").write_text(json.dumps(dict(status="identical_replay_verified",runs=records),indent=2)+"\n")
    print(json.dumps(comparisons,indent=2))


if __name__=="__main__":main()
