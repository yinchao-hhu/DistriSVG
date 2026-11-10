"""Replay spVC with immediate warnings, preserving the first official outputs."""
from pathlib import Path
import argparse
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


def main(compare_only=False):
    target=BASE/"spvc_warning_audit"
    target.mkdir(exist_ok=True)
    for pattern in ("official_method_gene_panel.csv","repeat*_complete_counts.csv",
                    "repeat*_complete_coordinates.csv","repeat*_complete_omega.csv",
                    "repeat*_complete_truth.csv"):
        for path in (BASE/"source_data").glob(pattern): shutil.copy2(path,target/path.name)
    adapter=BASE/"adapters/run_spvc_compact.R"
    expression="options(warn=1); source("+json.dumps(str(adapter),ensure_ascii=False)+")"
    command=[str(RUNTIME/"r/bin/Rscript"),"--vanilla","-e",expression,
             str(RUNTIME/"sources/spVC/R"),str(target),"0,1,2"]
    env=os.environ.copy()
    env.update(PATH=str(RUNTIME/"r/bin")+":"+env["PATH"],OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",RGL_USE_NULL="TRUE")
    if not compare_only:
        start=time.time()
        with (target/"run.log").open("w") as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record=dict(command=command,exit_code=result.returncode,elapsed_seconds=time.time()-start)
        (target/"run_status.json").write_text(json.dumps(record,indent=2)+"\n")
        result.check_returncode()
    comparisons=[]
    for rep in range(3):
        name=f"repeat{rep}_complete_spvc_pairs.csv"
        first=pd.read_csv(BASE/"source_data"/name).set_index(["gene_id","cell_type"])
        replay=pd.read_csv(target/name).set_index(["gene_id","cell_type"]).loc[first.index]
        error=float(np.max(np.abs(first.pair_p-replay.pair_p)))
        comparisons.append(dict(repeat=rep,max_pair_p_difference=error,
            numerically_equal=bool(np.allclose(first.pair_p,replay.pair_p,rtol=1e-10,atol=1e-14)),
            first_selected_pairs=int((first.pair_bh_q<.05).sum()),
            replay_selected_pairs=int((replay.pair_bh_q<.05).sum())))
    pd.DataFrame(comparisons).to_csv(target/"output_comparison.csv",index=False)
    print(json.dumps(comparisons,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--compare-only",action="store_true")
    main(parser.parse_args().compare_only)
