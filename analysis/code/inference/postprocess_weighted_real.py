"""Add corrected information covariance without rerunning fitted coefficients."""
import argparse
import json
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from run_completion import OUT, CODE, load_real, old, combine, sha, gene_sites
from weighted_information import weighted_gene


def main(workers, wait=False):
    directory = OUT / "weighted_information"
    directory.mkdir(exist_ok=True)
    frames, metrics, audits = [], [], []
    for dataset in ("PDAC","Mouse"):
        clients, genes, types = load_real(dataset)
        k = len(types)
        for offset in ("none","library"):
            family_p = []
            for family in old.FAMILIES:
                key = f"real_{dataset}_{offset}_{family}"
                sites,_ = old.make_sites(clients,family,offset)
                chunks = []
                for start in range(0,len(genes),80):
                    end = min(start+80,len(genes))
                    base = OUT / "refits" / f"{key}_g{start:04d}-{end:04d}.npz"
                    if wait:
                        deadline=time.monotonic()+3600
                        while not all(base.with_suffix(suffix).exists() for suffix in (".npz",".csv",".json")):
                            if time.monotonic()>deadline:
                                raise TimeoutError(f"Refit checkpoint not ready: {base.name}")
                            time.sleep(5)
                    target = directory / base.name
                    if target.exists():
                        chunk = np.load(target)["p_model_weighted"]
                        audit = pd.read_csv(target.with_suffix(".csv"))
                    else:
                        original = np.load(base)
                        valid = pd.read_csv(base.with_suffix(".csv")).valid.to_numpy(bool)
                        results = Parallel(n_jobs=workers)(delayed(weighted_gene)(parameter,
                            gene_sites(sites,start+j),k,valid[j]) for j,parameter in enumerate(original["parameter"]))
                        chunk = np.stack([x[0] for x in results])
                        audit = pd.DataFrame({"gene_id":genes[start:end],"status":[x[1] for x in results]})
                        np.savez_compressed(target,p_model_weighted=chunk)
                        audit.to_csv(target.with_suffix(".csv"),index=False)
                        target.with_suffix(".json").write_text(json.dumps({"fitted_input_sha256":sha(base),
                            "covariance_code_sha256":sha(CODE / "weighted_information.py")},indent=2)+"\n")
                    chunks.append(chunk)
                    audits.append(audit.assign(dataset=dataset,offset=offset,family=family))
                family_p.append(np.concatenate(chunks))
                print(key,"weighted covariance complete",flush=True)
            for adjustment,(p,q) in combine(family_p).items():
                common = dict(dataset=dataset,offset=offset,method="refit_weighted_model",adjustment=adjustment)
                frame = pd.DataFrame(dict(common,gene_id=np.repeat(genes,k),cell_type=np.tile(types,len(genes)),pair_p=p,pair_q=q))
                frames.append(frame)
                metrics.append(dict(common,n_genes=len(genes),n_pairs=len(p),selected_pairs=int((q<.05).sum()),
                                    selected_genes=frame.loc[q<.05,"gene_id"].nunique()))
            pd.concat(frames).to_csv(OUT / "real_weighted_pairs.csv",index=False)
            pd.DataFrame(metrics).to_csv(OUT / "real_weighted_metrics.csv",index=False)
            pd.concat(audits).to_csv(OUT / "real_weighted_covariance_audit.csv",index=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--wait",action="store_true",help="Wait for complete refit checkpoints without changing fits")
    args=parser.parse_args()
    main(args.workers,args.wait)
