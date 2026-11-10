"""Read-only replay of current generator defaults against preserved large inputs.

Candidate replay is a provenance check, not replacement data or model fitting.
The historical generation script's output-writing entrypoint is never called.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "04_数值实验/code/new_test1-5"
OUT = ROOT / "04_数值实验/results/distrisvg_completion_2026-09-25"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    source = ROOT / "analysis/code/simulation_complete/generate.py"
    spec = importlib.util.spec_from_file_location("retained_generator",source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows, files = [], {str(source.relative_to(ROOT)):sha(source)}
    for mode in ("linear","gau","cos"):
        torch.manual_seed(42); np.random.seed(42)
        coordinates = []
        for client in range(8):
            torch.manual_seed(42+client); np.random.seed(42+client)
            coordinates.append(torch.rand(500,2))
        basis = module.BasisGenerator(mode=mode,s_coords=torch.cat(coordinates))
        base = torch.normal(mean=2.,std=.2,size=(1000,))
        theta = []
        for k in range(5):
            current = base.clone()
            if k>0:
                idx = np.random.choice(np.arange(1000),300,replace=False)
                current[idx] = torch.normal(mean=3.+(k%2)+.5*(k//2),std=.2,size=(300,))
            theta.append(current)
        theta = torch.stack(theta).unsqueeze(0)
        beta = torch.zeros(2,5,1000)
        for k in range(5):
            begin=(k+1)*100; end=min(begin+200,1000)
            magnitude=torch.rand((2,end-begin))*3.+2.
            signs=torch.randint(0,2,(2,end-begin)).float()*2-1
            beta[:,k,begin:end]=magnitude*signs
        inputs = BASE / "sim_data" / f"Input_Omega_S_8_G_1000_L1_2e-07_Round1000_h_{mode}.csv"
        counts = BASE / "sim_data" / f"AllClients_Y_8_G_1000_L1_2e-07_Round1000_h_{mode}.csv"
        truth = BASE / "sim_data" / f"GlobalTrue_Beta_G1000_K5_L1_2e-07_h_{mode}.csv"
        for path in (inputs,counts,truth): files[str(path.relative_to(ROOT))]=sha(path)
        x,y,t = pd.read_csv(inputs),pd.read_csv(counts),pd.read_csv(truth)
        missing_path=ROOT/"04_数值实验/code/missing_celltype_test/sim_data"/counts.name
        missing_y=pd.read_csv(missing_path)
        files[str(missing_path.relative_to(ROOT))]=sha(missing_path)
        expected = np.array([beta[int(r.Feature[1:])-1,int(r.Cell_Type),int(r.Gene_ID)].item() for r in t.itertuples()])
        beta_error=float(np.max(np.abs(expected-t.True_Beta.to_numpy())))
        for client in range(8):
            generated=module.generate_synthetic_data(500,5,1000,basis,beta,theta,
                alpha0=100,mask_genes_indices=list(range(client*125,client*125+50)),seed=42+client,pi0=.4,psi_val=10.)
            xs=x[x.Client_ID==client].sort_values("Spot_ID")
            ys=y[y.Client_ID==client].sort_values("Spot_ID").filter(regex="^Gene_").to_numpy()
            sy=generated["y"].numpy()
            my=missing_y[missing_y.Client_ID==client].sort_values("Spot_ID").filter(regex="^Gene_").to_numpy()
            rows.append(dict(mode=mode,client=client,coordinates_max_error=float(np.max(np.abs(generated["s"].numpy()-xs[["Coord_X","Coord_Y"]].to_numpy()))),
                omega_max_error=float(np.max(np.abs(generated["omega"].numpy()-xs.filter(regex="^Omega_Cell_").to_numpy()))),
                beta_max_error=beta_error,count_entry_agreement=float(np.mean(sy==ys)),
                count_max_error=float(np.max(np.abs(sy-ys))),stored_zero_fraction=float(np.mean(ys==0)),
                candidate_zero_fraction=float(np.mean(sy==0)),missing_counts_identical_file=sha(missing_path)==sha(counts),
                missing_count_entry_agreement=float(np.mean(my==ys)),
                candidate_mismatched_entries=int(np.sum(sy!=ys)),
                candidate_omega_base_vs_supplied_mae=float(np.mean(np.abs(generated["omega_base"].numpy()-generated["omega"].numpy())))))
        print(mode,"checked",flush=True)
    pd.DataFrame(rows).to_csv(OUT/"large_generator_provenance_audit.csv",index=False)
    (OUT/"large_generator_provenance_manifest.json").write_text(json.dumps(dict(
        scope="Current-source default replay only; no archived data replacement or unknown historical settings inferred",
        torch=torch.__version__, numpy=np.__version__, candidate_pi0=.4,candidate_psi=10.,
        files=files),indent=2)+"\n")


if __name__=="__main__":
    main()
