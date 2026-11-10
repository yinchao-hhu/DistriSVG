#!/usr/bin/env python3
"""Matched-input offset pilots, not replacements for archived real-data analyses.

Each dataset uses a fixed 160-gene panel. Full input matrices supply library
sizes. Both paired fits use the same canonical mean, fixed dispersion, basis,
budget, penalty and site weights. There is no biological ground truth.
"""
import argparse
import json

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

from metrics import acat, bh
from nb_core import diagonal_score_p, loss_gradient
import run_revision as common


def pdac_inputs():
    root = common.ROOT / "04_数值实验/code/pdac_data_to_yc"
    clients, genes, cell_types = [], None, None
    for patient in "AB":
        for number in range(1, 4):
            section = f"PDAC_{patient}_ST{number}"
            expr = common.read(root / "preprocessed_data" / f"{section}_processed_data_mincell5.csv").set_index("Spot_ID")
            om = common.read(root / "spotlight_out" / f"{section}_SPOTlight_W_omega_with_xy.csv").set_index("spot")
            current_genes = [x for x in expr.columns if x not in ("x", "y", "X", "Y")]
            if genes is None:
                genes = np.random.default_rng(common.CONFIG["real_panel_seed"]).choice(sorted(current_genes), size=160, replace=False).tolist()
                cell_types = [x for x in om.columns if x not in ("x", "y")]
            if not expr.index.is_unique or not om.index.is_unique:
                raise ValueError("Non-unique PDAC spots")
            spots = expr.index[expr.index.isin(om.index)]
            expr, om = expr.loc[spots], om.loc[spots]
            weights = om[cell_types].to_numpy(float)
            weights /= weights.sum(1, keepdims=True)
            clients.append(dict(y=expr[genes].to_numpy(float), coords=om[["x", "y"]].to_numpy(float),
                                omega=weights, library=expr[current_genes].sum(1).to_numpy(float), section=section))
    return clients, genes, cell_types


def mouse_inputs():
    root = common.ROOT / "04_数值实验"
    panel = common.read(root / "figures/mouse_reference_deconv/mouse_reference_nnls_gene_panel.csv")
    genes = panel.gene.iloc[:160].tolist()
    cell_types = ["Astrocyte", "Endothelial Cell", "GABA-ergic Neuron", "Glutamatergic Neuron",
                  "Microglia", "Oligodendrocyte", "Oligodendrocyte Precursor Cell"]
    sections = {"Anterior": "V1_Mouse_Brain_Sagittal_Anterior", "Posterior": "V1_Mouse_Brain_Sagittal_Posterior",
                "Anterior2": "V1_Mouse_Brain_Sagittal_Anterior_Section_2"}
    clients = []
    for name, section in sections.items():
        path = common.track(root / "data" / section / "filtered_feature_bc_matrix.h5")
        with h5py.File(path, "r") as f:
            m = f["matrix"]
            counts = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
            gene_names = pd.Index([x.decode() for x in m["features/name"][:]])
            barcodes = pd.Index([x.decode() for x in m["barcodes"][:]])
        omega = common.read(root / "figures/mouse_reference_deconv" / f"mouse_reference_nnls_omega_{name}.csv").set_index("spot_id")
        spot_indices = barcodes.get_indexer(omega.index)
        if np.any(spot_indices < 0):
            raise ValueError("Mouse barcode missing from input counts")
        gene_indices = [int(np.flatnonzero(gene_names == gene)[0]) for gene in genes]
        weights = omega[cell_types].to_numpy(float)
        weights /= weights.sum(1, keepdims=True)
        clients.append(dict(y=counts[gene_indices][:, spot_indices].T.toarray().astype(float),
                            coords=omega[["x", "y"]].to_numpy(float), omega=weights,
                            library=np.asarray(counts[:, spot_indices].sum(0)).ravel(), section=name))
    return clients, genes, cell_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    common.OUTPUT.mkdir(parents=True, exist_ok=True)
    if (common.OUTPUT / "run_manifest.json").exists():
        common.INPUTS.update(json.loads((common.OUTPUT / "run_manifest.json").read_text())["inputs"])
    summaries, all_pairs, sensitivity, audits = [], [], [], []
    for dataset, loader in (("PDAC", pdac_inputs), ("Mouse", mouse_inputs)):
        clients, genes, types = loader()
        k = len(types)
        pd.DataFrame({"gene_id": genes}).to_csv(common.OUTPUT / f"{dataset.lower()}_offset_gene_panel.csv", index=False)
        for client in clients:
            audits.append(dict(dataset=dataset, section=client["section"], n_spots=len(client["y"]),
                               library_min=float(client["library"].min()), library_median=float(np.median(client["library"])),
                               library_max=float(client["library"].max()), n_genes=160, n_cell_types=k))
        per_offset = {}
        for offset in ("none", "library"):
            values, seconds, beta_norms, losses = [], 0., [], []
            for family in common.FAMILIES:
                sites, _ = common.make_sites(clients, family, offset)
                coef, elapsed = common.run_fit(f"real_{dataset}_{offset}_{family}", sites, k, "AP-FedAdam", resume=args.resume)
                seconds += elapsed
                values.append(acat(diagonal_score_p(coef, sites, k), axis=0))
                beta_norms.append(np.linalg.norm(coef[k:].reshape(2, k, len(genes)), axis=0))
                losses.append(np.mean([loss_gradient(coef, site, 12.)[0] for site in sites]))
            p = acat(np.stack(values), axis=0).ravel()
            q = bh(p)
            norm = np.max(beta_norms, axis=0).ravel()
            pair = pd.DataFrame(dict(dataset=dataset, offset=offset, gene_id=np.tile(genes, k),
                                     cell_type=np.repeat(types, len(genes)), pair_p=p, pair_q=q, beta_norm=norm))
            all_pairs.append(pair)
            summaries.append(dict(dataset=dataset, offset=offset, n_genes=len(genes), n_pairs=len(pair),
                                  n_spots=sum(len(c["y"]) for c in clients), selected_pairs=int((q < .05).sum()),
                                  selected_genes=pair.loc[q < .05, "gene_id"].nunique(),
                                  mean_family_nll=float(np.mean(losses)), elapsed_seconds=seconds))
            per_offset[offset] = pair
        a, b = per_offset["none"], per_offset["library"]
        topa = set(np.argsort(a.pair_p.to_numpy(), kind="stable")[:100])
        topb = set(np.argsort(b.pair_p.to_numpy(), kind="stable")[:100])
        qa, qb = set(np.flatnonzero(a.pair_q < .05)), set(np.flatnonzero(b.pair_q < .05))
        sensitivity.append(dict(dataset=dataset, spearman_p=float(spearmanr(a.pair_p, b.pair_p).statistic),
                                spearman_beta_norm=float(spearmanr(a.beta_norm, b.beta_norm).statistic),
                                top100_overlap=len(topa & topb), top100_jaccard=len(topa & topb) / len(topa | topb),
                                selected_jaccard=len(qa & qb) / len(qa | qb) if qa | qb else np.nan,
                                n_tied_min_none=int((a.pair_p == a.pair_p.min()).sum()),
                                n_tied_min_library=int((b.pair_p == b.pair_p.min()).sum())))
        pd.DataFrame(summaries).to_csv(common.OUTPUT / "real_offset_summary.csv", index=False)
        pd.concat(all_pairs).to_csv(common.OUTPUT / "real_offset_pairs.csv", index=False)
        pd.DataFrame(sensitivity).to_csv(common.OUTPUT / "real_offset_rank_sensitivity.csv", index=False)
        pd.DataFrame(audits).to_csv(common.OUTPUT / "real_offset_input_audit.csv", index=False)
        common.write_manifest()
    print(pd.DataFrame(sensitivity).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
