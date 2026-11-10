#!/usr/bin/env python3
"""Export a matched compact benchmark and AP-FedAdam pair-level inference.

The exported count, coordinate, and omega matrices are consumed unchanged by
the official R implementations.  AP-FedAdam is refit on the same client draws;
diagonal observed information is then used for two-sided Wald tests.  Tests are
combined first over the two coordinates and then over the same five spatial
basis families used by pooled CTSV, followed by one BH correction over all
gene--cell-type pairs.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import run_compact_benchmark as compact


ROOT = Path(__file__).resolve().parents[3] / "04_数值实验"
OUT_DIR = ROOT / "overleaf" / "figs" / "simulation_method_comparison" / "source_data"
REPEATS = 3
CONDITIONS = ("complete", "missing")
OFFICIAL_PANEL_SIZE = 40
OFFICIAL_PANEL_SEED = 20260806
BASIS_MODES = ("linear", "gau1", "gau2", "cos1", "cos2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normal_two_sided_p(z_values: np.ndarray) -> np.ndarray:
    flat = np.abs(np.asarray(z_values, dtype=float)).ravel()
    values = np.array([math.erfc(float(z) / math.sqrt(2.0)) for z in flat])
    return np.clip(values.reshape(np.asarray(z_values).shape), 1e-15, 1.0 - 1e-15)


def acat(p_values: np.ndarray, axis: int = 0) -> np.ndarray:
    p = np.clip(np.asarray(p_values, dtype=float), 1e-15, 1.0 - 1e-15)
    statistic = np.mean(np.tan((0.5 - p) * math.pi), axis=axis)
    return np.clip(0.5 - np.arctan(statistic) / math.pi, 1e-15, 1.0)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def raw_condition(base: list[dict[str, np.ndarray]], condition: str) -> list[dict[str, np.ndarray]]:
    clients = []
    for client_index, client in enumerate(base):
        omega = client["om"].copy()
        if condition == "missing" and client_index == compact.TARGET:
            omega[:, compact.MISSING_TYPES] = 0.0
            omega /= omega.sum(axis=1, keepdims=True).clip(1e-8)
        clients.append(
            {
                "y": client["y"].copy(),
                "coords": client["coords"].copy(),
                "omega": omega,
            }
        )
    return clients


def export_inputs(clients: list[dict[str, np.ndarray]], truth_beta: np.ndarray, prefix: str) -> dict[str, Path]:
    spot_ids = [
        f"client{client_index}_spot{spot_index}"
        for client_index, client in enumerate(clients)
        for spot_index in range(client["y"].shape[0])
    ]
    counts = np.concatenate([client["y"] for client in clients], axis=0).T.astype(int)
    coords = np.concatenate([client["coords"] for client in clients], axis=0)
    omega = np.concatenate([client["omega"] for client in clients], axis=0)

    count_path = OUT_DIR / f"{prefix}_counts.csv"
    coord_path = OUT_DIR / f"{prefix}_coordinates.csv"
    omega_path = OUT_DIR / f"{prefix}_omega.csv"
    truth_path = OUT_DIR / f"{prefix}_truth.csv"

    count_frame = pd.DataFrame(counts, columns=spot_ids)
    count_frame.insert(0, "gene_id", [f"gene{gene}" for gene in range(compact.G)])
    count_frame.to_csv(count_path, index=False)

    coord_frame = pd.DataFrame(coords, columns=["x", "y"])
    coord_frame.insert(0, "spot_id", spot_ids)
    coord_frame.insert(1, "client", np.repeat(np.arange(compact.CLIENTS), compact.SPOTS))
    coord_frame.to_csv(coord_path, index=False)

    omega_frame = pd.DataFrame(omega, columns=[f"CT{k}" for k in range(compact.K)])
    omega_frame.insert(0, "spot_id", spot_ids)
    omega_frame.to_csv(omega_path, index=False)

    truth_rows = []
    for gene in range(compact.G):
        for cell_type in range(compact.K):
            truth_rows.append(
                {
                    "gene_id": f"gene{gene}",
                    "cell_type": f"CT{cell_type}",
                    "true_beta_h1": float(truth_beta[0, cell_type, gene]),
                    "true_beta_h2": float(truth_beta[1, cell_type, gene]),
                    "active": int(np.any(np.abs(truth_beta[:, cell_type, gene]) > 1e-12)),
                }
            )
    pd.DataFrame(truth_rows).to_csv(truth_path, index=False)
    return {"counts": count_path, "coordinates": coord_path, "omega": omega_path, "truth": truth_path}


def model_clients_for_basis(
    raw_clients: list[dict[str, np.ndarray]], basis_mode: str
) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    all_coordinates = np.concatenate([client["coords"] for client in raw_clients], axis=0)
    coordinate_mean = all_coordinates.mean(axis=0)
    coordinate_sd = all_coordinates.std(axis=0, ddof=1).clip(1e-8)
    standardized = (all_coordinates - coordinate_mean) / coordinate_sd
    psi = np.quantile(np.abs(standardized), [0.4, 0.6], axis=0)

    if basis_mode == "linear":
        features = standardized
    elif basis_mode == "gau1":
        features = np.exp(-(standardized**2) / (2.0 * psi[0][None, :] ** 2))
    elif basis_mode == "gau2":
        features = np.exp(-(standardized**2) / (2.0 * psi[1][None, :] ** 2))
    elif basis_mode == "cos1":
        features = np.cos(2.0 * math.pi * standardized / psi[0][None, :])
    elif basis_mode == "cos2":
        features = np.cos(2.0 * math.pi * standardized / psi[1][None, :])
    else:
        raise ValueError(f"Unknown basis mode: {basis_mode}")

    active_blocks = np.ones((compact.CLIENTS, compact.K), dtype=float)
    clients = []
    offset = 0
    for client_index, client in enumerate(raw_clients):
        n_spots = client["y"].shape[0]
        clients.append(
            {
                "y": client["y"],
                "h": features[offset : offset + n_spots],
                "om": client["omega"],
                "n": n_spots,
            }
        )
        offset += n_spots
        if np.all(client["omega"][:, compact.MISSING_TYPES] == 0.0):
            active_blocks[client_index, compact.MISSING_TYPES] = 0.0
    return clients, active_blocks


def ap_family_inference(
    fitted: dict[str, np.ndarray], model_clients: list[dict[str, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    fisher = np.zeros_like(fitted["beta"], dtype=float)
    for client in model_clients:
        spatial = np.einsum("nl,lkg->nkg", client["h"], fitted["beta"])
        log_mean = np.sum(client["om"][:, :, None] * (fitted["theta"][0][None] + spatial), axis=1)
        mean = np.exp(np.clip(log_mean, -7, 6))
        score_residual = compact.PSI * (client["y"] - mean) / (compact.PSI + mean)
        design = client["h"][:, :, None] * client["om"][:, None, :]
        fisher += np.einsum("ng,nlk->lkg", score_residual**2, design**2)

    standard_error = 1.0 / np.sqrt(np.maximum(fisher, 1e-12))
    axis_p = normal_two_sided_p(fitted["beta"] / standard_error)
    return acat(axis_p, axis=0), np.linalg.norm(fitted["beta"], axis=0)


def ap_joint_inference(
    family_p_values: dict[str, np.ndarray],
    family_beta_norms: dict[str, np.ndarray],
    truth_beta: np.ndarray,
) -> pd.DataFrame:
    joint_p = acat(np.stack([family_p_values[mode] for mode in BASIS_MODES], axis=0), axis=0)
    joint_q = bh_fdr(joint_p.ravel()).reshape(joint_p.shape)
    maximum_beta_norm = np.max(np.stack([family_beta_norms[mode] for mode in BASIS_MODES], axis=0), axis=0)
    rows = []
    for cell_type in range(compact.K):
        for gene in range(compact.G):
            row = {
                "gene_id": f"gene{gene}",
                "cell_type": f"CT{cell_type}",
                "active": int(np.any(np.abs(truth_beta[:, cell_type, gene]) > 1e-12)),
                "pair_p": float(joint_p[cell_type, gene]),
                "pair_bh_q": float(joint_q[cell_type, gene]),
                "maximum_beta_norm": float(maximum_beta_norm[cell_type, gene]),
            }
            for basis_mode in BASIS_MODES:
                row[f"pair_p_{basis_mode}"] = float(family_p_values[basis_mode][cell_type, gene])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_pairs(frame: pd.DataFrame, method: str, repeat: int, condition: str) -> dict[str, float | int | str]:
    active = frame["active"].to_numpy(dtype=bool)
    score = -np.log10(np.clip(frame["pair_p"].to_numpy(dtype=float), 1e-300, 1.0))
    selected = frame["pair_bh_q"].to_numpy(dtype=float) < 0.05
    tp = int(np.sum(selected & active))
    fp = int(np.sum(selected & ~active))
    return {
        "method": method,
        "repeat": repeat,
        "condition": condition,
        "auroc": float(roc_auc_score(active, score)),
        "auprc": float(average_precision_score(active, score)),
        "selected_pairs": int(selected.sum()),
        "selected_genes": int(frame.loc[selected, "gene_id"].nunique()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": int(np.sum(~selected & active)),
        "empirical_fdr": float(fp / max(tp + fp, 1)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_rng = np.random.default_rng(OFFICIAL_PANEL_SEED)
    official_gene_indices = np.sort(panel_rng.choice(compact.G, size=OFFICIAL_PANEL_SIZE, replace=False))
    official_gene_ids = {f"gene{gene}" for gene in official_gene_indices}
    pd.DataFrame(
        {"gene_id": [f"gene{gene}" for gene in official_gene_indices]}
    ).to_csv(OUT_DIR / "official_method_gene_panel.csv", index=False)
    summaries = []
    manifest: dict[str, object] = {
        "benchmark": "matched_compact_nb_ctsv",
        "seed": compact.SEED,
        "repeats": REPEATS,
        "clients": compact.CLIENTS,
        "spots_per_client": compact.SPOTS,
        "genes": compact.G,
        "cell_types": compact.K,
        "conditions": list(CONDITIONS),
        "official_method_panel_size": OFFICIAL_PANEL_SIZE,
        "official_method_panel_seed": OFFICIAL_PANEL_SEED,
        "official_method_panel_selection": "uniform random sample of genes fixed before method fitting",
        "dispersion": compact.PSI,
        "ap_fedadam_rounds": compact.ROUNDS,
        "ap_fedadam_local_steps": compact.E,
        "testing_unit": "gene-cell-type pair",
        "ap_basis_modes": list(BASIS_MODES),
        "pair_test": "coordinate p-values combined within each basis family, then across five families by ACAT; BH once over G*K pairs",
        "inputs": {},
    }

    for repeat in range(REPEATS):
        base, truth_beta = compact.gen(np.random.default_rng(compact.SEED + repeat))
        for condition in CONDITIONS:
            prefix = f"repeat{repeat}_{condition}"
            raw_clients = raw_condition(base, condition)
            paths = export_inputs(raw_clients, truth_beta, prefix)
            family_p_values: dict[str, np.ndarray] = {}
            family_beta_norms: dict[str, np.ndarray] = {}
            family_diagnostics: dict[str, dict[str, float]] = {}
            for basis_mode in BASIS_MODES:
                model_clients, active_blocks = model_clients_for_basis(raw_clients, basis_mode)
                fitted, _, diagnostics = compact.run(
                    model_clients,
                    active_blocks,
                    compact.METHODS["AP-FedAdam"],
                    lr=compact.LR,
                    record_loss=False,
                )
                family_p_values[basis_mode], family_beta_norms[basis_mode] = ap_family_inference(
                    fitted, model_clients
                )
                family_diagnostics[basis_mode] = diagnostics
            pair_frame = ap_joint_inference(family_p_values, family_beta_norms, truth_beta)
            pair_frame.insert(0, "condition", condition)
            pair_frame.insert(0, "repeat", repeat)
            pair_frame["official_method_panel"] = pair_frame["gene_id"].isin(official_gene_ids).astype(int)
            pair_path = OUT_DIR / f"{prefix}_ap_fedadam_pairs.csv"
            pair_frame.to_csv(pair_path, index=False)
            summary = summarize_pairs(
                pair_frame[pair_frame["official_method_panel"] == 1].copy(),
                "AP-FedAdam",
                repeat,
                condition,
            )
            summary["evaluation_panel"] = "fixed_random_40_gene_panel"
            summary["mean_final_loss_across_bases"] = float(
                np.mean([family_diagnostics[mode]["final_loss"] for mode in BASIS_MODES])
            )
            summary["maximum_final_update_norm"] = float(
                np.max([family_diagnostics[mode]["final_update_norm"] for mode in BASIS_MODES])
            )
            summaries.append(summary)
            manifest["inputs"][prefix] = {
                key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for key, path in paths.items()
            }
            print(prefix, summary)

    pd.DataFrame(summaries).to_csv(OUT_DIR / "ap_fedadam_detection_summary.csv", index=False)
    (OUT_DIR / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
