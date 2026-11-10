#!/usr/bin/env python3
"""Prepare a pre-registered G=1000 panel and matched AP-FedAdam scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "overleaf" / "figs" / "simulation_method_comparison" / "source_data"
DETECTION_SOURCE = ROOT / "overleaf" / "figs" / "simulation_detection" / "source_data" / "simulation_detection_pair_results.csv"
PANEL_SEED = 20260806
PANEL_SIZE = 40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def summarize(frame: pd.DataFrame, scenario: str) -> dict[str, float | int | str]:
    active = frame["active"].to_numpy(dtype=bool)
    score = -np.log10(np.clip(frame["pair_p"].to_numpy(dtype=float), 1e-300, 1.0))
    selected = frame["pair_bh_q"].to_numpy(dtype=float) < 0.05
    tp = int(np.sum(selected & active))
    fp = int(np.sum(selected & ~active))
    return {
        "method": "AP-FedAdam",
        "scenario": scenario,
        "data_mode": "linear",
        "panel_genes": PANEL_SIZE,
        "tested_pairs": int(len(frame)),
        "active_pairs": int(active.sum()),
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
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(PANEL_SEED)
    panel = np.sort(rng.choice(1000, size=PANEL_SIZE, replace=False))
    panel_frame = pd.DataFrame({"Gene_ID": panel, "gene_id": [f"Gene_{gene}" for gene in panel]})
    panel_path = SOURCE_DIR / "official_g1000_gene_panel.csv"
    panel_frame.to_csv(panel_path, index=False)

    all_pairs = pd.read_csv(DETECTION_SOURCE)
    outputs = []
    summaries = []
    for scenario in ("complete", "missing"):
        frame = all_pairs[
            (all_pairs["scenario"] == scenario)
            & (all_pairs["data_mode"] == "linear")
            & (all_pairs["Gene_ID"].isin(panel))
        ].copy()
        frame = frame.rename(
            columns={
                "Gene_ID": "gene_index",
                "Cell_Type": "cell_type_index",
                "truth_active": "active",
                "p_joint_all_bases": "pair_p",
                "q_joint_all_bases": "full_5000_pair_bh_q",
            }
        )
        frame["gene_id"] = frame["gene_index"].map(lambda value: f"Gene_{int(value)}")
        frame["cell_type"] = frame["cell_type_index"].map(lambda value: f"CT{int(value)}")
        frame["pair_bh_q"] = bh_fdr(frame["pair_p"].to_numpy(dtype=float))
        frame["method"] = "AP-FedAdam"
        outputs.append(frame)
        summaries.append(summarize(frame, scenario))

    pair_path = SOURCE_DIR / "g1000_ap_fedadam_panel_pairs.csv"
    summary_path = SOURCE_DIR / "g1000_ap_fedadam_panel_summary.csv"
    pd.concat(outputs, ignore_index=True).to_csv(pair_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    manifest = {
        "panel_seed": PANEL_SEED,
        "panel_size": PANEL_SIZE,
        "selection": "uniform random sample from Gene_ID 0..999, fixed before pooled CTSV fitting",
        "data_mode": "linear",
        "ap_source": str(DETECTION_SOURCE.relative_to(ROOT)),
        "ap_source_sha256": sha256(DETECTION_SOURCE),
        "panel_path": str(panel_path.relative_to(ROOT)),
        "panel_sha256": sha256(panel_path),
    }
    (SOURCE_DIR / "g1000_method_panel_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
