#!/usr/bin/env python3
"""Summarise matched AP-FedAdam and pooled CTSV tests on the fixed G=1000 panel.

The panel was sampled before fitting.  This script does not refit either method;
it reconstructs pair-level tests from the archived p-values, applies one BH
adjustment over the same 200 gene--cell-type pairs, and records calibration
failures separately from matched-model performance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = (
    ROOT
    / "05_论文草稿"
    / "latex"
    / "figs"
    / "simulation_method_comparison"
    / "source_data"
)


def acat(values: np.ndarray, axis: int = -1) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-15, 1 - 1e-15)
    statistic = np.mean(np.tan((0.5 - clipped) * np.pi), axis=axis)
    return np.clip(0.5 - np.arctan(statistic) / np.pi, 1e-15, 1.0)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def summarise(frame: pd.DataFrame, method: str, test: str, scenario: str) -> dict:
    active = frame["active"].astype(bool).to_numpy()
    p_values = frame["pair_p"].to_numpy(dtype=float)
    q_values = bh_adjust(p_values)
    selected = q_values < 0.05
    tp = int(np.sum(selected & active))
    fp = int(np.sum(selected & ~active))
    return {
        "method": method,
        "test": test,
        "scenario": scenario,
        "panel_genes": int(frame["gene_id"].nunique()),
        "tested_pairs": int(len(frame)),
        "active_pairs": int(np.sum(active)),
        "auroc": float(roc_auc_score(active, -np.log10(np.maximum(p_values, 1e-300)))),
        "auprc": float(average_precision_score(active, -np.log10(np.maximum(p_values, 1e-300)))),
        "selected_pairs": int(np.sum(selected)),
        "selected_genes": int(frame.loc[selected, "gene_id"].nunique()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": int(np.sum(~selected & active)),
        "empirical_fdr": float(fp / max(tp + fp, 1)),
    }


def ap_rows() -> tuple[list[dict], list[pd.DataFrame]]:
    source = pd.read_csv(SOURCE_DIR / "g1000_ap_fedadam_panel_pairs.csv")
    summaries: list[dict] = []
    pair_frames: list[pd.DataFrame] = []
    for scenario, subset in source.groupby("scenario", sort=False):
        for test, column in (
            ("joint five-basis", "pair_p"),
            ("matched linear", "p_linear"),
        ):
            frame = subset[["gene_id", "cell_type", "active", column]].copy()
            frame = frame.rename(columns={column: "pair_p"})
            frame["method"] = "AP-FedAdam"
            frame["test"] = test
            frame["scenario"] = scenario
            frame["pair_bh_q"] = bh_adjust(frame["pair_p"].to_numpy())
            summaries.append(summarise(frame, "AP-FedAdam", test, scenario))
            pair_frames.append(frame)
    return summaries, pair_frames


def ctsv_linear_rows() -> tuple[dict, pd.DataFrame]:
    axis_frame = pd.read_csv(
        SOURCE_DIR / "g1000_complete_linear_ctsv_linear_axis_pval.csv"
    )
    ap_truth = pd.read_csv(SOURCE_DIR / "g1000_ap_fedadam_panel_pairs.csv")
    truth = ap_truth.loc[
        ap_truth["scenario"].eq("complete"),
        ["gene_id", "cell_type", "active"],
    ].copy()
    cell_types = [f"CT{index}" for index in range(5)]
    rows = []
    for cell_index, cell_type in enumerate(cell_types):
        axis_values = axis_frame.iloc[:, [1 + cell_index, 1 + 5 + cell_index]].to_numpy()
        axis_values[~np.isfinite(axis_values) | (axis_values < 0)] = 1.0
        pair_p = acat(axis_values, axis=1)
        rows.append(
            pd.DataFrame(
                {
                    "gene_id": axis_frame["gene_id"],
                    "cell_type": cell_type,
                    "pair_p": pair_p,
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True).merge(
        truth, on=["gene_id", "cell_type"], how="left", validate="one_to_one"
    )
    frame["method"] = "CTSV (pooled)"
    frame["test"] = "matched linear"
    frame["scenario"] = "complete"
    frame["pair_bh_q"] = bh_adjust(frame["pair_p"].to_numpy())
    return summarise(frame, "CTSV (pooled)", "matched linear", "complete"), frame


def ctsv_joint_rows() -> tuple[dict, pd.DataFrame]:
    frame = pd.read_csv(SOURCE_DIR / "g1000_complete_linear_ctsv_pairs.csv")
    frame["test"] = "joint five-basis"
    summary = summarise(frame, "CTSV (pooled)", "joint five-basis", "complete")
    return summary, frame


def main() -> None:
    summaries, pair_frames = ap_rows()
    ctsv_linear_summary, ctsv_linear_pairs = ctsv_linear_rows()
    ctsv_joint_summary, ctsv_joint_pairs = ctsv_joint_rows()
    summaries.extend([ctsv_linear_summary, ctsv_joint_summary])
    pair_frames.extend([ctsv_linear_pairs, ctsv_joint_pairs])

    summary_frame = pd.DataFrame(summaries)
    summary_frame["interpretation"] = "matched comparison"
    joint_ctsv = summary_frame["method"].eq("CTSV (pooled)") & summary_frame[
        "test"
    ].eq("joint five-basis")
    summary_frame.loc[joint_ctsv, "interpretation"] = (
        "calibration failure; not used as a performance comparator"
    )
    summary_frame.to_csv(
        SOURCE_DIR / "g1000_method_comparison_audited_summary.csv", index=False
    )
    pd.concat(pair_frames, ignore_index=True, sort=False).to_csv(
        SOURCE_DIR / "g1000_method_comparison_audited_pairs.csv", index=False
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
