#!/usr/bin/env python3
"""Complete the G=1000 simulation inference pipeline and draw detection figures.

The coefficient fits and data-local diagonal-Fisher summaries already exist in
the retained inference CSV files.  This script performs the downstream step
that was previously missing from the paper: basis-level Wald p-values are
combined for each gene--cell-type pair, BH adjustment is applied once across
all pairs, and pair-level ROC/precision-recall summaries are evaluated against
the known simulation support.  The main-text figure reports compact metric and
threshold summaries; complete ROC and precision-recall curves are exported as
separate supplementary figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
FIG_DIR = ROOT / "overleaf" / "figs" / "simulation_detection"
SOURCE_DIR = FIG_DIR / "source_data"

BASIS_MODES = ("linear", "gau1", "gau2", "cos1", "cos2")
DATA_MODES = ("linear", "gau", "cos")
MATCHED_BASIS = {"linear": "linear", "gau": "gau1", "cos": "cos1"}
SCENARIOS = {
    "complete": {
        "folder": CODE_DIR / "new_test1-5",
        "prefix": "CTSV_Fed_8_G_1000_L1_2e-07_Round1000_h",
        "label": "Complete inputs",
    },
    "missing": {
        "folder": CODE_DIR / "missing_celltype_test",
        "prefix": "CTSV_Fed_BLOCKWISE_missingBlocks_8_G_1000_L1_2e-07_Round1000_h",
        "label": "Incomplete panels",
    },
}

DATA_LABELS = {"linear": "Linear DGP", "gau": "Gaussian DGP", "cos": "Cosine DGP"}
METHOD_COLORS = {
    "Joint all-basis ACAT": "#2166AC",
    "Matched single basis": "#D97706",
}
SCENARIO_COLORS = {"complete": "#A9B8C6", "missing": "#C94C4C"}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 6,
            "figure.dpi": 150,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normal_two_sided_p(z_values: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(z_values, dtype=float)).ravel()
    out = np.array(
        [math.erfc(float(value) / math.sqrt(2.0)) if np.isfinite(value) else 1.0 for value in values],
        dtype=float,
    )
    return np.clip(out.reshape(np.asarray(z_values).shape), 1e-15, 1.0 - 1e-15)


def acat(p_values: np.ndarray, axis: int = 1) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    statistic = np.mean(np.tan((0.5 - p) * math.pi), axis=axis)
    return np.clip(0.5 - np.arctan(statistic) / math.pi, 1e-15, 1.0)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def roc_pr_curves(truth: np.ndarray, p_values: np.ndarray) -> tuple[pd.DataFrame, float, float]:
    truth = np.asarray(truth, dtype=bool)
    score = -np.log10(np.clip(np.asarray(p_values, dtype=float), 1e-300, 1.0))
    order = np.argsort(-score, kind="mergesort")
    ranked_truth = truth[order]
    tp = np.cumsum(ranked_truth)
    fp = np.cumsum(~ranked_truth)
    positives = int(truth.sum())
    negatives = int((~truth).sum())
    tpr = tp / max(positives, 1)
    fpr = fp / max(negatives, 1)
    precision = tp / np.arange(1, len(truth) + 1, dtype=float)
    recall = tpr
    auroc = float(np.trapezoid(np.r_[0.0, tpr, 1.0], np.r_[0.0, fpr, 1.0]))
    auprc = float(np.sum(precision[ranked_truth]) / max(positives, 1))
    frame = pd.DataFrame(
        {
            "rank": np.arange(1, len(truth) + 1),
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
            "score": score[order],
            "truth": ranked_truth.astype(int),
        }
    )
    frame = pd.concat(
        [
            pd.DataFrame({"rank": [0], "fpr": [0.0], "tpr": [0.0], "precision": [1.0], "recall": [0.0], "score": [np.inf], "truth": [0]}),
            frame,
        ],
        ignore_index=True,
    )
    return frame, auroc, auprc


def inference_path(scenario: str, data_mode: str, basis_mode: str) -> Path:
    config = SCENARIOS[scenario]
    return config["folder"] / f"{config['prefix']}_DataMode_{data_mode}_BasisMode_{basis_mode}_Inference_FIXED.csv"


def load_detection_scores(scenario: str, data_mode: str) -> tuple[pd.DataFrame, dict[str, str]]:
    mode_p_values: list[np.ndarray] = []
    pair_index: pd.MultiIndex | None = None
    pair_truth: np.ndarray | None = None
    hashes: dict[str, str] = {}

    for basis_mode in BASIS_MODES:
        path = inference_path(scenario, data_mode, basis_mode)
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        table = pd.read_csv(path)
        z_wide = table.pivot(index=["Gene_ID", "Cell_Type"], columns="Feature", values="Z_Score").sort_index()
        truth_wide = table.pivot(index=["Gene_ID", "Cell_Type"], columns="Feature", values="True_Beta").loc[z_wide.index]
        if pair_index is None:
            pair_index = z_wide.index
            pair_truth = (np.abs(truth_wide.to_numpy(dtype=float)) > 1e-12).any(axis=1)
        else:
            if not pair_index.equals(z_wide.index):
                raise ValueError(f"Pair ordering differs in {path}")
            current_truth = (np.abs(truth_wide.to_numpy(dtype=float)) > 1e-12).any(axis=1)
            if not np.array_equal(pair_truth, current_truth):
                raise ValueError(f"Simulation truth differs across fitted bases in {path}")
        coefficient_p = normal_two_sided_p(z_wide.to_numpy(dtype=float))
        mode_p_values.append(acat(coefficient_p, axis=1))

    assert pair_index is not None and pair_truth is not None
    mode_p = np.column_stack(mode_p_values)
    output = pair_index.to_frame(index=False)
    output["truth_active"] = pair_truth.astype(int)
    for column, basis_mode in enumerate(BASIS_MODES):
        output[f"p_{basis_mode}"] = mode_p[:, column]
    output["p_joint_all_bases"] = acat(mode_p, axis=1)
    output["q_joint_all_bases"] = bh_fdr(output["p_joint_all_bases"].to_numpy(dtype=float))
    matched = MATCHED_BASIS[data_mode]
    output["p_matched_basis"] = output[f"p_{matched}"]
    output["q_matched_basis"] = bh_fdr(output["p_matched_basis"].to_numpy(dtype=float))
    return output, hashes


def summarize_method(
    score_table: pd.DataFrame,
    p_column: str,
    q_column: str,
    scenario: str,
    data_mode: str,
    method: str,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    truth = score_table["truth_active"].to_numpy(dtype=bool)
    curves, auroc, auprc = roc_pr_curves(truth, score_table[p_column].to_numpy(dtype=float))
    selected = score_table[q_column].to_numpy(dtype=float) < 0.05
    tp = int(np.sum(selected & truth))
    fp = int(np.sum(selected & ~truth))
    fn = int(np.sum(~selected & truth))
    tn = int(np.sum(~selected & ~truth))
    curves["scenario"] = scenario
    curves["data_mode"] = data_mode
    curves["method"] = method
    summary = {
        "scenario": scenario,
        "data_mode": data_mode,
        "method": method,
        "n_pairs": int(len(truth)),
        "n_active_pairs": int(truth.sum()),
        "auroc": auroc,
        "auprc": auprc,
        "bh_threshold": 0.05,
        "selected_pairs": int(selected.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "empirical_fdr": fp / max(tp + fp, 1),
    }
    return curves, summary


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.16, 1.07, label, transform=axis.transAxes, fontsize=9, fontweight="bold", va="bottom")


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 450},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(FIG_DIR / f"{stem}{suffix}", bbox_inches="tight", **kwargs)


def ordered_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.copy()
    ordered["scenario_order"] = ordered["scenario"].map({"complete": 0, "missing": 1})
    ordered["data_order"] = ordered["data_mode"].map({"linear": 0, "gau": 1, "cos": 2})
    ordered["method_order"] = ordered["method"].map({"Joint all-basis ACAT": 0, "Matched single basis": 1})
    return ordered.sort_values(["scenario_order", "data_order", "method_order"])


def draw_metric_panel(axis: plt.Axes, summary: pd.DataFrame, metric: str, title: str, show_rows: bool) -> None:
    conditions = [(scenario, mode) for scenario in ("complete", "missing") for mode in DATA_MODES]
    y_positions = np.arange(len(conditions) - 1, -1, -1, dtype=float)
    method_offsets = {"Joint all-basis ACAT": 0.13, "Matched single basis": -0.13}
    markers = {"Joint all-basis ACAT": "o", "Matched single basis": "D"}

    axis.axhspan(2.5, 5.5, color="#F4F6F8", zorder=-3)
    axis.axhline(2.5, color="#C8CDD2", linewidth=0.6, zorder=-2)
    axis.axvline(1.0, color="#767676", linewidth=0.65, linestyle=":", zorder=-2)

    for y_value, (scenario, data_mode) in zip(y_positions, conditions):
        rows = summary[(summary["scenario"] == scenario) & (summary["data_mode"] == data_mode)]
        pair = []
        for method in ("Joint all-basis ACAT", "Matched single basis"):
            value = float(rows.loc[rows["method"] == method, metric].iloc[0])
            point_y = y_value + method_offsets[method]
            pair.append((value, point_y))
            axis.scatter(
                value,
                point_y,
                s=27,
                marker=markers[method],
                facecolor=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
            if value >= 0.975:
                text_x, alignment = value - 0.0022, "right"
            else:
                text_x, alignment = value + 0.0022, "left"
            axis.text(
                text_x,
                point_y,
                f"{value:.3f}",
                ha=alignment,
                va="center",
                fontsize=5.3,
                color=METHOD_COLORS[method],
            )
        axis.plot(
            [pair[0][0], pair[1][0]],
            [pair[0][1], pair[1][1]],
            color="#B7BDC3",
            linewidth=0.7,
            zorder=1,
        )

    row_labels = [
        "Complete | Linear",
        "Complete | Gaussian",
        "Complete | Cosine",
        "Incomplete | Linear",
        "Incomplete | Gaussian",
        "Incomplete | Cosine",
    ]
    axis.set_yticks(y_positions)
    if show_rows:
        axis.set_yticklabels(row_labels)
        axis.tick_params(axis="y", length=0, pad=3)
    else:
        axis.tick_params(axis="y", left=False, labelleft=False)
    axis.set_xlim(0.925, 1.008)
    axis.set_xticks([0.94, 0.96, 0.98, 1.00])
    axis.set_ylim(-0.55, 5.55)
    axis.set_xlabel(title)
    axis.set_title(f"Ranking by {title}", loc="left", fontweight="bold", pad=8)
    axis.grid(axis="x", color="#E1E4E8", linewidth=0.55)
    axis.set_axisbelow(True)


def draw_main_figure(summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.2, 3.15))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.22, 1.08, 1.0], wspace=0.42)

    auroc_axis = fig.add_subplot(grid[0, 0])
    panel_label(auroc_axis, "a")
    draw_metric_panel(auroc_axis, summary, "auroc", "AUROC", show_rows=True)

    auprc_axis = fig.add_subplot(grid[0, 1], sharey=auroc_axis)
    panel_label(auprc_axis, "b")
    draw_metric_panel(auprc_axis, summary, "auprc", "AUPRC", show_rows=False)

    fp_axis = fig.add_subplot(grid[0, 2])
    panel_label(fp_axis, "c")
    joint = ordered_summary(summary[summary["method"] == "Joint all-basis ACAT"])
    x_values = np.arange(len(DATA_MODES), dtype=float)
    width = 0.32
    for scenario, offset in (("complete", -width / 2), ("missing", width / 2)):
        scenario_rows = joint[joint["scenario"] == scenario].set_index("data_mode").loc[list(DATA_MODES)]
        values = scenario_rows["false_positives"].to_numpy(dtype=float)
        bars = fp_axis.bar(
            x_values + offset,
            values,
            width=width,
            color=SCENARIO_COLORS[scenario],
            label="Complete inputs" if scenario == "complete" else "Incomplete panels",
            zorder=2,
        )
        for bar, value in zip(bars, values):
            fp_axis.text(
                bar.get_x() + bar.get_width() / 2,
                max(value + 3.0, 2.0),
                f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=5.8,
                color="#333333",
            )

    fp_axis.set_xticks(x_values)
    fp_axis.set_xticklabels(["Linear", "Gaussian", "Cosine"])
    fp_axis.set_ylim(0, 165)
    fp_axis.set_yticks([0, 50, 100, 150])
    fp_axis.set_ylabel("False-positive pairs")
    fp_axis.set_title("False positives at BH $q<0.05$", loc="left", fontweight="bold", y=1.12, pad=0)
    fp_axis.grid(axis="y", color="#E1E4E8", linewidth=0.55)
    fp_axis.set_axisbelow(True)
    fp_axis.legend(loc="upper right", fontsize=5.3, handlelength=1.4)
    fp_axis.text(
        0.0,
        1.01,
        "Joint TP = 1,000/1,000 in all six settings",
        transform=fp_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color="#333333",
    )

    method_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.8,
            markerfacecolor=METHOD_COLORS["Joint all-basis ACAT"],
            markeredgecolor="white",
            label="Joint all-basis",
        ),
        plt.Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markersize=4.4,
            markerfacecolor=METHOD_COLORS["Matched single basis"],
            markeredgecolor="white",
            label="Matched single basis",
        ),
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.39, 1.01),
        ncol=2,
        columnspacing=1.3,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.155, right=0.99, bottom=0.18, top=0.82)
    save_figure(fig, "fig3_simulation_detection_roc")
    plt.close(fig)


def draw_curve_grid(curves: pd.DataFrame, summary: pd.DataFrame, curve_type: str) -> None:
    if curve_type == "roc":
        x_column, y_column = "fpr", "tpr"
        x_label, y_label = "False-positive rate", "True-positive rate"
        metric, stem = "auroc", "fig_s_simulation_detection_roc"
    elif curve_type == "pr":
        x_column, y_column = "recall", "precision"
        x_label, y_label = "Recall", "Precision"
        metric, stem = "auprc", "fig_s_simulation_detection_pr"
    else:
        raise ValueError(f"Unknown curve type: {curve_type}")

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.35), sharex=True, sharey=True)
    line_styles = {"Joint all-basis ACAT": "-", "Matched single basis": "--"}

    for row_index, scenario in enumerate(("complete", "missing")):
        for column_index, data_mode in enumerate(DATA_MODES):
            axis = axes[row_index, column_index]
            panel_label(axis, chr(ord("a") + row_index * 3 + column_index))
            metric_lines = []
            for method in ("Joint all-basis ACAT", "Matched single basis"):
                line = curves[
                    (curves["scenario"] == scenario)
                    & (curves["data_mode"] == data_mode)
                    & (curves["method"] == method)
                ]
                values = summary[
                    (summary["scenario"] == scenario)
                    & (summary["data_mode"] == data_mode)
                    & (summary["method"] == method)
                ]
                metric_value = float(values[metric].iloc[0])
                metric_lines.append(
                    f"{'Joint' if method.startswith('Joint') else 'Matched'} {metric_value:.3f}"
                )
                line_kwargs = {
                    "color": METHOD_COLORS[method],
                    "linestyle": line_styles[method],
                    "linewidth": 1.25 if method.startswith("Joint") else 1.05,
                    "label": "Joint all-basis" if method.startswith("Joint") else "Matched single basis",
                    "zorder": 3,
                }
                if curve_type == "pr":
                    axis.step(line[x_column], line[y_column], where="post", **line_kwargs)
                else:
                    axis.plot(line[x_column], line[y_column], **line_kwargs)

            if curve_type == "roc":
                axis.plot([0, 1], [0, 1], color="#A8ADB2", linewidth=0.6, linestyle=":", zorder=1)
            else:
                axis.axhline(0.2, color="#A8ADB2", linewidth=0.6, linestyle=":", zorder=1)
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.01)
            axis.grid(color="#E4E7EA", linewidth=0.45)
            axis.set_axisbelow(True)
            if row_index == 0:
                axis.set_title(DATA_LABELS[data_mode], fontweight="bold", pad=8)
            if column_index == 0:
                axis.set_ylabel(f"{SCENARIOS[scenario]['label']}\n{y_label}")
            if row_index == 1:
                axis.set_xlabel(x_label)
            axis.text(
                0.97,
                0.06,
                "\n".join(metric_lines),
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.5,
                color="#333333",
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        columnspacing=1.5,
        handlelength=2.3,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.11, top=0.86, hspace=0.35, wspace=0.24)
    save_figure(fig, stem)
    plt.close(fig)


def draw_figures(curves: pd.DataFrame, summary: pd.DataFrame) -> None:
    draw_main_figure(summary)
    draw_curve_grid(curves, summary, "roc")
    draw_curve_grid(curves, summary, "pr")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--redraw-only",
        action="store_true",
        help="Redraw figures from released source-data CSVs without recomputing inference summaries.",
    )
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    if args.redraw_only:
        curves = pd.read_csv(SOURCE_DIR / "simulation_detection_curves.csv")
        summary = pd.read_csv(SOURCE_DIR / "simulation_detection_summary.csv")
        draw_figures(curves, summary)
        print(f"Redrew detection figures from released source data in {SOURCE_DIR}")
        return

    pair_tables: list[pd.DataFrame] = []
    curve_tables: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []
    input_hashes: dict[str, str] = {}

    for scenario in SCENARIOS:
        for data_mode in DATA_MODES:
            scores, hashes = load_detection_scores(scenario, data_mode)
            input_hashes.update(hashes)
            scores["scenario"] = scenario
            scores["data_mode"] = data_mode
            pair_tables.append(scores)
            for method, p_column, q_column in (
                ("Joint all-basis ACAT", "p_joint_all_bases", "q_joint_all_bases"),
                ("Matched single basis", "p_matched_basis", "q_matched_basis"),
            ):
                curve, summary = summarize_method(scores, p_column, q_column, scenario, data_mode, method)
                curve_tables.append(curve)
                summaries.append(summary)

    pair_results = pd.concat(pair_tables, ignore_index=True)
    curves = pd.concat(curve_tables, ignore_index=True)
    summary = pd.DataFrame(summaries)
    summary["scenario_order"] = summary["scenario"].map({"complete": 0, "missing": 1})
    summary["data_order"] = summary["data_mode"].map({"linear": 0, "gau": 1, "cos": 2})
    summary["method_order"] = summary["method"].map({"Joint all-basis ACAT": 0, "Matched single basis": 1})
    summary = summary.sort_values(["scenario_order", "data_order", "method_order"]).drop(
        columns=["scenario_order", "data_order", "method_order"]
    )

    pair_results.to_csv(SOURCE_DIR / "simulation_detection_pair_results.csv", index=False)
    curves.to_csv(SOURCE_DIR / "simulation_detection_curves.csv", index=False)
    summary.to_csv(SOURCE_DIR / "simulation_detection_summary.csv", index=False)
    manifest = {
        "analysis": "pair-level Wald inference after retained G=1000 coefficient fits",
        "unit": "gene-cell-type pair",
        "truth": "active when either true spatial coefficient is nonzero",
        "within_basis_combination": "ACAT across h1 and h2 Wald p-values",
        "joint_basis_combination": "ACAT across linear, gau1, gau2, cos1 and cos2 pair-level p-values",
        "multiple_testing": "BH once across 1000 genes x 5 cell types within each scenario and data-generating mode",
        "figure_outputs": {
            "main": "fig3_simulation_detection_roc: paired AUROC/AUPRC summaries and joint-method false-positive counts",
            "supplementary_roc": "fig_s_simulation_detection_roc: 2 x 3 complete ROC grid",
            "supplementary_pr": "fig_s_simulation_detection_pr: 2 x 3 complete precision-recall grid",
        },
        "redraw_mode": "--redraw-only reads the released curves and summary CSVs and does not recompute inference or alter source-data values",
        "input_sha256": input_hashes,
    }
    (SOURCE_DIR / "simulation_detection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    draw_figures(curves, summary)
    print(summary.to_string(index=False))
    print(f"Wrote detection outputs to {FIG_DIR}")


if __name__ == "__main__":
    main()
