#!/usr/bin/env python3
"""Aggregate and plot the audited compact cross-method simulation pilot."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "05_论文草稿/latex/figs/simulation_method_comparison"
SOURCE_DIR = OUT_DIR / "source_data"
TABLE_DIR = ROOT / "05_论文草稿/latex/tables"

METHODS = [
    ("AP-FedAdam (joint)", "ap_fedadam", "#0072B2"),
    ("CTSV", "ctsv", "#555555"),
    ("STANCE", "stance", "#009E73"),
    ("Celina", "celina", "#CC79A7"),
    ("spVC", "spvc", "#E69F00"),
]


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def load_pairs(method_key: str, repeat: int) -> pd.DataFrame:
    path = SOURCE_DIR / f"repeat{repeat}_complete_{method_key}_pairs.csv"
    frame = pd.read_csv(path)
    if method_key == "ap_fedadam":
        frame = frame.loc[frame["official_method_panel"].eq(1)].copy()
    frame = frame[["gene_id", "cell_type", "active", "pair_p"]].copy()
    frame["pair_p"] = pd.to_numeric(frame["pair_p"], errors="coerce")
    frame["active"] = pd.to_numeric(frame["active"], errors="raise").astype(int)
    if len(frame) != 200:
        raise ValueError(f"Expected 200 pairs in {path}, found {len(frame)}")
    if frame["pair_p"].isna().any():
        raise ValueError(f"Missing p-values in {path}")
    frame["pair_p"] = frame["pair_p"].clip(1e-300, 1.0)
    frame["pair_bh_q_common"] = bh_adjust(frame["pair_p"].to_numpy())
    return frame.sort_values(["gene_id", "cell_type"]).reset_index(drop=True)


def fmt_mean_sd(mean: float, sd: float, digits: int) -> str:
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def write_latex_table(summary: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & AUROC & AUPRC & Selected pairs & TP & FP & Selected genes \\",
        r"\midrule",
    ]
    labels = {
        "AP-FedAdam (joint)": "AP-FedAdam (joint five-basis)",
        "CTSV": "CTSV (pooled)",
        "STANCE": "STANCE (pooled)",
        "Celina": "Celina (pooled)",
        "spVC": "spVC (pooled)",
    }
    for _, row in summary.iterrows():
        rows.append(
            " & ".join(
                [
                    labels[row["method"]],
                    fmt_mean_sd(row["auroc_mean"], row["auroc_sd"], 3),
                    fmt_mean_sd(row["auprc_mean"], row["auprc_sd"], 3),
                    fmt_mean_sd(row["selected_pairs_mean"], row["selected_pairs_sd"], 1),
                    fmt_mean_sd(row["tp_mean"], row["tp_sd"], 1),
                    fmt_mean_sd(row["fp_mean"], row["fp_sd"], 1),
                    fmt_mean_sd(row["selected_genes_mean"], row["selected_genes_sd"], 1),
                ]
            )
            + r" \\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLE_DIR / "table_compact_method_comparison.tex").write_text(
        "\n".join(rows) + "\n", encoding="ascii"
    )


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs: list[pd.DataFrame] = []
    per_repeat: list[dict[str, float | int | str]] = []
    roc_records: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        name: [] for name, _, _ in METHODS
    }
    truth_reference: dict[int, pd.DataFrame] = {}

    for method_name, method_key, _ in METHODS:
        for repeat in range(3):
            frame = load_pairs(method_key, repeat)
            truth = frame[["gene_id", "cell_type", "active"]]
            if repeat not in truth_reference:
                truth_reference[repeat] = truth
            elif not truth.equals(truth_reference[repeat]):
                raise ValueError(f"Truth/panel mismatch for {method_name}, repeat {repeat}")

            selected = frame["pair_bh_q_common"].lt(0.05)
            active = frame["active"].eq(1)
            score = -np.log10(frame["pair_p"].to_numpy())
            fpr, tpr, _ = roc_curve(active.astype(int), score)
            roc_records[method_name].append((fpr, tpr))

            tp = int((selected & active).sum())
            fp = int((selected & ~active).sum())
            fn = int((~selected & active).sum())
            selected_genes = int(frame.loc[selected, "gene_id"].nunique())
            per_repeat.append(
                {
                    "method": method_name,
                    "repeat": repeat,
                    "n_pairs": len(frame),
                    "n_active": int(active.sum()),
                    "auroc": roc_auc_score(active, score),
                    "auprc": average_precision_score(active, score),
                    "selected_pairs": int(selected.sum()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tpr_at_bh05": tp / max(int(active.sum()), 1),
                    "empirical_fdp": fp / max(tp + fp, 1),
                    "selected_genes": selected_genes,
                }
            )
            frame.insert(0, "repeat", repeat)
            frame.insert(0, "method", method_name)
            all_pairs.append(frame)

    pair_table = pd.concat(all_pairs, ignore_index=True)
    repeat_table = pd.DataFrame(per_repeat)
    summary = (
        repeat_table.groupby("method", sort=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            selected_pairs_mean=("selected_pairs", "mean"),
            selected_pairs_sd=("selected_pairs", "std"),
            tp_mean=("tp", "mean"),
            tp_sd=("tp", "std"),
            fp_mean=("fp", "mean"),
            fp_sd=("fp", "std"),
            fn_mean=("fn", "mean"),
            fn_sd=("fn", "std"),
            empirical_fdp_mean=("empirical_fdp", "mean"),
            empirical_fdp_sd=("empirical_fdp", "std"),
            selected_genes_mean=("selected_genes", "mean"),
            selected_genes_sd=("selected_genes", "std"),
        )
        .reset_index()
    )

    pair_table.to_csv(SOURCE_DIR / "compact_method_comparison_pairs.csv", index=False)
    repeat_table.to_csv(SOURCE_DIR / "compact_method_comparison_per_repeat.csv", index=False)
    summary.to_csv(SOURCE_DIR / "compact_method_comparison_summary.csv", index=False)
    write_latex_table(summary)

    configure_style()
    fig = plt.figure(figsize=(7.05, 2.65), constrained_layout=False)
    grid = fig.add_gridspec(
        1, 3, width_ratios=[1.35, 1.0, 1.05], left=0.07, right=0.99,
        bottom=0.25, top=0.86, wspace=0.43
    )
    ax_roc = fig.add_subplot(grid[0, 0])
    ax_metric = fig.add_subplot(grid[0, 1])
    ax_count = fig.add_subplot(grid[0, 2])

    common_fpr = np.linspace(0, 1, 501)
    for method_name, _, color in METHODS:
        curves = []
        for fpr, tpr in roc_records[method_name]:
            curves.append(np.interp(common_fpr, fpr, tpr))
        curves_arr = np.vstack(curves)
        mean_tpr = curves_arr.mean(axis=0)
        sd_tpr = curves_arr.std(axis=0, ddof=1)
        ax_roc.plot(common_fpr, mean_tpr, color=color, lw=1.5, label=method_name)
        ax_roc.fill_between(
            common_fpr,
            np.clip(mean_tpr - sd_tpr, 0, 1),
            np.clip(mean_tpr + sd_tpr, 0, 1),
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    ax_roc.plot([0, 1], [0, 1], ls="--", lw=0.7, color="#A0A0A0")
    ax_roc.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False-positive rate", ylabel="True-positive rate")
    ax_roc.set_xticks([0, 0.5, 1.0])
    ax_roc.set_yticks([0, 0.5, 1.0])
    ax_roc.legend(loc="lower right", borderaxespad=0.2, handlelength=1.5)

    order = [name for name, _, _ in METHODS]
    x = np.arange(len(order))
    metric_colors = {"AUROC": "#222222", "AUPRC": "#56B4E9"}
    for offset, metric, marker in [(-0.10, "AUROC", "o"), (0.10, "AUPRC", "s")]:
        means = [float(summary.loc[summary.method.eq(m), f"{metric.lower()}_mean"].iloc[0]) for m in order]
        sds = [float(summary.loc[summary.method.eq(m), f"{metric.lower()}_sd"].iloc[0]) for m in order]
        ax_metric.errorbar(
            x + offset, means, yerr=sds, fmt=marker, ms=3.5, capsize=2,
            color=metric_colors[metric], ecolor=metric_colors[metric], lw=0.9,
            label=metric, zorder=3
        )
    ax_metric.set_ylim(0.88, 1.012)
    ax_metric.set_ylabel("Ranking metric")
    ax_metric.set_xticks(x, ["AP", "CTSV", "STANCE", "Celina", "spVC"], rotation=35, ha="right")
    ax_metric.set_yticks([0.90, 0.95, 1.00])
    ax_metric.grid(axis="y", color="#E6E6E6", lw=0.6)
    ax_metric.legend(loc="lower left", handletextpad=0.3)

    tp_means = np.array([summary.loc[summary.method.eq(m), "tp_mean"].iloc[0] for m in order])
    fp_means = np.array([summary.loc[summary.method.eq(m), "fp_mean"].iloc[0] for m in order])
    selected_sds = np.array(
        [summary.loc[summary.method.eq(m), "selected_pairs_sd"].iloc[0] for m in order]
    )
    bar_colors = [color for _, _, color in METHODS]
    ax_count.bar(x, tp_means, color=bar_colors, width=0.68, edgecolor="white", linewidth=0.4, label="True positive")
    ax_count.bar(
        x, fp_means, bottom=tp_means, color=bar_colors, alpha=0.32,
        hatch="////", width=0.68, edgecolor="#4A4A4A", linewidth=0.4,
        label="False positive"
    )
    totals = tp_means + fp_means
    ax_count.errorbar(x, totals, yerr=selected_sds, fmt="none", ecolor="#333333", lw=0.8, capsize=2)
    for xi, total in zip(x, totals):
        ax_count.text(xi, total + 5.0, f"{total:.0f}", ha="center", va="bottom", fontsize=6)
    ax_count.set_ylim(0, max(totals + selected_sds) * 1.18)
    ax_count.set_ylabel("Pairs selected at BH $q<0.05$")
    ax_count.set_xticks(x, ["AP", "CTSV", "STANCE", "Celina", "spVC"], rotation=35, ha="right")
    ax_count.legend(loc="upper right", handlelength=1.4, handletextpad=0.4)

    for label, axis in zip(["a", "b", "c"], [ax_roc, ax_metric, ax_count]):
        axis.text(-0.18, 1.10, label, transform=axis.transAxes, fontsize=8, fontweight="bold", va="top")

    fig.text(
        0.5, 0.965,
        "Exploratory pooled-method comparison on the fixed 40-gene panel",
        ha="center", va="top", fontsize=8, fontweight="bold"
    )
    fig.text(
        0.5, 0.035,
        "Mean $\\pm$ s.d. across three complete-input simulation repeats; 200 pre-specified gene-cell-type pairs per repeat.",
        ha="center", va="bottom", fontsize=6.2
    )

    stem = OUT_DIR / "fig_s_method_comparison_roc"
    fig.savefig(stem.with_suffix(".pdf"), dpi=600)
    fig.savefig(stem.with_suffix(".svg"), dpi=600)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    qa_text = """# Compact method-comparison figure QA

Core conclusion: All five approaches rank active pairs well in this compact pilot, but nominal BH calibration is method- and working-model-dependent.
Figure archetype: quantitative grid.
Target journal/output: Nature Communications supplementary figure; double-column, 179 mm x 67 mm.
Backend: Python (matplotlib) only.
Panel map: a, mean ROC with s.d.; b, AUROC/AUPRC mean and s.d.; c, mean true- and false-positive counts at BH q<0.05 with s.d. of total selections.
Evidence hierarchy: ranking is primary; threshold calibration is validation; the compact fixed-panel design is a stated limitation.
Statistics: n=3 independent simulation repeats; center=mean; spread=sample s.d.; BH is recomputed once over the same 200 pair-level p-values for every method.
Source data: compact_method_comparison_pairs.csv, compact_method_comparison_per_repeat.csv, compact_method_comparison_summary.csv.
Image integrity: vector PDF/SVG and 600-dpi TIFF exported directly from the same script; no raster image manipulation.
Reviewer risk: the pilot is deliberately small, preprocessing and test statistics differ by method, and it is not a genome-wide benchmark.
"""
    (OUT_DIR / "compact_method_comparison_qa_notes.md").write_text(qa_text, encoding="ascii")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
