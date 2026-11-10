#!/usr/bin/env python3
'Publish traceable tables and the supplementary offset sensitivity figure.'
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_revision import ROOT, OUTPUT

TABLES = ROOT / "05_论文草稿/latex/tables/distrisvg_2026-09-18"
FIGURES = ROOT / "05_论文草稿/latex/figs/distrisvg_2026-09-18"
LABELS = {"ap_fedadam": "DistriSVG/AP", "ctsv": "CTSV", "stance": "STANCE", "celina": "Celina", "spvc": "spVC"}


def fmt(value):
    return "--" if pd.isna(value) else f"{value:.3f}"


def mean_sd(row, key):
    return f"{row[key + '_mean']:.3f} $\\pm$ {row[key + '_std']:.3f}"


def table(name, columns, rows, align=None):
    align = align or "l" + "r" * (len(columns) - 1)
    text = [rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}", r"\toprule", " & ".join(columns) + r" \\", r"\midrule"]
    text += [" & ".join(map(str, row)) + r" \\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / name).write_text("\n".join(text) + "\n")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(OUTPUT / "archive_competitor_metrics_summary.csv")
    keys = ["power", "precision", "fdp", "f1", "auprc"]
    for condition in ("complete", "missing"):
        rows = [[LABELS[r.method]] + [mean_sd(r, key) for key in keys]
                for _, r in comp[comp.condition == condition].iterrows()]
        name = "compact_discovery.tex" if condition == "complete" else "compact_missing_discovery.tex"
        table(name, ["Method", "Power", "Precision", "FDP", "F1", "AUPRC"], rows)
    large = pd.read_csv(OUTPUT / "archive_g1000_detection_metrics.csv")
    rows = []
    for _, row in large.iterrows():
        label = f"{row.scenario.title()}/{row.data_mode}"
        proc = "Joint" if row.procedure == "joint_all_bases" else "Matched"
        rows.append([label, proc] + [fmt(row[k]) for k in keys])
    table("large_discovery.tex", ["Input/generator", "Score", "Power", "Precision", "FDP", "F1", "AUPRC"], rows, "llrrrrr")
    sim = pd.read_csv(OUTPUT / "offset_simulation_summary.csv")
    for condition in ("complete", "missing", "rotation90"):
        rows = []
        for _, r in sim[sim.condition == condition].iterrows():
            rows.append([r.method, "Library" if r.offset == "library" else "None"] +
                        [mean_sd(r, k) for k in ("mae", "power", "fdp", "auprc")])
        table(f"offset_{condition}.tex", ["Method", "Offset", "MAE", "Power", "FDP", "AUPRC"], rows, "llrrrr")
    real = pd.read_csv(OUTPUT / "real_offset_summary.csv")
    table("real_offset.tex", ["Dataset", "Offset", "Pairs tested", "Pairs selected", "Genes selected", "NLL"],
          [[r.dataset, "Library" if r.offset == "library" else "None", r.n_pairs, r.selected_pairs,
            r.selected_genes, fmt(r.mean_family_nll)] for _, r in real.iterrows()], "llrrrr")
    ranks = pd.read_csv(OUTPUT / "real_offset_rank_sensitivity.csv")
    table("real_offset_ranks.tex", ["Dataset", "Rank correlation", "Top-100 overlap", "Top-100 Jaccard"],
          [[r.dataset, fmt(r.spearman_p), f"{r.top100_overlap}/100", fmt(r.top100_jaccard)] for _, r in ranks.iterrows()])

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .7, "legend.frameon": False})
    fig, axs = plt.subplots(2, 2, figsize=(7.01, 5.25))
    fig.subplots_adjust(left=.10, right=.98, bottom=.11, top=.89, wspace=.35, hspace=.65)
    raw = pd.read_csv(OUTPUT / "offset_simulation_per_repeat.csv")
    raw = raw[raw.method == "AP-FedAdam"]
    colors = {"none": "#0072B2", "library": "#D55E00"}
    for ax, metric, title, ylabel in ((axs[0, 0], "mae", "Coefficient recovery", "Mean absolute error"),
                                      (axs[0, 1], "fdp", "Threshold diagnostic", "Empirical FDP")):
        for i, condition in enumerate(("complete", "missing", "rotation90")):
            for rep in range(3):
                values = raw[(raw.condition == condition) & (raw["repeat"] == rep)].set_index("offset")
                x = np.array([i - .13, i + .13]) + (rep - 1) * .027
                ax.plot(x, values.loc[["none", "library"], metric], color="#B0B0B0", lw=.6, zorder=1)
                for j, offset in enumerate(("none", "library")):
                    ax.scatter(x[j], values.loc[offset, metric], color=colors[offset], s=21, marker="o" if j == 0 else "s", zorder=2)
        ax.set(xticks=[0, 1, 2], xticklabels=["Complete", "Missing", "90-degree\nrotation"], ylabel=ylabel, title=title)
    axs[0, 0].set_ylim(0, .34)
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].axhline(.05, color="#777777", ls=":", lw=.8)
    ax = axs[1, 0]
    for i, dataset in enumerate(("PDAC", "Mouse")):
        for j, offset in enumerate(("none", "library")):
            row = real[(real.dataset == dataset) & (real.offset == offset)].iloc[0]
            x = i + (j - .5) * .32
            ax.bar(x, row.selected_pairs / row.n_pairs, width=.28, color=colors[offset])
            ax.text(x, row.selected_pairs / row.n_pairs + .035, str(int(row.selected_pairs)), ha="center", fontsize=7)
    ax.set(xticks=[0, 1], xticklabels=["PDAC\n2,400 pairs", "Mouse\n1,120 pairs"],
           ylim=(0, 1), ylabel="Fraction passing q < 0.05", title="Real-data candidate counts")
    ax = axs[1, 1]
    ax.bar([0, 1], ranks.top100_overlap, color="#4B7772", width=.48)
    for i, row in ranks.iterrows():
        ax.text(i, row.top100_overlap + 3, str(row.top100_overlap), ha="center", fontsize=8)
    ax.set(xticks=[0, 1], xticklabels=["PDAC", "Mouse"], ylim=(0, 100),
           ylabel="Pairs retained in both top-100 lists", title="Sensitivity of candidate ranks")
    for letter, ax in zip("abcd", axs.ravel()):
        ax.text(-.18, 1.12, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    handles = [plt.Line2D([], [], color=color, marker="o" if key == "none" else "s", linestyle="", label="No offset" if key == "none" else "Library offset") for key, color in colors.items()]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(.5, .995))
    for suffix in ("pdf", "svg", "png", "tiff"):
        fig.savefig(FIGURES / f"offset_sensitivity.{suffix}", dpi=600 if suffix == "tiff" else 200)
    plt.close(fig)
    for name in ("offset_simulation_per_repeat.csv", "real_offset_summary.csv", "real_offset_rank_sensitivity.csv"):
        shutil.copy2(OUTPUT / name, FIGURES / name)
    print("Tables and supplementary figure exported", flush=True)


if __name__ == "__main__":
    main()
