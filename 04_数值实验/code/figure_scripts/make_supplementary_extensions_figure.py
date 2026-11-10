#!/usr/bin/env python3
"""Supplementary implementation-example figure for single-cell ST and 3D ST.

The inputs are the demo CSV outputs produced by
`run_real_data_prompt_experiments.py`. The figure is intentionally labelled as
an implementation example rather than a completed real-data validation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "overleaf" / "figs" / "real_data_prompt"
OUT_STEM = DATA_DIR / "supplementary_extensions_singlecell_3d"

CELL_TYPE_ORDER = ["Neuron", "Astrocyte", "Oligodendrocyte", "Microglia"]
CELL_TYPE_COLORS = {
    "Neuron": "#3B6FB6",
    "Astrocyte": "#3F9B6D",
    "Oligodendrocyte": "#B77932",
    "Microglia": "#8E5EA2",
}
FEATURE_COLORS = {"h_x": "#3B6FB6", "h_y": "#3F9B6D", "h_z": "#C14A3D"}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def draw_panel_a(ax: plt.Axes, onehot_df: pd.DataFrame) -> None:
    ax.set_axis_off()
    panel_label(ax, "a")
    ax.set_title("Single-cell ST omega construction", loc="left", pad=8)

    examples = onehot_df.head(5).copy()
    x0, y0 = 0.03, 0.78
    row_h = 0.105
    ax.text(x0, 0.95, "observed label", weight="bold", ha="left", va="center")
    ax.text(0.51, 0.95, r"one-hot $\Omega$", weight="bold", ha="left", va="center")

    for i, (_, row) in enumerate(examples.iterrows()):
        y = y0 - i * row_h
        ct = row["cell_type"]
        ax.add_patch(
            Rectangle(
                (x0, y - 0.032),
                0.32,
                0.064,
                facecolor=CELL_TYPE_COLORS[ct],
                edgecolor="none",
                alpha=0.18,
                transform=ax.transAxes,
            )
        )
        ax.text(x0 + 0.014, y, ct, ha="left", va="center", color="#222222")
        values = [int(row[ct_col]) for ct_col in CELL_TYPE_ORDER]
        for j, value in enumerate(values):
            bx = 0.51 + j * 0.075
            face = "#303030" if value else "#FFFFFF"
            ax.add_patch(
                Rectangle(
                    (bx, y - 0.029),
                    0.048,
                    0.058,
                    facecolor=face,
                    edgecolor="#303030",
                    linewidth=0.45,
                    transform=ax.transAxes,
                )
            )

    ax.add_patch(
        FancyArrowPatch(
            (0.38, 0.56),
            (0.48, 0.56),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.8,
            color="#333333",
            transform=ax.transAxes,
        )
    )
    for j, ct in enumerate(CELL_TYPE_ORDER):
        ax.text(0.535 + j * 0.075, 0.24, ct[:3], ha="center", va="center", fontsize=6)

    ax.text(
        0.03,
        0.08,
        r"$\Omega_{ic}=1\{\mathrm{label}_i=c\}$; each row sums to 1.",
        ha="left",
        va="center",
        color="#333333",
    )


def draw_panel_b(ax: plt.Axes, onehot_df: pd.DataFrame, counts: pd.DataFrame) -> None:
    panel_label(ax, "b")
    ax.set_title("Example single-cell label map", loc="left", pad=8)
    for ct in CELL_TYPE_ORDER:
        sub = onehot_df[onehot_df["cell_type"] == ct]
        n_cells = int(counts.loc[counts["cell_type"] == ct, "n_cells"].iloc[0])
        ax.scatter(
            sub["x"],
            sub["y"],
            s=18,
            color=CELL_TYPE_COLORS[ct],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.88,
            label=f"{ct} ({n_cells})",
        )
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.02), borderaxespad=0)


def draw_panel_c(ax: plt.Axes, onehot_df: pd.DataFrame) -> None:
    panel_label(ax, "c")
    ax.set_title("One-hot omega matrix", loc="left", pad=8)
    sorted_df = onehot_df.assign(
        cell_type=pd.Categorical(onehot_df["cell_type"], categories=CELL_TYPE_ORDER, ordered=True)
    ).sort_values(["cell_type", "cell_id"])
    mat = sorted_df[CELL_TYPE_ORDER].to_numpy(dtype=float)
    cmap = ListedColormap(["#FFFFFF", "#303030"])
    ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(CELL_TYPE_ORDER)))
    ax.set_xticklabels(["Neuron", "Astro.", "Oligo.", "Micro."], rotation=0)
    ax.set_yticks([])
    ax.set_ylabel("cells sorted by label")
    for x in np.arange(-0.5, len(CELL_TYPE_ORDER), 1):
        ax.axvline(x, color="#CFCFCF", linewidth=0.35)
    boundaries = sorted_df["cell_type"].astype(str).ne(sorted_df["cell_type"].astype(str).shift()).to_numpy()
    for idx in np.flatnonzero(boundaries)[1:]:
        ax.axhline(idx - 0.5, color="#C14A3D", linewidth=0.55)
    ax.text(
        0.02,
        1.02,
        "one non-zero cell-type weight per cell",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#333333",
        fontsize=6.5,
    )


def draw_panel_d(ax: plt.Axes, basis_summary: pd.DataFrame) -> None:
    panel_label(ax, "d")
    ax.set_title("3D basis summary", loc="left", pad=8)
    modes = ["linear3d", "cos3d", "gau3d"]
    features = ["h_x", "h_y", "h_z"]
    offsets = {"h_x": -0.22, "h_y": 0.0, "h_z": 0.22}
    for mi, mode in enumerate(modes):
        for feature in features:
            row = basis_summary[(basis_summary["mode"] == mode) & (basis_summary["feature"] == feature)].iloc[0]
            x = mi + offsets[feature]
            ax.vlines(
                x,
                row["min"],
                row["max"],
                color=FEATURE_COLORS[feature],
                linewidth=1.4,
                alpha=0.9,
            )
            ax.scatter(
                x,
                row["mean"],
                s=18,
                color=FEATURE_COLORS[feature],
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
                label=feature if mi == 0 else None,
            )
    ax.axhline(0, color="#9A9A9A", linewidth=0.55, linestyle="--", zorder=0)
    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels(["linear", "cosine", "Gaussian"])
    ax.set_ylabel("basis value range")
    ax.set_xlabel("3D basis mode")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", ncol=3, handletextpad=0.3, columnspacing=0.7)
    ax.text(
        0.02,
        0.04,
        r"$h(s)=[h_x(x),h_y(y),h_z(z)]$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#333333",
        fontsize=6.5,
    )


def main() -> None:
    configure_matplotlib()
    onehot = pd.read_csv(DATA_DIR / "single_cell_st_onehot_omega_demo.csv")
    counts = pd.read_csv(DATA_DIR / "single_cell_st_onehot_omega_demo_summary.csv")
    basis_summary = pd.read_csv(DATA_DIR / "spot_level_3d_basis_demo_summary.csv")

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.02, 1.0],
        height_ratios=[1.0, 1.05],
        wspace=0.48,
        hspace=0.52,
    )

    draw_panel_a(fig.add_subplot(gs[0, 0]), onehot)
    draw_panel_b(fig.add_subplot(gs[0, 1]), onehot, counts)
    draw_panel_c(fig.add_subplot(gs[1, 0]), onehot)
    draw_panel_d(fig.add_subplot(gs[1, 1]), basis_summary)

    for ext, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 600},
        "tiff": {"dpi": 600},
    }.items():
        fig.savefig(OUT_STEM.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)

    qa = OUT_STEM.with_name(f"{OUT_STEM.name}_qa_notes.md")
    qa.write_text(
        "\n".join(
            [
                "# Supplementary extensions figure QA",
                "",
                "- Inputs: single_cell_st_onehot_omega_demo.csv, single_cell_st_onehot_omega_demo_summary.csv, spot_level_3d_basis_demo_summary.csv.",
                "- Exported formats: PDF, SVG, PNG, TIFF.",
                "- Interpretation: implementation examples only; not formal single-cell ST or 3D ST real-data validation.",
                "- Panel a encodes the one-hot omega rule from observed single-cell labels.",
                "- Panel b shows the simulated single-cell label map and cell-type counts.",
                "- Panel c visualizes the one-hot omega matrix.",
                "- Panel d summarizes min-to-max ranges and means for x, y, and z basis features under three 3D basis modes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
