"""Re-plot Figure 1 (complete) and Figure 2 (missing) with a taller, more legible
layout, using ONLY the saved source-data CSVs so that no simulation numbers change.

Layout: a top strip with the training curves and a larger 2xK heatmap block below,
so the coefficient heatmaps and CT labels are no longer cramped.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path(__file__).resolve().parents[2] / "figures" / "simulation_robustness"
LATEX_FIG_DIR = (
    Path(__file__).resolve().parents[3]
    / "05_论文草稿" / "latex" / "figs" / "simulation_robustness"
)
SOURCE_DIR = LATEX_FIG_DIR / "source_data"

PALETTE = {"ap": "#1f4e79", "missing": "#B0B0B0", "available": "#3F7F4C"}


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "savefig.dpi": 450,
    })


def load_matrices(condition: str, genes: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    df = pd.read_csv(SOURCE_DIR / "selected_coefficient_matrices.csv")
    df = df[(df["condition"] == condition) & (df["method"] == "AP-FedAdam") & (df["repeat"] == 0)]
    truth: dict[int, np.ndarray] = {}
    learned: dict[int, np.ndarray] = {}
    for gene in genes:
        sub = df[df["gene"] == gene]
        t = np.zeros((2, 5))
        l = np.zeros((2, 5))
        for _, row in sub.iterrows():
            b = int(row["basis"]) - 1
            k = int(row["cell_type"]) - 1
            t[b, k] = row["true_beta"]
            l[b, k] = row["learned_beta"]
        truth[gene] = t
        learned[gene] = l
    return truth, learned


def load_history(condition: str) -> pd.DataFrame:
    df = pd.read_csv(SOURCE_DIR / "simulation_training_history.csv")
    df = df[(df["condition"] == condition) & (df["method"] == "AP-FedAdam") & (df["repeat"] == 0)]
    return df.sort_values("round")


def draw_curve(ax: plt.Axes, hist: pd.DataFrame, col: str, title: str, log: bool = False) -> None:
    ax.plot(hist["round"], hist[col], color=PALETTE["ap"], linewidth=1.4)
    ax.scatter(hist["round"].iloc[-1], hist[col].iloc[-1], color=PALETTE["ap"], s=12, zorder=3)
    if log:
        ax.set_yscale("log")
    ax.set_title(title, loc="left", pad=3, fontweight="bold", fontsize=8.5)
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("NB loss" if col == "loss" else "Update norm", fontsize=8)
    ax.set_xlim(1, int(hist["round"].max()))
    ax.tick_params(labelsize=7)


def draw_heatmap(ax: plt.Axes, m: np.ndarray, title: str, vlim: float, show_y: bool, show_x: bool):
    img = ax.imshow(m, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_title(title, pad=3, fontsize=8, fontweight="bold")
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels([f"CT{k+1}" for k in range(m.shape[1])] if show_x else [], fontsize=7.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["h1", "h2"] if show_y else [], fontsize=8)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    return img


def build_figure(condition: str, genes: list[int], suptitle: str, stem: str,
                 missing_panel: bool = False) -> None:
    truth, learned = load_matrices(condition, genes)
    hist = load_history(condition)
    allvals = np.concatenate([np.abs(truth[g]).ravel() for g in genes]
                             + [np.abs(learned[g]).ravel() for g in genes])
    vlim = float(allvals.max()) * 1.05

    fig = plt.figure(figsize=(9.6, 5.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.3], hspace=0.42)

    n_top = 3 if missing_panel else 2
    top = outer[0].subgridspec(1, n_top, wspace=0.42)
    if missing_panel:
        ax_av = fig.add_subplot(top[0, 0])
        avail = np.ones((3, 5))
        avail[1, [1, 3]] = 0.0
        ax_av.imshow(avail, cmap=mpl.colors.ListedColormap([PALETTE["missing"], PALETTE["available"]]),
                     vmin=0, vmax=1, aspect="auto")
        ax_av.set_xticks(range(5)); ax_av.set_xticklabels([f"CT{k+1}" for k in range(5)], fontsize=7.5)
        ax_av.set_yticks(range(3)); ax_av.set_yticklabels([f"C{k+1}" for k in range(3)], fontsize=8)
        ax_av.set_title("Observed cell-type panel", loc="left", pad=3, fontweight="bold", fontsize=8.5)
        ax_av.tick_params(length=0, pad=2)
        for c in (1, 3):
            ax_av.text(c, 1.0, "miss", ha="center", va="center", color="white", fontsize=6.5, fontweight="bold")
        ax_loss = fig.add_subplot(top[0, 1])
        ax_norm = fig.add_subplot(top[0, 2])
    else:
        ax_loss = fig.add_subplot(top[0, 0])
        ax_norm = fig.add_subplot(top[0, 1])
    ax_loss.text(-0.14, 1.06, "a", transform=ax_loss.transAxes, fontsize=11, fontweight="bold")
    draw_curve(ax_loss, hist, "loss", "Training loss")
    draw_curve(ax_norm, hist, "update_norm", "Update norm", log=True)

    ng = len(genes)
    bottom = outer[1].subgridspec(2, ng, wspace=0.30, hspace=0.32)
    img = None
    for col, gene in enumerate(genes):
        ax_t = fig.add_subplot(bottom[0, col])
        img = draw_heatmap(ax_t, truth[gene], f"G{gene}: true", vlim, col == 0, False)
        ax_l = fig.add_subplot(bottom[1, col])
        draw_heatmap(ax_l, learned[gene], f"G{gene}: learned", vlim, col == 0, True)
        if col == 0:
            ax_t.text(-0.42, 1.10, "b", transform=ax_t.transAxes, fontsize=11, fontweight="bold")
    cbar = fig.colorbar(img, ax=fig.axes[n_top:], fraction=0.018, pad=0.015)
    cbar.set_label("Spatial coefficient", fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2)
    fig.suptitle(suptitle, x=0.5, y=0.99, fontsize=9.5, fontweight="bold")

    for suffix, kw in {".svg": {}, ".pdf": {}, ".png": {"dpi": 450}, ".tiff": {"dpi": 600}}.items():
        fig.savefig(FIG_DIR / f"{stem}{suffix}", bbox_inches="tight", **kw)
        fig.savefig(LATEX_FIG_DIR / f"{stem}{suffix}", bbox_inches="tight", **kw)
    plt.close(fig)


def main() -> None:
    configure()
    manifest = json.load(open(SOURCE_DIR / "simulation_manifest.json"))
    genes = manifest["candidate_genes"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    build_figure("complete", genes,
                 "Complete-client simulation: AP-FedAdam recovers sparse spatial coefficients",
                 "fig1_complete_recovery")
    build_figure("missing", genes,
                 "Incomplete-client simulation: block-wise federation retains coefficient recovery",
                 "fig2_missing_celltypes", missing_panel=True)
    print("Re-plotted fig1 and fig2 to", FIG_DIR, "and", LATEX_FIG_DIR)


if __name__ == "__main__":
    main()
