"""Regenerate fig1 (complete) and fig2 (missing cell types) from the large-scale
G=1000, 8-client, L1=2e-7 federated CTSV runs, with numeric value annotations in
each coefficient heatmap (matching the earlier high-quality result).

Data source: existing inference CSVs (True_Beta / Learned_Beta already computed by
the user's PyTorch runs), so no re-fitting is needed.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[3] / "04_数值实验/code"
ROOT = Path(__file__).resolve().parents[3] / "04_数值实验"
FIG_DIR = ROOT / "figures" / "simulation_robustness"
LATEX_FIG_DIR = ROOT / "overleaf" / "figs" / "simulation_robustness"
SRC = LATEX_FIG_DIR / "source_data"

COMPLETE_CSV = CODE / "new_test1-5" / "CTSV_Fed_8_G_1000_L1_2e-07_Round1000_h_DataMode_linear_BasisMode_linear_Inference_FIXED.csv"
MISSING_CSV = CODE / "missing_celltype_test" / "CTSV_Fed_BLOCKWISE_missingBlocks_8_G_1000_L1_2e-07_Round1000_h_DataMode_linear_BasisMode_linear_Inference_FIXED.csv"
LOSS_CSV = CODE / "loss_history.csv"
K, G, L = 5, 1000, 2
MISSING_BY_CLIENT = {2: [4], 5: [1, 3], 7: [0, 2, 4]}


def load_beta(csv):
    d = pd.read_csv(csv)
    f2l = {"h1": 0, "h2": 1}
    true = np.zeros((L, K, G)); learn = np.zeros((L, K, G))
    for _, r in d.iterrows():
        g = int(r["Gene_ID"]); k = int(r["Cell_Type"]); l = f2l[str(r["Feature"])]
        true[l, k, g] = r["True_Beta"]; learn[l, k, g] = r["Learned_Beta"]
    return true, learn


def pick_genes_distinct_types(true, n=5):
    """For each cell type pick the gene with the largest true loading (distinct genes)."""
    picks = []
    strength = np.sum(np.abs(true), axis=0)
    for k in range(min(n, K)):
        order = np.argsort(strength[k])[::-1]
        for g in order:
            if g not in picks:
                picks.append(int(g)); break
    return picks


def heat(ax, M, title, vlim=5.0, show_y=True):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)
    ax.set_xticks(range(K)); ax.set_xticklabels([f"CT{k}" for k in range(K)], fontsize=7.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["h1", "h2"] if show_y else [], fontsize=8)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.0,
                    color="white" if abs(v) > 2.6 else "#222222")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    return im


def build(true, learn, genes, suptitle, stem, missing_note=None):
    fig = plt.figure(figsize=(2.25 * len(genes) + 0.6, 4.4))
    grid = fig.add_gridspec(2, len(genes), hspace=0.42, wspace=0.28)
    im = None
    for c, g in enumerate(genes):
        axt = fig.add_subplot(grid[0, c])
        im = heat(axt, true[:, :, g], f"G{g}: true", show_y=(c == 0))
        if c == 0:
            axt.text(-0.5, 1.10, "a", transform=axt.transAxes, fontsize=12, fontweight="bold")
        axb = fig.add_subplot(grid[1, c])
        heat(axb, learn[:, :, g], f"G{g}: learned", show_y=(c == 0))
    cbar = fig.colorbar(im, ax=fig.axes, fraction=0.015, pad=0.012)
    cbar.set_label("Spatial coefficient", fontsize=8); cbar.ax.tick_params(labelsize=7)
    fig.suptitle(suptitle, x=0.5, y=1.02, fontsize=9.2, fontweight="bold")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kw in {".svg": {}, ".pdf": {}, ".png": {"dpi": 400}, ".tiff": {"dpi": 500}}.items():
        fig.savefig(FIG_DIR / f"{stem}{suffix}", bbox_inches="tight", **kw)
        fig.savefig(LATEX_FIG_DIR / f"{stem}{suffix}", bbox_inches="tight", **kw)
    plt.close(fig)


def main():
    tc, lc = load_beta(COMPLETE_CSV)
    genes_c = pick_genes_distinct_types(tc)
    corr_c = np.corrcoef(tc.ravel(), lc.ravel())[0, 1]
    build(tc, lc, genes_c,
          f"Complete-input recovery (G=1000, 8 clients)  [r={corr_c:.3f}]",
          "fig1_complete_recovery")

    tm, lm = load_beta(MISSING_CSV)
    dropped = sorted({k for v in MISSING_BY_CLIENT.values() for k in v})
    exclude = set(genes_c)
    strength = np.sum(np.abs(tm), axis=0)
    genes_m = []
    for k in dropped[:5]:
        for g in np.argsort(strength[k])[::-1]:
            g = int(g)
            if g not in genes_m and g not in exclude:
                genes_m.append(g); break
    corr_m = np.corrcoef(tm.ravel(), lm.ravel())[0, 1]
    build(tm, lm, genes_m,
          f"Incomplete-input recovery (block-wise aggregation)  [r={corr_m:.3f}]",
          "fig2_missing_celltypes")
    print("complete genes", genes_c, "corr", round(corr_c, 3))
    print("missing  genes", genes_m, "corr", round(corr_m, 3), "dropped types", dropped)


if __name__ == "__main__":
    main()
