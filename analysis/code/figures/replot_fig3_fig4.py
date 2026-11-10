"""Re-plot fig3 (coordinate rotation) and fig4 (FedAvg comparison) from saved
source-data CSVs (compact 3-client benchmark), without PyTorch.

fig3 change: show TRUE and LEARNED coefficient matrices in separate columns
(like fig1/fig2), with numeric value annotations. The MAE-versus-angle panel is
reported in the source-data table rather than duplicated in the figure.
fig4 change: move the method legend to upper-center with headroom so it does not
overlap the bars.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "04_数值实验"
FIG_DIR = ROOT / "figures" / "simulation_robustness"
LATEX = ROOT / "overleaf" / "figs" / "simulation_robustness"
SRC = LATEX / "source_data"
AP, FED = "#1f4e79", "#b4382f"
GENES = sorted(pd.read_csv(SRC / "selected_coefficient_matrices.csv").gene.unique().tolist())


def matrices(cond, which="learned_beta"):
    d = pd.read_csv(SRC / "selected_coefficient_matrices.csv")
    d = d[(d.condition == cond) & (d.method == "AP-FedAdam") & (d.repeat == 0)]
    out = {}
    for g in GENES:
        s = d[d.gene == g]; m = np.zeros((2, 5))
        for _, r in s.iterrows():
            m[int(r.basis) - 1, int(r.cell_type)] = r[which]
        out[g] = m
    return out


def heat(ax, M, title, vlim, show_y):
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_title(title, fontsize=7.6, fontweight="bold", pad=3)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"CT{k}" for k in range(5)], fontsize=6.8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["h1", "h2"] if show_y else [], fontsize=7.5)
    ax.tick_params(length=0)
    for i in range(2):
        for j in range(5):
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.2,
                    color="white" if abs(v) > 0.55 * vlim else "#222")
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    return im


def fig3():
    show = GENES[:3]
    true = matrices("complete", "true_beta")
    aligned = matrices("complete", "learned_beta")
    rot = matrices("rotation90", "learned_beta")
    vlim = max(np.abs(true[g]).max() for g in show) * 1.05
    fig = plt.figure(figsize=(7.2, 4.5))
    inner = fig.add_gridspec(len(show), 3, hspace=0.55, wspace=0.28)
    coltitles = ["true", "learned (aligned)", "learned (90 deg rot.)"]
    im = None
    for ri, g in enumerate(show):
        mats = [true[g], aligned[g], rot[g]]
        for ci, M in enumerate(mats):
            ax = fig.add_subplot(inner[ri, ci])
            im = heat(ax, M, f"G{g}: {coltitles[ci]}", vlim, ci == 0)
    cb = fig.colorbar(im, ax=fig.axes, fraction=0.018, pad=0.018)
    cb.set_label("Spatial coefficient", fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
    fig.suptitle("Coefficient recovery under aligned and rotated client coordinates",
                 x=0.5, y=1.02, fontsize=9.5, fontweight="bold")
    for suffix, kw in {".svg": {}, ".pdf": {}, ".png": {"dpi": 400}, ".tiff": {"dpi": 500}}.items():
        fig.savefig(FIG_DIR / f"fig3_coordinate_rotation{suffix}", bbox_inches="tight", **kw)
        fig.savefig(LATEX / f"fig3_coordinate_rotation{suffix}", bbox_inches="tight", **kw)
    plt.close(fig)


def fig4():
    summ = pd.read_csv(SRC / "simulation_summary.csv")
    hist = pd.read_csv(SRC / "simulation_training_history.csv")
    conds = ["complete", "missing"]; labels = ["Complete", "Missing CT"]
    colors = {"AP-FedAdam": AP, "FedAvg": FED}
    fig = plt.figure(figsize=(6.4, 5.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.6, wspace=0.32)
    axb = fig.add_subplot(grid[0, :])
    axb.text(-0.06, 1.14, "a", transform=axb.transAxes, fontsize=12, fontweight="bold")
    sub = summ[summ.condition.isin(conds)].copy()
    agg = sub.groupby(["condition", "method"], as_index=False).agg(mean=("mae", "mean"), sd=("mae", "std")); agg["sd"] = agg["sd"].fillna(0)
    x = np.arange(len(conds)); w = 0.34
    top = 0
    for mi, method in enumerate(["AP-FedAdam", "FedAvg"]):
        vals = agg[agg.method == method].set_index("condition").loc[conds]
        bars = axb.bar(x + (mi - 0.5) * w, vals["mean"], width=w, yerr=vals["sd"], capsize=2.5,
                       color=colors[method], edgecolor="black", linewidth=0.45, label=method)
        top = max(top, float((vals["mean"] + vals["sd"]).max()))
        for b, v in zip(bars, vals["mean"]):
            axb.text(b.get_x() + b.get_width() / 2, b.get_height() + agg["mean"].max() * 0.02, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=5.8)
    axb.set_xticks(x); axb.set_xticklabels(labels)
    axb.set_ylabel("Coefficient MAE")
    axb.set_ylim(0, top * 1.32)
    axb.set_title("AP-FedAdam versus tuned FedAvg", loc="left", pad=4, fontweight="bold")
    axb.legend(ncol=2, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    for pi, (cond, lab) in enumerate(zip(conds, labels)):
        ax = fig.add_subplot(grid[1, pi])
        ax.text(-0.2, 1.05, chr(ord("b") + pi), transform=ax.transAxes, fontsize=11, fontweight="bold")
        view = hist[(hist.condition == cond) & (hist.repeat == 0)]
        for method in ["AP-FedAdam", "FedAvg"]:
            ln = view[view.method == method].sort_values("round")
            ax.plot(ln["round"], ln["loss"], color=colors[method], lw=1.3, label=method)
        ax.set_title(lab, loc="left", pad=3, fontweight="bold", fontsize=8.5)
        ax.set_xlabel("Round", fontsize=8); ax.set_ylabel("NB loss" if pi == 0 else "", fontsize=8)
        ax.set_xlim(1, int(view["round"].max())); ax.tick_params(labelsize=7)
    fig.suptitle("Federated optimizer comparison on complete and incomplete cell-type inputs",
                 x=0.5, y=1.0, fontsize=9.2, fontweight="bold")
    for suffix, kw in {".svg": {}, ".pdf": {}, ".png": {"dpi": 400}, ".tiff": {"dpi": 500}}.items():
        fig.savefig(FIG_DIR / f"fig4_apfedadam_vs_fedavg{suffix}", bbox_inches="tight", **kw)
        fig.savefig(LATEX / f"fig4_apfedadam_vs_fedavg{suffix}", bbox_inches="tight", **kw)
    plt.close(fig)


if __name__ == "__main__":
    fig3(); fig4(); print("re-plotted fig3 and fig4")
