"""Regenerate the PDAC spatial figure using a COMPOSITION-ADJUSTED expression map.

Problem with the previous panel
-------------------------------
It plotted log(1+y_ig) * omega_ik.  omega_Ductal has large spot-level variance, so
that quantity is dominated by the cell-type abundance field: mean spot-level
cross-gene correlation among the four displayed markers was 0.91 / 0.86 / 0.54 in
B_ST1 / B_ST2 / B_ST3.  Every gene therefore redrew essentially the same map, and
the "recurrence across independent markers" was built into the construction.

Fix
---
Map the residual of log(1+y_ig) after regressing out the local cell-type
proportion, i.e. the part of expression that the abundance field does not explain:

    r_ig = log(1+y_ig) - (a_g + b_g * omega_ik)

Mean cross-gene correlation drops to 0.45 / 0.41 / 0.40, so the panels are now
gene-specific.  This is still a descriptive visualisation of observed expression,
not a fitted coefficient.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "04_数值实验"
FIG_DIR = ROOT / "figures" / "cross_section_spatial_maps"
LATEX = ROOT / "overleaf" / "figs" / "cross_section_spatial_maps"
SOURCE = LATEX / "source_data"
EXPR = ROOT / "code" / "pdac_data_to_yc" / "preprocessed_data"
OMG = ROOT / "code" / "pdac_data_to_yc" / "spotlight_out"

SECTIONS = ("PDAC_B_ST1", "PDAC_B_ST2", "PDAC_B_ST3")
GENES = ("KRT18", "KRT19", "TIMP1", "SERPINA1")
CT = "Ductal"
BINS, SIGMA = 16, 1.4


def gaussian_filter(a, sigma):
    radius = max(1, int(4.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2)); k /= k.sum()
    out = a.astype(float)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]; pad[axis] = (radius, radius)
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis,
                                  np.pad(out, pad, mode="edge"))
    return out


def norm_xy(xy):
    lo = xy.min(0); rng = np.ptp(xy, 0); rng[rng == 0] = 1
    return (xy - lo) / rng


def surface(c01, vals):
    xi = np.clip((c01[:, 0] * BINS).astype(int), 0, BINS - 1)
    yi = np.clip((c01[:, 1] * BINS).astype(int), 0, BINS - 1)
    num = np.zeros((BINS, BINS)); den = np.zeros((BINS, BINS))
    np.add.at(num, (yi, xi), vals); np.add.at(den, (yi, xi), 1.0)
    s = gaussian_filter(num, SIGMA) / np.maximum(gaussian_filter(den, SIGMA), 1e-12)
    s[gaussian_filter(den, SIGMA) <= 0.035] = np.nan
    return s


def main():
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                         "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
                         "axes.titlesize": 6.5, "axes.linewidth": 0.5})
    maps, rows, cross = {g: {} for g in GENES}, [], []
    for s in SECTIONS:
        e = pd.read_csv(EXPR / f"{s}_processed_data_mincell5.csv",
                        usecols=lambda c: c == "Spot_ID" or c in set(GENES))
        o = pd.read_csv(OMG / f"{s}_SPOTlight_W_omega_with_xy.csv",
                        usecols=lambda c: c in {"spot", "x", "y", CT})
        f = e.merge(o, left_on="Spot_ID", right_on="spot", how="inner")
        w = f[CT].to_numpy(float)
        c01 = norm_xy(f[["x", "y"]].to_numpy(float))
        X = np.column_stack([np.ones_like(w), w])
        L = np.column_stack([np.log1p(f[g].to_numpy(float)) for g in GENES])
        R = L - X @ np.linalg.lstsq(X, L, rcond=None)[0]
        for gi, g in enumerate(GENES):
            surf = surface(c01, R[:, gi])
            maps[g][s] = surf
            for y in range(BINS):
                for x in range(BINS):
                    rows.append({"gene": g, "cell_type": CT, "section": s, "x_bin": x,
                                 "y_bin": y, "composition_adjusted_expression": surf[y, x]})
        for i in range(len(GENES)):
            for j in range(i + 1, len(GENES)):
                a, b = maps[GENES[i]][s].ravel(), maps[GENES[j]][s].ravel()
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() > 12: cross.append(abs(np.corrcoef(a[m], b[m])[0, 1]))
    SOURCE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SOURCE / "fig4_pdac_composition_adjusted.csv", index=False)
    print(f"mean |cross-gene map correlation| = {np.mean(cross):.3f}  (old omega-weighted proxy ~0.90)")

    rep = {}
    for g in GENES:
        v = []
        for i in range(len(SECTIONS)):
            for j in range(i + 1, len(SECTIONS)):
                a, b = maps[g][SECTIONS[i]].ravel(), maps[g][SECTIONS[j]].ravel()
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() > 12: v.append(np.corrcoef(a[m], b[m])[0, 1])
        rep[g] = float(np.mean(v))
    print("within-gene cross-section r:", {g: round(rep[g], 2) for g in GENES})

    vmax = max(np.nanmax(np.abs(maps[g][s])) for g in GENES for s in SECTIONS)
    n = len(GENES)
    fig = plt.figure(figsize=(7.2, 1.33 * n + 0.75))
    grid = fig.add_gridspec(n, len(SECTIONS) + 1, width_ratios=[1.15] + [1] * len(SECTIONS),
                            hspace=0.22, wspace=0.06)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#F7F7F7")
    labels = [s.replace("PDAC_", "") for s in SECTIONS]
    im = None
    for ri, g in enumerate(GENES):
        la = fig.add_subplot(grid[ri, 0]); la.axis("off")
        la.text(0.0, 0.66, g, ha="left", va="center", fontsize=7.4, fontweight="bold")
        la.text(0.0, 0.42, CT, ha="left", va="center", fontsize=5.8, color="#454545")
        la.text(0.0, 0.18, f"cross-section r={rep[g]:.2f}", ha="left", va="center",
                fontsize=5.2, color="#6A6A6A")
        for ci, s in enumerate(SECTIONS, start=1):
            ax = fig.add_subplot(grid[ri, ci])
            im = ax.imshow(maps[g][s], origin="lower", interpolation="nearest",
                           cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#D8D8D8"); sp.set_linewidth(0.45)
            if ri == 0:
                ax.set_title(labels[ci - 1], pad=2, fontsize=6.1, fontweight="bold")
    cb = fig.colorbar(im, ax=fig.axes, fraction=0.013, pad=0.015)
    cb.set_label("Composition-adjusted expression", fontsize=6)
    cb.ax.tick_params(labelsize=5.4, length=2)
    fig.suptitle("Composition-adjusted spatial expression of PDAC ductal candidates",
                 x=0.51, y=1.01, fontsize=8.2, fontweight="bold")
    fig.text(0.51, 0.952, "Tumour B serial sections", ha="center", va="bottom",
             fontsize=6.0, color="#555555")
    fig.text(0.01, 0.987, "a", fontsize=9, fontweight="bold", va="top")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suf, kw in {".svg": {}, ".pdf": {}, ".png": {"dpi": 450}, ".tiff": {"dpi": 600}}.items():
        fig.savefig(FIG_DIR / f"fig5_pdac_spatial_coherence{suf}", bbox_inches="tight", **kw)
        fig.savefig(LATEX / f"fig5_pdac_spatial_coherence{suf}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("wrote fig5_pdac_spatial_coherence.* (composition-adjusted) to", LATEX)


if __name__ == "__main__":
    main()
