"""Replace the omega-dominated descriptive proxy in the PDAC spatial figure with the
FITTED cell-type-specific spatial effect.

Motivation
----------
The previous panel plotted log(1+y_ig) * omega_ik.  Because omega_Ductal has large
spot-level variance, that quantity is dominated by the cell-type abundance field:
across the four displayed ductal markers the spot-level proxy correlation was
0.86-0.91 in sections B_ST1/B_ST2, so every gene reproduced essentially the same
map.  The apparent "recurrence across independent markers" was therefore largely
built into the construction rather than being gene-specific evidence.

Fix
---
Plot the model's own estimand: the fitted ductal spatial effect

    eta_g(s) = beta_{g,Ductal,h1} * h1(s) + beta_{g,Ductal,h2} * h2(s)

evaluated on each section's own coordinates.  This differs between genes (because
beta differs) and between sections (because the coordinate layout differs), and it
is a fitted coefficient surface, not a descriptive proxy.

Basis functions reproduce BasisGenerator in
code/missing_celltype_test/demo_FedAdamW_test1-6_yyq.py.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures" / "cross_section_spatial_maps"
LATEX_FIG_DIR = ROOT / "overleaf" / "figs" / "cross_section_spatial_maps"
SOURCE_DIR = LATEX_FIG_DIR / "source_data"
OMG = ROOT / "code" / "pdac_data_to_yc" / "spotlight_out"
RANKED = ROOT / "overleaf" / "figs" / "pdac_ctsv_all_modes_ranked.csv"

SECTIONS = ("PDAC_B_ST1", "PDAC_B_ST2", "PDAC_B_ST3")
GENES = ("KRT18", "KRT19", "TIMP1", "SERPINA1")
CELL_TYPE = "Ductal"
MODE = "gau1"
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


def configure():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
        "axes.titlesize": 6.5, "axes.linewidth": 0.5,
    })


class BasisGenerator:
    """Mirrors the generator used when the coefficients were fitted."""

    def __init__(self, mode, s_coords):
        self.mode = mode.lower()
        m = s_coords.mean(0, keepdims=True)
        sd = s_coords.std(0, keepdims=True) + 1e-8
        sn = (s_coords - m) / sd
        a = np.abs(sn)
        self.psi40 = np.array([np.quantile(a[:, 0], 0.4), np.quantile(a[:, 1], 0.4)])
        self.psi60 = np.array([np.quantile(a[:, 0], 0.6), np.quantile(a[:, 1], 0.6)])
        s1, s2 = sn[:, 0], sn[:, 1]
        h1 = np.exp(-(s1 ** 2) / (2 * self.psi40[0] ** 2))
        h2 = np.exp(-(s2 ** 2) / (2 * self.psi40[1] ** 2))
        self.gau1_mean = np.array([h1.mean(), h2.mean()])
        self.gau1_std = np.array([h1.std() + 1e-8, h2.std() + 1e-8])
        h1 = np.exp(-(s1 ** 2) / (2 * self.psi60[0] ** 2))
        h2 = np.exp(-(s2 ** 2) / (2 * self.psi60[1] ** 2))
        self.gau2_mean = np.array([h1.mean(), h2.mean()])
        self.gau2_std = np.array([h1.std() + 1e-8, h2.std() + 1e-8])

    def compute_h(self, s):
        m = s.mean(0, keepdims=True); sd = s.std(0, keepdims=True) + 1e-8
        sn = (s - m) / sd
        s1, s2 = sn[:, 0], sn[:, 1]
        if self.mode == "linear":
            return sn
        if self.mode == "gau1":
            h = np.column_stack([np.exp(-(s1 ** 2) / (2 * self.psi40[0] ** 2)),
                                 np.exp(-(s2 ** 2) / (2 * self.psi40[1] ** 2))])
            return (h - self.gau1_mean) / self.gau1_std
        if self.mode == "gau2":
            h = np.column_stack([np.exp(-(s1 ** 2) / (2 * self.psi60[0] ** 2)),
                                 np.exp(-(s2 ** 2) / (2 * self.psi60[1] ** 2))])
            return (h - self.gau2_mean) / self.gau2_std
        if self.mode == "cos1":
            return np.column_stack([np.cos(2 * np.pi * s1 / self.psi40[0]),
                                    np.cos(2 * np.pi * s2 / self.psi40[1])])
        if self.mode == "cos2":
            return np.column_stack([np.cos(2 * np.pi * s1 / self.psi60[0]),
                                    np.cos(2 * np.pi * s2 / self.psi60[1])])
        raise ValueError(self.mode)


def norm_xy(xy):
    lo = xy.min(0); rng = np.ptp(xy, 0); rng[rng == 0] = 1
    return (xy - lo) / rng


def binned_surface(coords01, vals, bins=BINS, sigma=SIGMA):
    xi = np.clip((coords01[:, 0] * bins).astype(int), 0, bins - 1)
    yi = np.clip((coords01[:, 1] * bins).astype(int), 0, bins - 1)
    num = np.zeros((bins, bins)); den = np.zeros((bins, bins))
    np.add.at(num, (yi, xi), vals); np.add.at(den, (yi, xi), 1.0)
    sn = gaussian_filter(num, sigma); sd = gaussian_filter(den, sigma)
    surf = sn / np.maximum(sd, 1e-12)
    surf[gaussian_filter(den, sigma) <= 0.035] = np.nan
    return surf


def main():
    configure()
    coords, frames = {}, {}
    for s in SECTIONS:
        o = pd.read_csv(OMG / f"{s}_SPOTlight_W_omega_with_xy.csv",
                        usecols=lambda c: c in {"spot", "x", "y", CELL_TYPE})
        frames[s] = o
        coords[s] = o[["x", "y"]].to_numpy(float)
    basis = BasisGenerator(MODE, np.vstack([coords[s] for s in SECTIONS]))

    rk = pd.read_csv(RANKED, low_memory=False)
    rk = rk[(rk.Mode == MODE) & (rk.Cell_Type_Name == CELL_TYPE) & (rk.Gene.isin(GENES))]
    beta = {r.Gene: (float(r.Learned_Beta_h1), float(r.Learned_Beta_h2)) for r in rk.itertuples()}
    qval = {r.Gene: float(r.BH_Q) for r in rk.itertuples()}
    missing = [g for g in GENES if g not in beta]
    if missing:
        raise RuntimeError(f"no fitted coefficients for {missing} in mode {MODE}")

    surfaces, rows = {}, []
    for g in GENES:
        b1, b2 = beta[g]
        surfaces[g] = {}
        for s in SECTIONS:
            h = basis.compute_h(coords[s])
            eta = b1 * h[:, 0] + b2 * h[:, 1]
            surf = binned_surface(norm_xy(coords[s]), eta)
            surfaces[g][s] = surf
            for y in range(surf.shape[0]):
                for x in range(surf.shape[1]):
                    rows.append({"gene": g, "cell_type": CELL_TYPE, "mode": MODE,
                                 "section": s, "x_bin": x, "y_bin": y,
                                 "fitted_spatial_effect": surf[y, x]})
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SOURCE_DIR / "fig4_pdac_fitted_spatial_effect.csv", index=False)

    def flat(g, s):
        return surfaces[g][s].ravel()
    cross = []
    for s in SECTIONS:
        for i in range(len(GENES)):
            for j in range(i + 1, len(GENES)):
                a, b = flat(GENES[i], s), flat(GENES[j], s)
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() > 12 and a[m].std() > 1e-12 and b[m].std() > 1e-12:
                    cross.append(abs(np.corrcoef(a[m], b[m])[0, 1]))
    print(f"mean |cross-gene map correlation| = {np.mean(cross):.3f} (proxy version was ~0.90)")

    vmax = max(np.nanmax(np.abs(surfaces[g][s])) for g in GENES for s in SECTIONS)
    n = len(GENES)
    fig = plt.figure(figsize=(7.2, 1.33 * n + 0.75))
    grid = fig.add_gridspec(n, len(SECTIONS) + 1,
                            width_ratios=[1.15] + [1] * len(SECTIONS), hspace=0.22, wspace=0.06)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#F7F7F7")
    labels = [s.replace("PDAC_", "") for s in SECTIONS]
    image = None
    for ri, g in enumerate(GENES):
        la = fig.add_subplot(grid[ri, 0]); la.axis("off")
        la.text(0.0, 0.70, g, ha="left", va="center", fontsize=7.4, fontweight="bold")
        la.text(0.0, 0.46, CELL_TYPE, ha="left", va="center", fontsize=5.8, color="#454545")
        b1, b2 = beta[g]
        la.text(0.0, 0.24, f"$\\beta_{{h1}}$={b1:.2f}, $\\beta_{{h2}}$={b2:.2f}",
                ha="left", va="center", fontsize=5.2, color="#6A6A6A")
        la.text(0.0, 0.05, f"BH q={qval[g]:.1e}", ha="left", va="center", fontsize=5.2, color="#6A6A6A")
        for ci, s in enumerate(SECTIONS, start=1):
            ax = fig.add_subplot(grid[ri, ci])
            image = ax.imshow(surfaces[g][s], origin="lower", interpolation="nearest",
                              cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#D8D8D8"); sp.set_linewidth(0.45)
            if ri == 0:
                ax.set_title(labels[ci - 1], pad=2, fontsize=6.1, fontweight="bold")
    cbar = fig.colorbar(image, ax=fig.axes, fraction=0.013, pad=0.015)
    cbar.set_label("Fitted ductal spatial effect", fontsize=6)
    cbar.ax.tick_params(labelsize=5.4, length=2)
    fig.suptitle("Fitted cell-type-specific spatial effects for PDAC ductal candidates",
                 x=0.51, y=1.01, fontsize=8.2, fontweight="bold")
    fig.text(0.51, 0.952, "Tumour B serial sections", ha="center", va="bottom",
             fontsize=6.0, color="#555555")
    fig.text(0.01, 0.987, "a", fontsize=9, fontweight="bold", va="top")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kw in {".svg": {}, ".pdf": {}, ".png": {"dpi": 450}, ".tiff": {"dpi": 600}}.items():
        fig.savefig(FIG_DIR / f"fig4_pdac_fitted_spatial_effect{suffix}", bbox_inches="tight", **kw)
        fig.savefig(LATEX_FIG_DIR / f"fig4_pdac_fitted_spatial_effect{suffix}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("wrote fig4_pdac_fitted_spatial_effect.* to", LATEX_FIG_DIR)


if __name__ == "__main__":
    main()
