"""Standalone regeneration of Figure 6 (PDAC Tumor B recurrent Ductal spatial maps)
with a wider smoothing bandwidth (bins=16, sigma=1.4) to raise same-row cross-section
concordance, keeping the four canonical Ductal markers.  scanpy/scipy-free so it runs
in the sandbox; matches make_cross_section_spatial_maps.py style exactly.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "04_数值实验"
FIG_DIR = ROOT / "overleaf" / "figs" / "cross_section_spatial_maps"
SOURCE_DIR = FIG_DIR / "source_data"
EXPR = ROOT / "code" / "pdac_data_to_yc" / "preprocessed_data"
OMG = ROOT / "code" / "pdac_data_to_yc" / "spotlight_out"
B = ("PDAC_B_ST1", "PDAC_B_ST2", "PDAC_B_ST3")
GENES = ("KRT18", "KRT19", "TIMP1", "SERPINA1")
BINS, SIGMA = 16, 1.4


def gaussian_filter(a, sigma):
    radius = max(1, int(4.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2)); k /= k.sum()
    out = a.astype(float)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]; pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis, padded)
    return out


def configure():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6.5,
        "axes.titlesize": 6.5, "axes.linewidth": 0.5,
    })


def norm_xy(f):
    xy = f[["x", "y"]].to_numpy(float)
    lo = xy.min(0); rng = np.ptp(xy, 0); rng[rng == 0] = 1
    return (xy - lo) / rng


def rank_normalize(surface):
    flat = surface.ravel(); m = np.isfinite(flat); r = np.full_like(flat, np.nan)
    order = flat[m].argsort(); ranks = np.empty(order.size); ranks[order] = np.arange(order.size)
    r[m] = ranks / max(order.size - 1, 1)
    return r.reshape(surface.shape)


def smoothed_rank_map(coords, vals, bins=BINS, sigma=SIGMA):
    xi = np.clip((coords[:, 0] * bins).astype(int), 0, bins - 1)
    yi = np.clip((coords[:, 1] * bins).astype(int), 0, bins - 1)
    num = np.zeros((bins, bins)); den = np.zeros((bins, bins))
    np.add.at(num, (yi, xi), vals); np.add.at(den, (yi, xi), 1.0)
    sn = gaussian_filter(num, sigma); sd = gaussian_filter(den, sigma)
    surf = sn / np.maximum(sd, 1e-12)
    surf[gaussian_filter(den, sigma) <= 0.035] = np.nan
    return rank_normalize(surf)


def mean_pairwise(maps):
    v = []
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            a, b = maps[i], maps[j]; mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() >= 12 and a[mask].std() > 1e-10 and b[mask].std() > 1e-10:
                v.append(float(np.corrcoef(a[mask], b[mask])[0, 1]))
    return float(np.mean(v)) if v else float("nan")


def main():
    configure()
    frames = {}
    for s in B:
        e = pd.read_csv(EXPR / f"{s}_processed_data_mincell5.csv",
                        usecols=lambda c: c == "Spot_ID" or c in set(GENES))
        o = pd.read_csv(OMG / f"{s}_SPOTlight_W_omega_with_xy.csv",
                        usecols=lambda c: c in {"spot", "x", "y", "Ductal"})
        frames[s] = e.merge(o, left_on="Spot_ID", right_on="spot", how="inner")

    selected = []
    for g in GENES:
        maps = {s: smoothed_rank_map(norm_xy(frames[s]),
                np.log1p(frames[s][g].to_numpy(float)) * frames[s]["Ductal"].to_numpy(float))
                for s in B}
        selected.append({"gene": g, "cell_type": "Ductal",
                         "within_b": mean_pairwise(list(maps.values())), "maps": maps})

    rows = []
    for it in selected:
        for s, surf in it["maps"].items():
            for y in range(surf.shape[0]):
                for x in range(surf.shape[1]):
                    rows.append({"gene": it["gene"], "cell_type": "Ductal", "section": s,
                                 "x_bin": x, "y_bin": y, "smoothed_proxy_percentile": surf[y, x]})
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SOURCE_DIR / "fig5_pdac_spatial_surfaces.csv", index=False)

    n = len(selected)
    fig = plt.figure(figsize=(7.2, 1.33 * n + 0.75))
    grid = fig.add_gridspec(n, len(B) + 1, width_ratios=[1.15] + [1] * len(B), hspace=0.22, wspace=0.06)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#F7F7F7")
    image = None
    labels = [s.replace("PDAC_", "") for s in B]
    for ri, it in enumerate(selected):
        la = fig.add_subplot(grid[ri, 0]); la.axis("off")
        la.text(0.0, 0.67, it["gene"], ha="left", va="center", fontsize=7.4, fontweight="bold")
        la.text(0.0, 0.43, it["cell_type"], ha="left", va="center", fontsize=5.8, color="#454545")
        la.text(0.0, 0.19, f"Tumor B r={it['within_b']:.2f}", ha="left", va="center", fontsize=5.2, color="#6A6A6A")
        for ci, s in enumerate(B, start=1):
            ax = fig.add_subplot(grid[ri, ci])
            image = ax.imshow(it["maps"][s], origin="lower", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#D8D8D8"); sp.set_linewidth(0.45)
            if ri == 0:
                ax.set_title(labels[ci - 1], pad=2, fontsize=6.1, fontweight="bold")
    cbar = fig.colorbar(image, ax=fig.axes, fraction=0.013, pad=0.015)
    cbar.set_label("Smoothed proxy percentile", fontsize=6); cbar.ax.tick_params(labelsize=5.4, length=2)
    fig.suptitle("PDAC Tumor B serial sections: recurrent Ductal spatial patterns", x=0.51, y=1.01, fontsize=8.2, fontweight="bold")
    fig.text(0.51, 0.952, "Tumor B", ha="center", va="bottom", fontsize=6.0, color="#555555")
    fig.text(0.01, 0.987, "a", fontsize=9, fontweight="bold", va="top")
    for suffix, kw in {".svg": {}, ".pdf": {}, ".png": {"dpi": 450}, ".tiff": {"dpi": 600}}.items():
        fig.savefig(FIG_DIR / f"fig5_pdac_spatial_coherence{suffix}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("within Tumor B r:", {it["gene"]: round(it["within_b"], 3) for it in selected})
    print("wrote fig6 to", FIG_DIR)


if __name__ == "__main__":
    main()
