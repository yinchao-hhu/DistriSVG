"""Experiment: find Ductal PDAC candidates with highest within-Tumor-B cross-section
correlation, and test how smoothing bandwidth affects the correlation.  Read-only."""
from pathlib import Path
import numpy as np
import pandas as pd


def gaussian_filter(a, sigma, mode="nearest"):
    """Separable 2D Gaussian blur with edge ('nearest') padding; numpy-only."""
    radius = max(1, int(4.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()
    out = a.astype(float)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis, padded)
    return out

ROOT = Path(__file__).resolve().parents[3] / "04_数值实验"
EXPR = ROOT / "code" / "pdac_data_to_yc" / "preprocessed_data"
OMG = ROOT / "code" / "pdac_data_to_yc" / "spotlight_out"
B = ("PDAC_B_ST1", "PDAC_B_ST2", "PDAC_B_ST3")
EXCL_PRE = ("MT-", "MTRNR", "RPL", "RPS")
EXCL = {"ACTB", "GAPDH", "EEF1A1", "EEF1D", "EEF2", "TMSB4X", "TMSB10", "TMSB15"}

cand = pd.read_csv(ROOT / "overleaf" / "figs" / "pdac_basis_consensus_candidates.csv", low_memory=False)
cand = cand[(cand["Min_Q"] < 0.05) & (cand["Modes_Q_lt_0_05"] >= 3)].head(1000)
duct = cand[cand["Cell_Type_Name"] == "Ductal"].copy()
duct = duct[~duct["Gene"].astype(str).isin(EXCL)]
duct = duct[~duct["Gene"].astype(str).str.startswith(EXCL_PRE)]
genes = duct["Gene"].astype(str).unique().tolist()

frames = {}
for s in B:
    e = pd.read_csv(EXPR / f"{s}_processed_data_mincell5.csv",
                    usecols=lambda c: c == "Spot_ID" or c in set(genes))
    o = pd.read_csv(OMG / f"{s}_SPOTlight_W_omega_with_xy.csv",
                    usecols=lambda c: c in {"spot", "x", "y", "Ductal"})
    frames[s] = e.merge(o, left_on="Spot_ID", right_on="spot", how="inner")

def norm_xy(f):
    xy = f[["x", "y"]].to_numpy(float)
    lo = xy.min(0); rng = np.ptp(xy, 0); rng[rng == 0] = 1
    return (xy - lo) / rng

def rankmap(coords, vals, bins, sigma):
    xi = np.clip((coords[:, 0] * bins).astype(int), 0, bins - 1)
    yi = np.clip((coords[:, 1] * bins).astype(int), 0, bins - 1)
    num = np.zeros((bins, bins)); den = np.zeros((bins, bins))
    np.add.at(num, (yi, xi), vals); np.add.at(den, (yi, xi), 1.0)
    sn = gaussian_filter(num, sigma, mode="nearest"); sd = gaussian_filter(den, sigma, mode="nearest")
    surf = sn / np.maximum(sd, 1e-12)
    surf[gaussian_filter(den, sigma, mode="nearest") <= 0.035] = np.nan
    flat = surf.ravel(); m = np.isfinite(flat); r = np.full_like(flat, np.nan)
    order = flat[m].argsort(); ranks = np.empty(order.size); ranks[order] = np.arange(order.size)
    r[m] = ranks / max(order.size - 1, 1); return r.reshape(surf.shape)

def meanpair(maps):
    v = []
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            a, b = maps[i], maps[j]; mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() >= 12 and a[mask].std() > 1e-10 and b[mask].std() > 1e-10:
                v.append(np.corrcoef(a[mask], b[mask])[0, 1])
    return float(np.mean(v)) if v else float("nan")

for bins, sigma in [(18, 1.05), (16, 1.4), (14, 1.6)]:
    rows = []
    for g in genes:
        maps = [rankmap(norm_xy(frames[s]),
                        np.log1p(frames[s][g].to_numpy(float)) * frames[s]["Ductal"].to_numpy(float),
                        bins, sigma) for s in B]
        rows.append((g, meanpair(maps)))
    df = pd.DataFrame(rows, columns=["gene", "r"]).sort_values("r", ascending=False)
    cur = df[df.gene.isin(["KRT18", "KRT19", "TIMP1", "SERPINA1"])]
    print(f"\n=== bins={bins} sigma={sigma} ===  (n ductal genes={len(df)})")
    print("TOP8:", ", ".join(f"{r.gene}:{r.r:.2f}" for r in df.head(8).itertuples()))
    print("CURRENT4:", ", ".join(f"{r.gene}:{r.r:.2f}" for r in cur.itertuples()))
