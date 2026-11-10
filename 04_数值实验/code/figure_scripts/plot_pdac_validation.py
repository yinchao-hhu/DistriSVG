#!/usr/bin/env python3
"""Generate PDAC real-data validation tables and figures for AP-FedAdam."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "code" / "pdac_data_to_yc"
EXPR_DIR = DATA_DIR / "preprocessed_data"
OMEGA_DIR = DATA_DIR / "spotlight_out"
FIG_DIR = ROOT / "overleaf" / "figs"

MODES = ["linear", "gau1", "gau2", "cos1", "cos2"]
SLICES = ["PDAC_A_ST1", "PDAC_A_ST2", "PDAC_A_ST3", "PDAC_B_ST1", "PDAC_B_ST2", "PDAC_B_ST3"]

KNOWN_PDAC_MARKERS = {
    "PRSS1", "PRSS2", "PRSS3", "SPINK1", "CLDN2", "SERPINA1", "SERPING1",
    "KRT8", "KRT18", "KRT19", "EPCAM", "MUC1", "MUC4", "MUC5AC", "MSLN",
    "CEACAM6", "S100A6", "S100A10", "SLC29A1", "MMP7", "MMP11", "TFF1",
    "TFF2", "TFF3", "LGALS1", "LGALS3", "LGALS9", "SPP1", "NNMT",
}
KNOWN_STROMAL_IMMUNE_MARKERS = {
    "COL1A1", "COL1A2", "COL3A1", "COL6A1", "COL6A2", "COL6A3", "DCN",
    "LUM", "FN1", "ACTA2", "TAGLN", "POSTN", "VIM", "SPARC", "THY1",
    "A2M", "APOE", "C1QA", "C1QB", "C1QC", "CD68", "CD74", "CD163",
    "LYZ", "TYROBP", "FCER1G", "HLA-DRA", "HLA-DRB1", "FTL", "FTH1",
    "PECAM1", "VWF", "KDR", "RAMP2", "ENG", "HBB", "HBA1", "HBA2",
}
PREFERRED_SPATIAL_CANDIDATES = [
    ("PRSS2", "Ductal"),
    ("SLC29A1", "Cancer.clone.B"),
    ("APOE", "mDCs"),
    ("POSTN", "Fibroblasts"),
    ("S100A6", "Cancer.clone.B"),
    ("SPINK1", "Ductal"),
    ("FTL", "mDCs"),
    ("AEBP1", "Fibroblasts"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def read_gene_names() -> list[str]:
    first_expr = EXPR_DIR / "PDAC_A_ST1_processed_data_mincell5.csv"
    return pd.read_csv(first_expr, nrows=0).columns.tolist()[1:]


def read_cell_types() -> list[str]:
    first_omega = OMEGA_DIR / "PDAC_A_ST1_SPOTlight_W_omega_with_xy.csv"
    return pd.read_csv(first_omega, nrows=0).columns.tolist()[3:]


def normal_two_sided_p(z_values: np.ndarray) -> np.ndarray:
    z_abs = np.abs(np.asarray(z_values, dtype=float))
    out = np.empty_like(z_abs, dtype=float)
    sqrt2 = math.sqrt(2.0)
    flat_z = z_abs.ravel()
    flat_out = out.ravel()
    for i, z in enumerate(flat_z):
        flat_out[i] = math.erfc(float(z) / sqrt2) if np.isfinite(z) else 1.0
    return np.clip(out, 1e-15, 1.0 - 1e-15)


def cauchy_combine(p_matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(p_matrix, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    stat = np.mean(np.tan((0.5 - p) * math.pi), axis=1)
    out = 0.5 - np.arctan(stat) / math.pi
    return np.clip(out, 1e-15, 1.0 - 1e-15)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    p = np.where(np.isfinite(p), p, 1.0)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * n / np.arange(1, n + 1, dtype=float)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q = np.empty_like(q_ranked)
    q[order] = np.clip(q_ranked, 0.0, 1.0)
    return q


def marker_group(gene: str) -> str:
    if gene in KNOWN_PDAC_MARKERS:
        return "PDAC/epithelial marker list"
    if gene in KNOWN_STROMAL_IMMUNE_MARKERS:
        return "stromal/immune marker list"
    return ""


def load_mode_results(mode: str, gene_names: list[str], cell_types: list[str]) -> pd.DataFrame:
    path = DATA_DIR / f"PDAC_CTSV_Fed_6_G_4740_L1_2e-07_Round1000_h_BasisMode_{mode}_Inference_FIXED.csv"
    df = pd.read_csv(path)
    df["P_Value"] = normal_two_sided_p(df["Z_Score"].to_numpy())
    wide = df.pivot(index=["Gene_ID", "Cell_Type"], columns="Feature")
    wide.columns = [f"{value}_{feature}" for value, feature in wide.columns]
    wide = wide.reset_index()

    for col in ["P_Value_h1", "P_Value_h2", "Z_Score_h1", "Z_Score_h2", "Learned_Beta_h1", "Learned_Beta_h2"]:
        if col not in wide.columns:
            wide[col] = np.nan

    p_mat = wide[["P_Value_h1", "P_Value_h2"]].to_numpy(dtype=float)
    z_mat = wide[["Z_Score_h1", "Z_Score_h2"]].to_numpy(dtype=float)
    beta_mat = wide[["Learned_Beta_h1", "Learned_Beta_h2"]].to_numpy(dtype=float)

    wide["Mode"] = mode
    wide["Cauchy_P"] = cauchy_combine(p_mat)
    wide["BH_Q"] = bh_fdr(wide["Cauchy_P"].to_numpy())
    wide["Max_Abs_Z"] = np.nanmax(np.abs(z_mat), axis=1)
    wide["Sum_Abs_Beta"] = np.nansum(np.abs(beta_mat), axis=1)
    dominant_idx = np.nanargmax(np.abs(z_mat), axis=1)
    wide["Dominant_Feature"] = np.where(dominant_idx == 0, "h1", "h2")
    wide["Dominant_Z"] = z_mat[np.arange(z_mat.shape[0]), dominant_idx]
    wide["Dominant_Beta"] = beta_mat[np.arange(beta_mat.shape[0]), dominant_idx]
    wide["Gene"] = [gene_names[int(i)] for i in wide["Gene_ID"]]
    wide["Cell_Type_Name"] = [cell_types[int(i)] for i in wide["Cell_Type"]]
    wide["Marker_Group"] = [marker_group(g) for g in wide["Gene"]]
    wide = wide.sort_values(["BH_Q", "Cauchy_P", "Max_Abs_Z"], ascending=[True, True, False])
    wide["Rank_In_Mode"] = np.arange(1, len(wide) + 1)

    cols = [
        "Mode", "Rank_In_Mode", "Gene_ID", "Gene", "Cell_Type", "Cell_Type_Name",
        "Cauchy_P", "BH_Q", "Max_Abs_Z", "Dominant_Feature", "Dominant_Z",
        "Dominant_Beta", "Sum_Abs_Beta", "Learned_Beta_h1", "Learned_Beta_h2",
        "Z_Score_h1", "Z_Score_h2", "P_Value_h1", "P_Value_h2", "Marker_Group",
    ]
    return wide[cols]


def build_consensus(all_ranked: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    frames = []
    for (gene, cell), sub in all_ranked.groupby(["Gene", "Cell_Type_Name"], sort=False):
        p_by_mode = sub.set_index("Mode")["Cauchy_P"].reindex(MODES).to_numpy(dtype=float)
        mode_acat_p = cauchy_combine(p_by_mode.reshape(1, -1))[0]
        frames.append(
            {
                "Gene": gene,
                "Gene_ID": int(sub["Gene_ID"].iloc[0]),
                "Cell_Type_Name": cell,
                "Cell_Type": int(sub["Cell_Type"].iloc[0]),
                "Mode_ACAT_P": mode_acat_p,
                "Min_Q": float(sub["BH_Q"].min()),
                "Min_P": float(sub["Cauchy_P"].min()),
                "Best_Mode": str(sub.loc[sub["BH_Q"].idxmin(), "Mode"]),
                "Best_Dominant_Feature": str(sub.loc[sub["BH_Q"].idxmin(), "Dominant_Feature"]),
                "Best_Dominant_Beta": float(sub.loc[sub["BH_Q"].idxmin(), "Dominant_Beta"]),
                "Best_Dominant_Z": float(sub.loc[sub["BH_Q"].idxmin(), "Dominant_Z"]),
                "Max_Abs_Z": float(sub["Max_Abs_Z"].max()),
                "Median_Rank": float(sub["Rank_In_Mode"].median()),
                "Modes_Q_lt_0_05": int((sub["BH_Q"] < 0.05).sum()),
                "Modes_Q_lt_0_10": int((sub["BH_Q"] < 0.10).sum()),
                f"Modes_Top_{top_n}": int((sub["Rank_In_Mode"] <= top_n).sum()),
                "Significant_Modes": ";".join(sub.loc[sub["BH_Q"] < 0.05, "Mode"].tolist()),
                f"Top_{top_n}_Modes": ";".join(sub.loc[sub["Rank_In_Mode"] <= top_n, "Mode"].tolist()),
                "Marker_Group": first_marker_group(sub["Marker_Group"]),
            }
        )
    consensus = pd.DataFrame(frames)
    consensus["Mode_ACAT_Q"] = bh_fdr(consensus["Mode_ACAT_P"].to_numpy())
    consensus["Joint_Basis_P"] = consensus["Mode_ACAT_P"]
    consensus["Joint_Basis_Q"] = consensus["Mode_ACAT_Q"]
    consensus = consensus.sort_values(
        ["Joint_Basis_Q", "Joint_Basis_P", "Max_Abs_Z", "Median_Rank"],
        ascending=[True, True, False, True],
    )
    consensus["Joint_Basis_Rank"] = np.arange(1, len(consensus) + 1)
    consensus["Consensus_Rank"] = consensus["Joint_Basis_Rank"]
    order = [
        "Joint_Basis_Rank", "Consensus_Rank", "Gene_ID", "Gene", "Cell_Type", "Cell_Type_Name",
        "Joint_Basis_P", "Joint_Basis_Q", "Mode_ACAT_P", "Mode_ACAT_Q", "Min_Q", "Min_P", "Best_Mode",
        "Best_Dominant_Feature", "Best_Dominant_Beta", "Best_Dominant_Z",
        "Max_Abs_Z", "Median_Rank", "Modes_Q_lt_0_05", "Modes_Q_lt_0_10",
        f"Modes_Top_{top_n}", "Significant_Modes", f"Top_{top_n}_Modes",
        "Marker_Group",
    ]
    return consensus[order]


def first_marker_group(values: pd.Series) -> str:
    markers = [str(x) for x in values.tolist() if isinstance(x, str) and x.strip()]
    return markers[0] if markers else ""


def hypergeom_sf(total: int, marked: int, draws: int, observed: int) -> float:
    """Return P[X >= observed] for X ~ Hypergeometric(total, marked, draws)."""
    if observed <= 0:
        return 1.0
    upper = min(marked, draws)
    if observed > upper:
        return 0.0

    def logcomb(n: int, k: int) -> float:
        if k < 0 or k > n:
            return -math.inf
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    denominator = logcomb(total, draws)
    log_terms = [
        logcomb(marked, x) + logcomb(total - marked, draws - x) - denominator
        for x in range(observed, upper + 1)
    ]
    max_log = max(log_terms)
    return float(sum(math.exp(term - max_log) for term in log_terms) * math.exp(max_log))


def marker_annotation_tables(consensus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotated = consensus.copy()
    annotated["Marker_Category"] = annotated["Marker_Group"].fillna("").replace("", "unannotated")
    annotated["Marker_Annotated"] = annotated["Marker_Category"] != "unannotated"

    total = len(annotated)
    rows = []
    group_specs = [
        ("Any curated marker", annotated["Marker_Annotated"]),
        ("PDAC/epithelial marker list", annotated["Marker_Category"] == "PDAC/epithelial marker list"),
        ("stromal/immune marker list", annotated["Marker_Category"] == "stromal/immune marker list"),
    ]
    for cutoff in [20, 50, 100, 200]:
        top = annotated.head(cutoff)
        for group_name, mask in group_specs:
            background_marked = int(mask.sum())
            if group_name == "Any curated marker":
                observed = int(top["Marker_Annotated"].sum())
            else:
                observed = int((top["Marker_Category"] == group_name).sum())
            rows.append(
                {
                    "Top_N": cutoff,
                    "Marker_Group": group_name,
                    "Observed": observed,
                    "Draws": cutoff,
                    "Background_Marked": background_marked,
                    "Background_Total": total,
                    "Observed_Fraction": observed / cutoff,
                    "Background_Fraction": background_marked / total,
                    "Hypergeom_P_Greater_Equal": hypergeom_sf(total, background_marked, cutoff, observed),
                }
            )
    summary = pd.DataFrame(rows)
    top20_cols = [
        "Joint_Basis_Rank", "Gene", "Cell_Type_Name", "Joint_Basis_Q", "Best_Dominant_Z",
        "Modes_Q_lt_0_05", "Significant_Modes", "Marker_Category",
        "Marker_Annotated",
    ]
    return annotated.head(20)[top20_cols], summary


def color_scale(value: float, vmax: float) -> tuple[int, int, int]:
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    if not np.isfinite(value) or vmax <= 0:
        return (245, 245, 245)
    t = min(max(value / vmax, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(math.floor(t)), len(stops) - 2)
    frac = t - i
    c0, c1 = stops[i], stops[i + 1]
    return tuple(int(round(c0[j] + frac * (c1[j] - c0[j]))) for j in range(3))


def draw_top_heatmap(all_ranked: pd.DataFrame, consensus: pd.DataFrame, out_path: Path) -> None:
    rows = consensus.query("Joint_Basis_Q < 0.05").head(20)
    if rows.empty:
        rows = consensus.head(20)
    rows = rows[["Gene", "Cell_Type_Name"]].drop_duplicates()

    lookup = all_ranked.set_index(["Gene", "Cell_Type_Name", "Mode"])["BH_Q"]
    matrix = []
    labels = []
    for _, row in rows.iterrows():
        gene = row["Gene"]
        cell = row["Cell_Type_Name"]
        q_vals = [float(lookup.get((gene, cell, mode), 1.0)) for mode in MODES]
        matrix.append([-math.log10(max(q, 1e-15)) for q in q_vals])
        labels.append(f"{gene} / {cell}")

    if not matrix:
        return
    mat = np.asarray(matrix, dtype=float)
    cap = max(2.0, min(15.0, float(np.nanpercentile(mat, 95))))

    left = 270
    top = 90
    cell_w = 105
    cell_h = 28
    width = left + cell_w * len(MODES) + 210
    height = top + cell_h * len(labels) + 110
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    f_title = font(24, bold=True)
    f_axis = font(16, bold=True)
    f_text = font(13)
    f_small = font(11)

    draw.text((30, 24), "Jointly selected PDAC AP-FedAdam candidates across spatial bases", fill=(20, 20, 20), font=f_title)
    draw.text((30, 54), "Rows are ranked by one all-basis ACAT test; cells show within-basis -log10(BH q).", fill=(70, 70, 70), font=f_small)

    for j, mode in enumerate(MODES):
        x = left + j * cell_w + cell_w // 2
        draw.text((x - 24, top - 32), mode, fill=(25, 25, 25), font=f_axis)

    for i, label in enumerate(labels):
        y = top + i * cell_h
        draw.text((30, y + 6), label, fill=(25, 25, 25), font=f_text)
        for j in range(len(MODES)):
            x = left + j * cell_w
            val = mat[i, j]
            draw.rectangle([x, y, x + cell_w - 2, y + cell_h - 2], fill=color_scale(val, cap))
            text = f"{val:.1f}" if val < 15 else ">15"
            draw.text((x + 35, y + 6), text, fill=(255, 255, 255) if val < cap * 0.55 else (30, 30, 30), font=f_small)

    bar_x = left + cell_w * len(MODES) + 45
    bar_y = top
    bar_h = min(280, cell_h * len(labels))
    for k in range(bar_h):
        val = cap * (1.0 - k / max(bar_h - 1, 1))
        draw.line([(bar_x, bar_y + k), (bar_x + 18, bar_y + k)], fill=color_scale(val, cap))
    draw.rectangle([bar_x, bar_y, bar_x + 18, bar_y + bar_h], outline=(80, 80, 80))
    draw.text((bar_x + 28, bar_y - 6), f"{cap:.1f}", fill=(30, 30, 30), font=f_small)
    draw.text((bar_x + 28, bar_y + bar_h - 8), "0", fill=(30, 30, 30), font=f_small)
    draw.text((bar_x + 28, bar_y + bar_h + 12), "-log10(q)", fill=(30, 30, 30), font=f_small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, dpi=(300, 300))


def choose_spatial_candidates(consensus: pd.DataFrame, n: int = 4) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    seen_cells: set[str] = set()
    for gene, cell in PREFERRED_SPATIAL_CANDIDATES:
        hit = consensus[(consensus["Gene"] == gene) & (consensus["Cell_Type_Name"] == cell)]
        if hit.empty:
            continue
        if float(hit["Joint_Basis_Q"].iloc[0]) > 0.05:
            continue
        if cell in seen_cells and len(seen_cells) < n:
            continue
        selected.append((gene, cell))
        seen_cells.add(cell)
        if len(selected) == n:
            return selected
    for _, row in consensus.iterrows():
        item = (str(row["Gene"]), str(row["Cell_Type_Name"]))
        if item in selected:
            continue
        if row["Cell_Type_Name"] in seen_cells and len(seen_cells) < n:
            continue
        selected.append(item)
        seen_cells.add(item[1])
        if len(selected) == n:
            break
    return selected


def detection_count_summary(all_ranked: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in MODES:
        subset = all_ranked[(all_ranked["Mode"] == mode) & (all_ranked["BH_Q"] < 0.05)]
        rows.append(
            {
                "analysis": mode,
                "decision_rule": "within-basis ACAT followed by BH q<0.05",
                "selected_pairs": int(len(subset)),
                "selected_genes": int(subset["Gene"].nunique()),
            }
        )
    joint = consensus[consensus["Joint_Basis_Q"] < 0.05]
    rows.append(
        {
            "analysis": "joint_all_bases",
            "decision_rule": "ACAT across all five basis-level tests followed by one BH q<0.05",
            "selected_pairs": int(len(joint)),
            "selected_genes": int(joint["Gene"].nunique()),
        }
    )
    return pd.DataFrame(rows)


def load_spatial_values(gene: str, cell_type: str) -> dict[str, pd.DataFrame]:
    out = {}
    for sid in SLICES:
        expr_path = EXPR_DIR / f"{sid}_processed_data_mincell5.csv"
        omega_path = OMEGA_DIR / f"{sid}_SPOTlight_W_omega_with_xy.csv"
        expr = pd.read_csv(expr_path, usecols=["Spot_ID", gene])
        omega = pd.read_csv(omega_path, usecols=["spot", "x", "y", cell_type])
        df = expr.merge(omega, left_on="Spot_ID", right_on="spot", how="inner", validate="one_to_one")
        df["component_proxy"] = np.log1p(df[gene].astype(float)) * df[cell_type].astype(float)
        out[sid] = df[["Spot_ID", "x", "y", gene, cell_type, "component_proxy"]].copy()
    return out


def draw_spatial_panel(values_by_candidate: dict[tuple[str, str], dict[str, pd.DataFrame]], out_path: Path) -> pd.DataFrame:
    panel_w = 190
    panel_h = 170
    left_label_w = 185
    top_h = 86
    gap = 20
    width = left_label_w + len(SLICES) * panel_w + 90
    height = top_h + len(values_by_candidate) * panel_h + 90
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    f_title = font(23, bold=True)
    f_header = font(14, bold=True)
    f_text = font(13)
    f_small = font(10)

    draw.text((30, 18), "Spatial component proxy for selected PDAC CTSV candidates", fill=(20, 20, 20), font=f_title)
    draw.text((30, 48), "Rows show representative gene-cell-type pairs; columns show the six PDAC sections.", fill=(70, 70, 70), font=f_small)
    for j, sid in enumerate(SLICES):
        x = left_label_w + j * panel_w + 28
        draw.text((x, top_h - 23), sid.replace("PDAC_", ""), fill=(30, 30, 30), font=f_header)

    summary_rows = []
    for r, ((gene, cell), by_slice) in enumerate(values_by_candidate.items()):
        row_y = top_h + r * panel_h
        draw.text((28, row_y + 48), gene, fill=(20, 20, 20), font=f_header)
        draw.text((28, row_y + 70), cell, fill=(65, 65, 65), font=f_text)

        all_vals = np.concatenate([df["component_proxy"].to_numpy(dtype=float) for df in by_slice.values()])
        vmax = float(np.nanpercentile(all_vals, 98)) if all_vals.size else 1.0
        vmax = max(vmax, 1e-8)

        for j, sid in enumerate(SLICES):
            df = by_slice[sid]
            x0 = left_label_w + j * panel_w + 10
            y0 = row_y + 18
            w = panel_w - 30
            h = panel_h - 38
            draw.rectangle([x0, y0, x0 + w, y0 + h], outline=(218, 218, 218), fill=(250, 250, 250))

            xs = df["x"].to_numpy(dtype=float)
            ys = df["y"].to_numpy(dtype=float)
            vals = df["component_proxy"].to_numpy(dtype=float)
            xmin, xmax = float(xs.min()), float(xs.max())
            ymin, ymax = float(ys.min()), float(ys.max())
            sx = (w - 12) / max(xmax - xmin, 1.0)
            sy = (h - 12) / max(ymax - ymin, 1.0)
            box = max(3, min(7, int(min(sx, sy) * 0.72)))
            for x, y, val in zip(xs, ys, vals):
                px = x0 + 6 + int(round((x - xmin) * sx))
                py = y0 + h - 6 - int(round((y - ymin) * sy))
                color = color_scale(float(val), vmax)
                draw.rectangle([px - box, py - box, px + box, py + box], fill=color)

            summary_rows.append(
                {
                    "Gene": gene,
                    "Cell_Type_Name": cell,
                    "Slice": sid,
                    "Mean_Component_Proxy": float(np.mean(vals)),
                    "Max_Component_Proxy": float(np.max(vals)),
                    "P98_Component_Proxy": vmax,
                    "Nonzero_Spots": int(np.sum(vals > 0)),
                    "Total_Spots": int(len(vals)),
                }
            )

        bar_x = left_label_w + len(SLICES) * panel_w + 24
        bar_y = row_y + 32
        bar_h = 92
        for k in range(bar_h):
            val = vmax * (1.0 - k / max(bar_h - 1, 1))
            draw.line([(bar_x, bar_y + k), (bar_x + 15, bar_y + k)], fill=color_scale(val, vmax))
        draw.rectangle([bar_x, bar_y, bar_x + 15, bar_y + bar_h], outline=(90, 90, 90))
        draw.text((bar_x + 21, bar_y - 4), f"{vmax:.2f}", fill=(50, 50, 50), font=f_small)
        draw.text((bar_x + 21, bar_y + bar_h - 7), "0", fill=(50, 50, 50), font=f_small)

    draw.text((30, height - 32), "Proxy = log1p(gene count) x estimated cell-type proportion; shown for visualization, not as a fitted coefficient.", fill=(75, 75, 75), font=f_small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, dpi=(300, 300))
    return pd.DataFrame(summary_rows)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    gene_names = read_gene_names()
    cell_types = read_cell_types()

    ranked_by_mode = [load_mode_results(mode, gene_names, cell_types) for mode in MODES]
    all_ranked = pd.concat(ranked_by_mode, ignore_index=True)
    all_ranked = all_ranked.sort_values(["Mode", "Rank_In_Mode"])
    all_ranked.to_csv(FIG_DIR / "pdac_ctsv_all_modes_ranked.csv", index=False)

    top_by_cell = (
        all_ranked.sort_values(["Mode", "Cell_Type_Name", "BH_Q", "Cauchy_P"])
        .groupby(["Mode", "Cell_Type_Name"], as_index=False, group_keys=False)
        .head(20)
    )
    top_by_cell.to_csv(FIG_DIR / "pdac_top_ctsv_by_celltype.csv", index=False)

    consensus = build_consensus(all_ranked, top_n=200)
    consensus.to_csv(FIG_DIR / "pdac_basis_consensus_candidates.csv", index=False)
    detection_count_summary(all_ranked, consensus).to_csv(
        FIG_DIR / "pdac_joint_basis_detection_summary.csv", index=False
    )

    top20_markers, marker_summary = marker_annotation_tables(consensus)
    top20_markers.to_csv(FIG_DIR / "pdac_top20_marker_annotations.csv", index=False)
    marker_summary.to_csv(FIG_DIR / "pdac_marker_annotation_summary.csv", index=False)

    draw_top_heatmap(all_ranked, consensus, FIG_DIR / "fig4a_pdac_ctsv_basis_consensus.png")

    candidates = choose_spatial_candidates(consensus, n=4)
    values = {candidate: load_spatial_values(*candidate) for candidate in candidates}
    spatial_summary = draw_spatial_panel(values, FIG_DIR / "fig4b_pdac_spatial_component_maps.png")
    spatial_summary.to_csv(FIG_DIR / "pdac_spatial_component_proxy_summary.csv", index=False)

    selected = consensus[
        consensus.apply(lambda r: (r["Gene"], r["Cell_Type_Name"]) in set(candidates), axis=1)
    ].sort_values("Joint_Basis_Rank")
    selected.to_csv(FIG_DIR / "pdac_selected_spatial_candidates.csv", index=False)

    print("Wrote:")
    for name in [
        "pdac_ctsv_all_modes_ranked.csv",
        "pdac_top_ctsv_by_celltype.csv",
        "pdac_basis_consensus_candidates.csv",
        "pdac_joint_basis_detection_summary.csv",
        "pdac_top20_marker_annotations.csv",
        "pdac_marker_annotation_summary.csv",
        "pdac_selected_spatial_candidates.csv",
        "pdac_spatial_component_proxy_summary.csv",
        "fig4a_pdac_ctsv_basis_consensus.png",
        "fig4b_pdac_spatial_component_maps.png",
    ]:
        print(f"  {FIG_DIR / name}")
    print("Selected spatial candidates:", ", ".join(f"{g}/{c}" for g, c in candidates))


if __name__ == "__main__":
    main()
