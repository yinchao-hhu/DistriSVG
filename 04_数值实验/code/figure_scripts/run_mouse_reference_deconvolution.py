#!/usr/bin/env python3
"""Reference-based deconvolution for the mouse brain Visium sections.

This script uses the GEO GSE71585 Allen/Tasic mouse visual cortex
single-cell reference matrix plus cell-type labels as a matched mouse brain
reference. It produces spot-level cell-type weights that can be used as omega
inputs for the CTSV/AP-FedAdam real-data analysis.

The local workspace currently has no R/Rscript installation, so the runnable
path here is a lightweight NNLS reference deconvolution. RCTD/SPOTlight remain
the recommended R implementations when an R environment is available.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import sparse
from scipy.optimize import nnls

from run_mouse_brain_analysis import (
    SECTIONS,
    BasisGenerator,
    CTSVModel,
    clone_state,
    dense_selected_matrix,
    run_federated,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "reference" / "mouse_brain"
VISIUM_DIR = ROOT / "data"
OUT_DIR = ROOT / "overleaf" / "figs" / "mouse_reference_deconv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTS_PATH = REFERENCE_DIR / "GSE71585_RefSeq_counts.csv.gz"
LABELS_PATH = REFERENCE_DIR / "GSE71585_Clustering_Results.csv.gz"


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def short_section_name(section: str) -> str:
    return (
        section.replace("V1_Mouse_Brain_Sagittal_", "")
        .replace("_Section_2", "2")
        .replace("_", "")
    )


def load_visium_section(section: str, source: str):
    """Load a Visium section from the bundled input or Squidpy's public cache."""
    if source == "local":
        section_dir = VISIUM_DIR / section
        if not section_dir.exists():
            raise FileNotFoundError(
                f"Missing local Visium directory: {section_dir}. "
                "Use --visium-source squidpy to use Squidpy's public-data cache."
            )
        return sc.read_visium(
            section_dir,
            count_file="filtered_feature_bc_matrix.h5",
            load_images=False,
        )
    return sq.datasets.visium(section)


def dense_array(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def normalize_rows(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, None)
    sums = x.sum(axis=1, keepdims=True)
    out = x / (sums + eps)
    zero = sums.ravel() <= eps
    if np.any(zero):
        out[zero, :] = 1.0 / x.shape[1]
    return out


def entropy_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def library_normalize_columns(counts: pd.DataFrame) -> pd.DataFrame:
    lib = counts.sum(axis=0).astype(float)
    good = lib > 0
    counts = counts.loc[:, good]
    lib = lib.loc[good]
    scale = float(np.median(lib))
    return counts.divide(lib, axis=1).multiply(scale)


def load_reference_profiles(
    counts_path: Path,
    labels_path: Path,
    label_col: str,
    min_cells_per_type: int,
    exclude_unclassified: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not counts_path.exists():
        raise FileNotFoundError(f"Missing reference counts file: {counts_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing reference labels file: {labels_path}")

    labels = pd.read_csv(labels_path)
    if label_col not in labels.columns:
        raise ValueError(f"Label column {label_col!r} not found. Available: {labels.columns.tolist()}")
    labels = labels[labels["pass_qc_checks"].astype(str).str.upper() == "Y"].copy()

    counts = pd.read_csv(counts_path, index_col=0)
    counts.index = counts.index.astype(str)
    if counts.index.has_duplicates:
        counts = counts.groupby(counts.index).sum()

    common_cells = [c for c in counts.columns if c in set(labels["sample_title"])]
    labels = labels.set_index("sample_title").loc[common_cells].copy()
    counts = counts.loc[:, common_cells]

    label_values = labels[label_col].fillna("Unclassified").astype(str)
    if exclude_unclassified:
        mask = ~label_values.str.lower().str.contains("unclassified")
        label_values = label_values.loc[mask]
        counts = counts.loc[:, label_values.index]

    label_counts = label_values.value_counts()
    keep_labels = label_counts[label_counts >= min_cells_per_type].index.tolist()
    keep_cells = label_values.index[label_values.isin(keep_labels)].tolist()
    label_values = label_values.loc[keep_cells]
    counts = counts.loc[:, keep_cells].astype(np.float32)

    norm = np.log1p(library_normalize_columns(counts))
    label_values = label_values.loc[norm.columns]
    cell_types = sorted(label_values.unique().tolist())

    profiles = {}
    for ct in cell_types:
        profiles[ct] = norm.loc[:, label_values == ct].mean(axis=1)
    profiles_df = pd.DataFrame(profiles, index=norm.index).astype(np.float32)

    ref_summary = (
        label_values.value_counts()
        .rename_axis("cell_type")
        .reset_index(name="n_cells")
        .sort_values(["n_cells", "cell_type"], ascending=[False, True])
    )
    ref_summary["label_col"] = label_col

    manifest = {
        "counts_path": str(counts_path),
        "labels_path": str(labels_path),
        "label_col": label_col,
        "min_cells_per_type": min_cells_per_type,
        "exclude_unclassified": exclude_unclassified,
        "n_reference_cells_used": int(len(label_values)),
        "n_reference_genes": int(profiles_df.shape[0]),
        "cell_types": cell_types,
    }
    return profiles_df, ref_summary, manifest


def log1p_gene_variance_for_common(adata, common_genes: list[str]) -> pd.Series:
    indices = adata.var_names.get_indexer(common_genes)
    if np.any(indices < 0):
        missing = [g for g, idx in zip(common_genes, indices) if idx < 0][:5]
        raise ValueError(f"Section missing common genes unexpectedly: {missing}")
    x = adata.X[:, indices]
    if sparse.issparse(x):
        x = x.copy().astype(np.float64)
        x.data = np.log1p(x.data)
        sums = np.asarray(x.sum(axis=0)).ravel()
        sums_sq = np.asarray(x.multiply(x).sum(axis=0)).ravel()
        n = x.shape[0]
        var = np.maximum(sums_sq / n - (sums / n) ** 2, 0.0)
    else:
        arr = np.log1p(np.asarray(x, dtype=np.float64))
        var = arr.var(axis=0)
    return pd.Series(var, index=common_genes)


def select_deconv_genes(
    adatas: list,
    profiles: pd.DataFrame,
    gene_limit: int,
) -> pd.DataFrame:
    common = set(profiles.index)
    for adata in adatas:
        common &= set(adata.var_names.astype(str))
    common_genes = sorted(common)
    if len(common_genes) < 200:
        raise ValueError(f"Only {len(common_genes)} shared genes between reference and Visium sections.")

    ref_var = profiles.loc[common_genes].var(axis=1).astype(float)
    st_vars = [log1p_gene_variance_for_common(adata, common_genes) for adata in adatas]
    st_var = pd.concat(st_vars, axis=1).mean(axis=1)

    ref_score = ref_var / (ref_var.max() + 1e-12)
    st_score = st_var / (st_var.max() + 1e-12)
    score = 0.65 * ref_score + 0.35 * st_score

    panel = pd.DataFrame(
        {
            "gene": common_genes,
            "reference_celltype_variance": ref_var.values,
            "visium_log1p_variance": st_var.values,
            "selection_score": score.values,
        }
    ).sort_values("selection_score", ascending=False)
    return panel.head(min(gene_limit, len(panel))).reset_index(drop=True)


def normalize_spot_matrix(adata, selected_genes: list[str]) -> np.ndarray:
    indices = adata.var_names.get_indexer(selected_genes)
    x_sel = dense_array(adata.X[:, indices]).astype(np.float64)
    lib = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float64)
    scale = float(np.median(lib[lib > 0]))
    x_norm = x_sel / (lib[:, None] + 1e-10) * scale
    return np.log1p(x_norm)


def deconvolve_section(
    adata,
    section: str,
    profiles: pd.DataFrame,
    selected_genes: list[str],
    out_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    cell_types = profiles.columns.tolist()
    a = profiles.loc[selected_genes, cell_types].to_numpy(dtype=np.float64)
    keep = a.sum(axis=1) > 0
    a = a[keep, :]
    genes_used = [g for g, flag in zip(selected_genes, keep) if flag]
    x = normalize_spot_matrix(adata, genes_used)

    gene_scale = np.maximum(a.std(axis=1), 0.05)
    a_scaled = a / gene_scale[:, None]
    x_scaled = x / gene_scale[None, :]

    weights = np.zeros((adata.n_obs, len(cell_types)), dtype=np.float64)
    residual = np.zeros(adata.n_obs, dtype=np.float64)
    for i in range(adata.n_obs):
        w, r = nnls(a_scaled, x_scaled[i])
        weights[i, :] = w
        residual[i] = r
    omega = normalize_rows(weights)
    rel_resid = residual / (np.linalg.norm(x_scaled, axis=1) + 1e-10)

    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    out = pd.DataFrame(
        {
            "spot_id": adata.obs_names.astype(str),
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )
    for j, ct in enumerate(cell_types):
        out[ct] = omega[:, j]
    out["dominant_ref_cell_type"] = [cell_types[j] for j in np.argmax(omega, axis=1)]
    out["omega_entropy"] = entropy_rows(omega)
    out["nnls_residual"] = residual
    out["relative_residual"] = rel_resid

    out_path = out_dir / f"mouse_reference_nnls_omega_{section}.csv"
    out.to_csv(out_path, index=False)

    summary = {
        "section": section,
        "n_spots": int(adata.n_obs),
        "n_genes_used": int(len(genes_used)),
        "median_relative_residual": float(np.median(rel_resid)),
        "mean_entropy": float(out["omega_entropy"].mean()),
        "output": str(out_path),
    }
    for ct in cell_types:
        summary[f"mean_omega_{ct}"] = float(out[ct].mean())
        summary[f"dominant_count_{ct}"] = int((out["dominant_ref_cell_type"] == ct).sum())
    return out, summary


def draw_deconv_qc(summary: pd.DataFrame, cell_types: list[str], out_path: Path):
    sections = summary["section"].tolist()
    w, h = 1380, 760
    left, top, plot_w, plot_h = 95, 130, 910, 430
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    title_font = font(30, bold=True)
    label_font = font(19, bold=True)
    small_font = font(14)
    draw.text((42, 34), "Mouse brain reference NNLS deconvolution", fill=(20, 20, 20), font=title_font)
    draw.text((42, 72), "Mean spot-level cell-type weights by Visium section", fill=(72, 72, 72), font=small_font)

    ymax = max(0.05, float(summary[[f"mean_omega_{ct}" for ct in cell_types]].to_numpy().max()) * 1.20)
    for tick in np.linspace(0, ymax, 5):
        y = top + plot_h - tick / ymax * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill=(226, 226, 226, 255))
        draw.text((left - 62, y - 8), f"{tick:.2f}", fill=(45, 45, 45), font=small_font)
    draw.line([(left, top), (left, top + plot_h)], fill=(20, 20, 20), width=2)
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=(20, 20, 20), width=2)

    colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    group_gap = 50
    group_w = (plot_w - group_gap * (len(sections) + 1)) / len(sections)
    bar_gap = 5
    bar_w = (group_w - bar_gap * (len(cell_types) - 1)) / len(cell_types)
    for si, section in enumerate(sections):
        gx = left + group_gap + si * (group_w + group_gap)
        row = summary[summary["section"] == section].iloc[0]
        for ci, ct in enumerate(cell_types):
            val = float(row[f"mean_omega_{ct}"])
            x0 = gx + ci * (bar_w + bar_gap)
            x1 = x0 + bar_w
            y0 = top + plot_h - val / ymax * plot_h
            draw.rectangle([x0, y0, x1, top + plot_h], fill=colors[ci % len(colors)] + (225,))
        draw.text((gx, top + plot_h + 16), section, fill=(25, 25, 25), font=small_font)

    lx, ly = left + plot_w + 42, top + 4
    draw.text((lx, ly - 30), "Cell type", fill=(20, 20, 20), font=label_font)
    for ci, ct in enumerate(cell_types):
        yy = ly + ci * 36
        draw.rectangle([lx, yy, lx + 24, yy + 17], fill=colors[ci % len(colors)] + (225,))
        draw.text((lx + 34, yy - 2), ct[:35], fill=(20, 20, 20), font=small_font)

    rx, ry = left, top + plot_h + 88
    draw.text((rx, ry), "Median relative residual", fill=(20, 20, 20), font=label_font)
    for i, (_, row) in enumerate(summary.iterrows()):
        txt = f"{row['section']}: {row['median_relative_residual']:.3f}"
        draw.text((rx + 245 + i * 255, ry + 2), txt, fill=(55, 55, 55), font=small_font)
    img.save(out_path)


def make_ap_clients(adatas, omega_tables, cell_types: list[str], genes: list[str], device: str):
    clients = []
    for section, adata in zip([short_section_name(s) for s in SECTIONS], adatas):
        omega_df = omega_tables[section].set_index("spot_id")
        omega_df = omega_df.loc[adata.obs_names.astype(str), cell_types]
        omega = omega_df.to_numpy(dtype=np.float32)
        omega = normalize_rows(omega).astype(np.float32)
        indices = adata.var_names.get_indexer(genes)
        x_sel = dense_selected_matrix(adata.X, indices)
        clients.append(
            {
                "name": section,
                "y": torch.tensor(x_sel, device=device),
                "s": torch.tensor(np.asarray(adata.obsm["spatial"], dtype=np.float32), device=device),
                "omega": torch.tensor(omega, device=device),
                "n_spots": int(adata.n_obs),
            }
        )
    return clients


def extract_reference_top_genes(global_state, cell_types: list[str], gene_names: list[str]) -> pd.DataFrame:
    beta = global_state["beta"].detach().cpu().numpy()
    theta = global_state["theta"].detach().cpu().numpy()
    rows = []
    for kk, ct in enumerate(cell_types):
        for gg, gene in enumerate(gene_names):
            z1 = float(beta[0, kk, gg])
            z2 = float(beta[1, kk, gg])
            rows.append(
                {
                    "Cell_Type": kk,
                    "Cell_Type_Name": ct,
                    "Gene": gene,
                    "Gene_ID": gg,
                    "Beta_h1": z1,
                    "Beta_h2": z2,
                    "Abs_Beta_Max": max(abs(z1), abs(z2)),
                    "Theta": float(theta[0, kk, gg]),
                }
            )
    return pd.DataFrame(rows).sort_values("Abs_Beta_Max", ascending=False)


def run_reference_apfedadam(
    adatas,
    omega_tables,
    cell_types: list[str],
    selected_panel: pd.DataFrame,
    args,
) -> dict:
    genes = selected_panel["gene"].head(args.ap_gene_limit).tolist()
    clients = make_ap_clients(adatas, omega_tables, cell_types, genes, args.device)
    ap_args = SimpleNamespace(
        rounds=args.ap_rounds,
        local_steps=args.ap_local_steps,
        lr=args.ap_lr,
        l1_lambda=args.ap_l1_lambda,
        drift_scale=args.ap_drift_scale,
        print_every=args.ap_print_every,
    )
    print(
        f"Running AP-FedAdam with reference omega: {ap_args.rounds} rounds, "
        f"K={len(cell_types)}, G={len(genes)}"
    )
    t0 = time.time()
    global_state, history = run_federated(clients, len(cell_types), len(genes), ap_args)
    elapsed = time.time() - t0

    ranking = extract_reference_top_genes(global_state, cell_types, genes)
    hist_df = pd.DataFrame(history)
    ranking_path = OUT_DIR / "mouse_reference_nnls_ctsv_ranking.csv"
    history_path = OUT_DIR / "mouse_reference_nnls_ctsv_history.csv"
    ranking.to_csv(ranking_path, index=False)
    hist_df.to_csv(history_path, index=False)
    return {
        "ranking_path": str(ranking_path),
        "history_path": str(history_path),
        "elapsed_seconds": float(elapsed),
        "final_loss": float(history[-1]["loss"]),
        "top_pairs": ranking.head(10).to_dict(orient="records"),
    }


def markdown_table(df: pd.DataFrame) -> str:
    """Return a compact GitHub-flavored Markdown table without extra deps."""
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.6g}")
        else:
            view[col] = view[col].astype(str)
        view[col] = view[col].str.replace("|", "\\|", regex=False)
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    manifest: dict,
    ref_summary: pd.DataFrame,
    deconv_summary: pd.DataFrame,
    gene_panel: pd.DataFrame,
    ap_result: dict | None,
) -> Path:
    report_path = out_dir / "mouse_reference_deconv_results.md"
    lines = [
        "# Mouse Brain Reference Deconvolution",
        "",
        "## Reference",
        "",
        "- Source files: GEO GSE71585 `RefSeq_counts` + `Clustering_Results`.",
        f"- Label column: `{manifest['label_col']}`.",
        f"- Reference cells used: {manifest['n_reference_cells_used']}.",
        f"- Reference genes: {manifest['n_reference_genes']}.",
        f"- Cell types: {', '.join(manifest['cell_types'])}.",
        "",
        "## Reference Cell Counts",
        "",
        markdown_table(ref_summary),
        "",
        "## Visium Deconvolution Summary",
        "",
        markdown_table(
            deconv_summary[
                [
                    "section",
                    "n_spots",
                    "n_genes_used",
                    "median_relative_residual",
                    "mean_entropy",
                ]
            ]
        ),
        "",
        "## Selected Genes",
        "",
        markdown_table(gene_panel.head(20)),
        "",
    ]
    if ap_result is not None:
        top = pd.DataFrame(ap_result["top_pairs"])[
            ["Cell_Type_Name", "Gene", "Abs_Beta_Max", "Beta_h1", "Beta_h2"]
        ]
        lines += [
            "## AP-FedAdam With Reference Omega",
            "",
            f"- Final loss: {ap_result['final_loss']:.4f}.",
            f"- Elapsed seconds: {ap_result['elapsed_seconds']:.1f}.",
            f"- Ranking CSV: `{ap_result['ranking_path']}`.",
            "",
            markdown_table(top),
            "",
        ]
    lines += [
        "## Method Note",
        "",
        "This run uses NNLS over matched scRNA pseudobulk profiles because the current",
        "machine has no R/Rscript installation for RCTD/SPOTlight and no installed",
        "cell2location/scvi stack. The output is still a formal reference-matrix",
        "deconvolution input for downstream omega-based CTSV experiments.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args():
    p = argparse.ArgumentParser(description="Mouse brain reference deconvolution")
    p.add_argument("--label-col", default="broad_type")
    p.add_argument(
        "--visium-source",
        choices=("local", "squidpy"),
        default="local",
        help="Use bundled local Visium inputs (default) or Squidpy's public-data cache.",
    )
    p.add_argument("--min-cells-per-type", type=int, default=10)
    p.add_argument("--include-unclassified", action="store_true")
    p.add_argument("--gene-limit", type=int, default=800)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-apfedadam", action="store_true")
    p.add_argument("--ap-rounds", type=int, default=80)
    p.add_argument("--ap-local-steps", type=int, default=5)
    p.add_argument("--ap-gene-limit", type=int, default=300)
    p.add_argument("--ap-lr", type=float, default=0.05)
    p.add_argument("--ap-l1-lambda", type=float, default=5e-6)
    p.add_argument("--ap-drift-scale", type=float, default=0.1)
    p.add_argument("--ap-print-every", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(42)
    torch.manual_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading reference profiles...")
    profiles, ref_summary, manifest = load_reference_profiles(
        COUNTS_PATH,
        LABELS_PATH,
        args.label_col,
        args.min_cells_per_type,
        exclude_unclassified=not args.include_unclassified,
    )
    cell_types = profiles.columns.tolist()
    ref_summary_path = OUT_DIR / "mouse_reference_celltype_counts.csv"
    ref_summary.to_csv(ref_summary_path, index=False)

    print("Loading Visium sections...")
    adatas = []
    for section in SECTIONS:
        adata = load_visium_section(section, args.visium_source)
        adata.var_names_make_unique()
        adatas.append(adata)
        print(f"  {short_section_name(section)}: {adata.n_obs} spots, {adata.n_vars} genes")

    print("Selecting shared informative genes...")
    gene_panel = select_deconv_genes(adatas, profiles, args.gene_limit)
    gene_panel_path = OUT_DIR / "mouse_reference_nnls_gene_panel.csv"
    gene_panel.to_csv(gene_panel_path, index=False)
    selected_genes = gene_panel["gene"].tolist()
    print(f"  selected {len(selected_genes)} genes")
    print("  top genes: " + ", ".join(selected_genes[:10]))

    print("Running NNLS deconvolution...")
    omega_tables = {}
    summary_rows = []
    for full_section, adata in zip(SECTIONS, adatas):
        section = short_section_name(full_section)
        table, summary = deconvolve_section(adata, section, profiles, selected_genes, OUT_DIR)
        omega_tables[section] = table
        summary_rows.append(summary)
        means = ", ".join(f"{ct}={summary[f'mean_omega_{ct}']:.3f}" for ct in cell_types)
        print(f"  {section}: median rel resid={summary['median_relative_residual']:.3f}; {means}")

    deconv_summary = pd.DataFrame(summary_rows)
    deconv_summary_path = OUT_DIR / "mouse_reference_nnls_omega_summary.csv"
    deconv_summary.to_csv(deconv_summary_path, index=False)
    qc_path = OUT_DIR / "mouse_reference_nnls_omega_qc.png"
    draw_deconv_qc(deconv_summary, cell_types, qc_path)

    ap_result = None
    if args.run_apfedadam:
        ap_result = run_reference_apfedadam(adatas, omega_tables, cell_types, gene_panel, args)

    manifest.update(
        {
            "gene_limit": args.gene_limit,
            "gene_panel_path": str(gene_panel_path),
            "reference_summary_path": str(ref_summary_path),
            "deconvolution_summary_path": str(deconv_summary_path),
            "qc_figure_path": str(qc_path),
            "ap_fedadam": ap_result,
        }
    )
    manifest_path = OUT_DIR / "mouse_reference_deconv_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path = write_report(OUT_DIR, manifest, ref_summary, deconv_summary, gene_panel, ap_result)

    print("\nOutputs:")
    print(f"  {ref_summary_path}")
    print(f"  {gene_panel_path}")
    print(f"  {deconv_summary_path}")
    print(f"  {qc_path}")
    print(f"  {manifest_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
