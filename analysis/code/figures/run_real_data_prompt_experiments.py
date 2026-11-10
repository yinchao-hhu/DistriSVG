#!/usr/bin/env python3
"""Utilities and runnable outputs for `Real data prompt(1).md`.

This script covers the real-data directions that can be executed with the
current workspace:

1. Mouse brain Visium marker-score omega outputs and QC summaries.
2. A reusable NNLS reference-deconvolution function for when a scRNA reference
   matrix with cell-type labels is supplied.
3. A single-cell-level ST one-hot omega demo.
4. A 3D spot-level basis-function demo.

The current workspace does not contain a matched mouse-brain scRNA reference
matrix with cell-type labels, so the formal reference-based deconvolution path
is implemented but not run by default.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import sparse
from scipy.optimize import nnls

try:
    import scanpy as sc
    import squidpy as sq
except Exception as exc:
    raise SystemExit("This script requires scanpy and squidpy in the local .venv.") from exc

from run_mouse_brain_analysis import CELLTYPE_MARKERS, SECTIONS, compute_celltype_proportions


ROOT = Path(__file__).resolve().parents[3] / "04_数值实验"
OUT_DIR = ROOT / "overleaf" / "figs" / "real_data_prompt"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def dense_array(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, None)
    return x / (x.sum(axis=1, keepdims=True) + eps)


def entropy_rows(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def draw_mouse_omega_qc(summary: pd.DataFrame, out_path: Path):
    sections = summary["section"].drop_duplicates().tolist()
    cell_types = [ct for ct in CELLTYPE_MARKERS]
    w, h = 1120, 720
    left, top, plot_w, plot_h = 100, 120, 760, 410
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    title_font = font(31, bold=True)
    label_font = font(21, bold=True)
    small_font = font(16)
    draw.text((40, 35), "Mouse brain marker-score omega QC", fill=(20, 20, 20), font=title_font)
    draw.text((40, 72), "Mean normalized marker-score weights by section", fill=(70, 70, 70), font=small_font)

    ymax = max(0.05, float(summary[[f"mean_omega_{ct}" for ct in cell_types]].to_numpy().max()) * 1.25)
    for tick in np.linspace(0, ymax, 5):
        y = top + plot_h - tick / ymax * plot_h
        draw.line([(left, y), (left + plot_w, y)], fill=(226, 226, 226, 255))
        draw.text((left - 70, y - 9), f"{tick:.2f}", fill=(40, 40, 40), font=small_font)
    draw.line([(left, top), (left, top + plot_h)], fill=(20, 20, 20), width=2)
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill=(20, 20, 20), width=2)

    colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
    ]
    group_gap = 58
    group_w = (plot_w - group_gap * (len(sections) + 1)) / len(sections)
    bar_gap = 7
    bar_w = (group_w - bar_gap * (len(cell_types) - 1)) / len(cell_types)
    for si, sec in enumerate(sections):
        gx = left + group_gap + si * (group_w + group_gap)
        row = summary[summary["section"] == sec].iloc[0]
        for ci, ct in enumerate(cell_types):
            val = float(row[f"mean_omega_{ct}"])
            x0 = gx + ci * (bar_w + bar_gap)
            x1 = x0 + bar_w
            y0 = top + plot_h - val / ymax * plot_h
            draw.rectangle([x0, y0, x1, top + plot_h], fill=colors[ci] + (220,))
        sec_label = sec.replace("V1_Mouse_Brain_Sagittal_", "").replace("_", "\n")
        draw.multiline_text((gx, top + plot_h + 16), sec_label, fill=(25, 25, 25), font=small_font, spacing=1)

    lx, ly = left + plot_w + 40, top + 18
    draw.text((lx, ly - 30), "Cell type", fill=(20, 20, 20), font=label_font)
    for ci, ct in enumerate(cell_types):
        yy = ly + ci * 38
        draw.rectangle([lx, yy, lx + 26, yy + 18], fill=colors[ci] + (220,))
        draw.text((lx + 38, yy - 2), ct, fill=(20, 20, 20), font=small_font)
    draw.text((left, top - 34), "Mean omega", fill=(20, 20, 20), font=label_font)
    img.save(out_path)


def run_mouse_marker_proxy() -> dict[str, Path]:
    marker_rows = []
    summary_rows = []
    section_output_paths = {}

    for section in SECTIONS:
        adata = sq.datasets.visium(section)
        adata.var_names_make_unique()
        omega_dict = compute_celltype_proportions(adata, CELLTYPE_MARKERS)
        cell_types = list(CELLTYPE_MARKERS)
        omega = np.stack([omega_dict[ct] for ct in cell_types], axis=1).astype(np.float64)
        omega = normalize_rows(omega)
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
        out = pd.DataFrame({"spot_id": adata.obs_names.astype(str), "x": coords[:, 0], "y": coords[:, 1]})
        for i, ct in enumerate(cell_types):
            out[ct] = omega[:, i]
        out["dominant_proxy_cell_type"] = [cell_types[i] for i in np.argmax(omega, axis=1)]
        out["omega_entropy"] = entropy_rows(omega)
        out_path = OUT_DIR / f"mouse_marker_proxy_omega_{section}.csv"
        out.to_csv(out_path, index=False)
        section_output_paths[section] = out_path

        row = {
            "section": section,
            "n_spots": adata.n_obs,
            "n_genes": adata.n_vars,
            "mean_entropy": float(out["omega_entropy"].mean()),
        }
        dom_counts = out["dominant_proxy_cell_type"].value_counts().to_dict()
        for ct in cell_types:
            row[f"mean_omega_{ct}"] = float(out[ct].mean())
            row[f"dominant_count_{ct}"] = int(dom_counts.get(ct, 0))
            present = [m for m in CELLTYPE_MARKERS[ct] if m in adata.var_names]
            marker_rows.append(
                {
                    "section": section,
                    "cell_type": ct,
                    "markers_requested": ",".join(CELLTYPE_MARKERS[ct]),
                    "markers_present": ",".join(present),
                    "n_markers_present": len(present),
                }
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    markers = pd.DataFrame(marker_rows)
    summary_path = OUT_DIR / "mouse_marker_proxy_omega_summary.csv"
    marker_path = OUT_DIR / "mouse_marker_proxy_marker_presence.csv"
    fig_path = OUT_DIR / "mouse_marker_proxy_omega_qc.png"
    summary.to_csv(summary_path, index=False)
    markers.to_csv(marker_path, index=False)
    draw_mouse_omega_qc(summary, fig_path)
    return {"summary": summary_path, "markers": marker_path, "figure": fig_path, **section_output_paths}


def reference_nnls_deconvolution(st_adata, ref_adata, label_col: str, max_spots: int | None = None):
    """Reference-based spot deconvolution by nonnegative least squares.

    This is a lightweight fallback when a full method such as SPOTlight/RCTD or
    cell2location is unavailable. It constructs cell-type pseudobulk profiles
    from `ref_adata` and fits each spot expression vector as a nonnegative
    combination of those profiles.
    """
    if label_col not in ref_adata.obs:
        raise ValueError(f"Reference AnnData missing label column: {label_col}")

    shared = st_adata.var_names.intersection(ref_adata.var_names)
    if len(shared) < 100:
        raise ValueError(f"Only {len(shared)} shared genes; deconvolution is likely unstable.")

    ref = ref_adata[:, shared].copy()
    st = st_adata[:, shared].copy()
    labels = ref.obs[label_col].astype(str)
    cell_types = sorted(labels.unique().tolist())
    profiles = []
    for ct in cell_types:
        x = ref[labels == ct].X
        profiles.append(np.asarray(dense_array(x).mean(axis=0)).ravel())
    profile_matrix = np.stack(profiles, axis=1)
    profile_matrix = np.log1p(profile_matrix)

    x_st = dense_array(st.X)
    if max_spots is not None:
        x_st = x_st[:max_spots]
    x_st = np.log1p(x_st)
    weights = []
    residuals = []
    for i in range(x_st.shape[0]):
        w, resid = nnls(profile_matrix, x_st[i])
        weights.append(w)
        residuals.append(resid)
    omega = normalize_rows(np.vstack(weights))
    return pd.DataFrame(omega, columns=cell_types), np.asarray(residuals)


def make_single_cell_onehot_demo() -> dict[str, Path]:
    rng = np.random.default_rng(42)
    cell_types = ["Neuron", "Astrocyte", "Oligodendrocyte", "Microglia"]
    n_cells = 160
    labels = rng.choice(cell_types, size=n_cells, p=[0.48, 0.20, 0.22, 0.10])
    coords = rng.normal(size=(n_cells, 2))
    omega = np.zeros((n_cells, len(cell_types)), dtype=int)
    for i, label in enumerate(labels):
        omega[i, cell_types.index(label)] = 1
    out = pd.DataFrame({"cell_id": [f"cell_{i:04d}" for i in range(n_cells)], "x": coords[:, 0], "y": coords[:, 1], "cell_type": labels})
    for j, ct in enumerate(cell_types):
        out[ct] = omega[:, j]
    out_path = OUT_DIR / "single_cell_st_onehot_omega_demo.csv"
    out.to_csv(out_path, index=False)

    summary = (
        out["cell_type"]
        .value_counts()
        .rename_axis("cell_type")
        .reset_index(name="n_cells")
        .sort_values("cell_type")
    )
    summary["omega_rule"] = "one-hot: 1 for the observed cell type label, 0 otherwise"
    summary_path = OUT_DIR / "single_cell_st_onehot_omega_demo_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {"onehot": out_path, "summary": summary_path}


class BasisGenerator3D:
    def __init__(self, coords: np.ndarray, mode: str = "cos3d"):
        self.mode = mode
        self.mean = coords.mean(axis=0, keepdims=True)
        self.std = coords.std(axis=0, keepdims=True) + 1e-8
        norm = (coords - self.mean) / self.std
        abs_norm = np.abs(norm)
        self.psi = np.quantile(abs_norm, 0.6, axis=0) + 1e-8

    def compute_h(self, coords: np.ndarray) -> np.ndarray:
        norm = (coords - self.mean) / self.std
        if self.mode == "linear3d":
            return norm
        if self.mode == "cos3d":
            return np.stack([np.cos(np.pi * norm[:, j] / self.psi[j]) for j in range(3)], axis=1)
        if self.mode == "gau3d":
            h = np.stack([np.exp(-(norm[:, j] ** 2) / (2 * self.psi[j] ** 2)) for j in range(3)], axis=1)
            return (h - h.mean(axis=0, keepdims=True)) / (h.std(axis=0, keepdims=True) + 1e-8)
        raise ValueError(f"Unknown 3D basis mode: {self.mode}")


def make_3d_basis_demo() -> dict[str, Path]:
    rng = np.random.default_rng(42)
    slices = []
    for z in [0.0, 1.0, 2.0]:
        xy = rng.uniform(0, 10, size=(120, 2))
        zz = np.full((120, 1), z)
        slices.append(np.hstack([xy, zz]))
    coords = np.vstack(slices)
    rows = []
    summary_rows = []
    for mode in ["linear3d", "cos3d", "gau3d"]:
        basis = BasisGenerator3D(coords, mode=mode)
        h = basis.compute_h(coords)
        for i in range(min(30, coords.shape[0])):
            rows.append(
                {
                    "mode": mode,
                    "spot_id": f"demo_spot_{i:03d}",
                    "x": coords[i, 0],
                    "y": coords[i, 1],
                    "z": coords[i, 2],
                    "h_x": h[i, 0],
                    "h_y": h[i, 1],
                    "h_z": h[i, 2],
                }
            )
        for j, axis in enumerate(["x", "y", "z"]):
            summary_rows.append(
                {
                    "mode": mode,
                    "feature": f"h_{axis}",
                    "mean": float(h[:, j].mean()),
                    "std": float(h[:, j].std()),
                    "min": float(h[:, j].min()),
                    "max": float(h[:, j].max()),
                }
            )
    demo_path = OUT_DIR / "spot_level_3d_basis_demo_first30.csv"
    summary_path = OUT_DIR / "spot_level_3d_basis_demo_summary.csv"
    pd.DataFrame(rows).to_csv(demo_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return {"basis_rows": demo_path, "summary": summary_path}


def write_report(paths: dict[str, Path]):
    report = OUT_DIR / "real_data_prompt_results.md"
    lines = [
        "# Real Data Prompt Results",
        "",
        "## Mouse Brain Deconvolution Status",
        "",
        "The workspace currently contains the three local 10x Visium mouse-brain sections but does not contain a matched scRNA-seq reference matrix with cell-type labels. Therefore, formal reference-based deconvolution was not run.",
        "",
        "Instead, this round outputs the existing marker-score proxy omega matrices and QC summaries. This marker-score approach is acceptable only as an exploratory substitute when a scRNA reference is unavailable; it should not be described as SPOTlight/RCTD/cell2location-style deconvolution.",
        "",
        "Recommended formal route when a reference is available: build cell-type pseudobulk profiles from the scRNA reference and run SPOTlight, RCTD, cell2location, or the lightweight NNLS template implemented in this script. For mouse brain, a reasonable public reference source is the Allen mouse brain single-cell/single-nucleus transcriptomic atlas.",
        "",
        "## Single-cell-level ST",
        "",
        "For single-cell-resolution spatial transcriptomics, deconvolution is unnecessary. The omega matrix should be one-hot by cell type label. The demo output shows this AP-FedAdam-ready encoding.",
        "",
        "Compared with running SVG detection separately within each cell type, the unified AP-FedAdam formulation keeps all cell types in a single likelihood, gives comparable gene-cell-type coefficients, supports federation across samples/sites, and can share the same inference machinery. Separate per-cell-type SVG tests are simpler and may be sufficient when each cell type has many cells and no cross-cell-type comparison or federated analysis is needed.",
        "",
        "## 3D spot-level ST",
        "",
        "The model extension is straightforward at the design-matrix level: replace 2D basis features h(x), h(y) by 3D features h(x), h(y), h(z). The demo output verifies linear, cosine, and Gaussian 3D basis construction. A real 3D experiment still needs 3D coordinates plus either deconvolved spot-level omega or one-hot cell labels if the assay is single-cell resolution.",
        "",
        "## Output Files",
        "",
    ]
    for key, path in sorted(paths.items()):
        lines.append(f"- `{key}`: `{path}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Run outputs requested by Real data prompt(1).md.")
    parser.add_argument("--skip-mouse", action="store_true")
    parser.add_argument("--skip-single-cell-demo", action="store_true")
    parser.add_argument("--skip-3d-demo", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    all_paths: dict[str, Path] = {}
    if not args.skip_mouse:
        for key, path in run_mouse_marker_proxy().items():
            all_paths[f"mouse_{key}"] = path
    if not args.skip_single_cell_demo:
        for key, path in make_single_cell_onehot_demo().items():
            all_paths[f"single_cell_{key}"] = path
    if not args.skip_3d_demo:
        for key, path in make_3d_basis_demo().items():
            all_paths[f"3d_{key}"] = path
    manifest = OUT_DIR / "real_data_prompt_manifest.json"
    manifest.write_text(json.dumps({k: str(v) for k, v in sorted(all_paths.items())}, indent=2), encoding="utf-8")
    all_paths["manifest"] = manifest
    report = write_report(all_paths)
    print(f"Wrote report: {report}")
    for key, path in sorted(all_paths.items()):
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
