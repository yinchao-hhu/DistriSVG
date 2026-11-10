#!/usr/bin/env python3
'Create histology-backed mouse-brain residual maps for anatomical review.'

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[3] / "04_数值实验"
DATA_DIR = ROOT / "data"
OMEGA_DIR = ROOT / "overleaf" / "figs" / "mouse_reference_deconv"
FIG_DIR = ROOT / "overleaf" / "figs" / "cross_section_spatial_maps"
SOURCE_DIR = FIG_DIR / "source_data"

SECTIONS = {
    "Anterior": "V1_Mouse_Brain_Sagittal_Anterior",
    "Posterior": "V1_Mouse_Brain_Sagittal_Posterior",
    "Anterior2": "V1_Mouse_Brain_Sagittal_Anterior_Section_2",
}
SECTION_LABELS = {
    "Anterior": "Anterior section",
    "Posterior": "Posterior section",
    "Anterior2": "Anterior replicate",
}
PAIRS = (
    ("Olfm1", "Endothelial Cell"),
    ("Cck", "Astrocyte"),
    ("Enpp2", "GABA-ergic Neuron"),
    ("Bex2", "Oligodendrocyte"),
)

REGION_INTERPRETATIONS = {
    ("Olfm1", "Anterior"): ("Olfactory bulb + cortical mantle", "moderate", "Nakaya et al., 2012"),
    ("Olfm1", "Posterior"): ("Cortex/hippocampus + cerebellar cortex", "moderate", "Nakaya et al., 2012"),
    ("Olfm1", "Anterior2"): ("Olfactory bulb + cortical mantle", "moderate", "Nakaya et al., 2012"),
    ("Cck", "Anterior"): ("Cortical mantle + olfactory bulb", "moderate", "Vitale et al., 1991"),
    ("Cck", "Posterior"): ("Cortex/hippocampus + cerebellar cortex", "moderate", "Vitale et al., 1991"),
    ("Cck", "Anterior2"): ("Cortical mantle + olfactory bulb", "moderate", "Vitale et al., 1991"),
    ("Enpp2", "Anterior"): ("Olfactory bulb + caudal/deep forebrain", "low-moderate", "Savaskan et al., 2007"),
    ("Enpp2", "Posterior"): ("Cerebellar/hindbrain + deep forebrain", "low-moderate", "Savaskan et al., 2007"),
    ("Enpp2", "Anterior2"): ("Olfactory bulb + caudal/deep forebrain", "low-moderate", "Savaskan et al., 2007"),
    ("Bex2", "Anterior"): ("Caudal/deep forebrain", "low-moderate", "Koo et al., 2005"),
    ("Bex2", "Posterior"): ("Cerebellar + fiber-rich/deep regions", "low-moderate", "Koo et al., 2005"),
    ("Bex2", "Anterior2"): ("Caudal/deep forebrain", "low-moderate", "Koo et al., 2005"),
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.3,
            "axes.linewidth": 0.45,
            "figure.dpi": 150,
        }
    )


def read_10x_selected_genes(path: Path, genes: list[str]) -> pd.DataFrame:
    """Read selected genes and library sizes directly from a 10x HDF5 matrix."""

    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        names = np.asarray(group["features/name"]).astype("U")
        feature_types = np.asarray(group["features/feature_type"]).astype("U")
        barcodes = np.asarray(group["barcodes"]).astype("U")
        shape = tuple(int(value) for value in np.asarray(group["shape"]))
        matrix = sparse.csc_matrix(
            (
                np.asarray(group["data"]),
                np.asarray(group["indices"]),
                np.asarray(group["indptr"]),
            ),
            shape=shape,
        )

    gene_expression = feature_types == "Gene Expression"
    library_size = np.asarray(matrix[gene_expression, :].sum(axis=0)).ravel().astype(float)
    selected_rows = []
    for gene in genes:
        matches = np.flatnonzero((names == gene) & gene_expression)
        if len(matches) != 1:
            raise ValueError(f"Expected one Gene Expression row for {gene}; found {len(matches)}")
        selected_rows.append(int(matches[0]))
    counts = matrix[selected_rows, :].T.toarray().astype(float)
    expression = np.log1p(counts / (library_size[:, None] + 1e-10) * 1e4)
    expr = pd.DataFrame(expression, columns=genes)
    expr.insert(0, "library_size", library_size)
    expr.insert(0, "spot_id", barcodes)
    return expr


def load_section(section: str, folder: str, genes: list[str]) -> tuple[pd.DataFrame, np.ndarray, float]:
    expr = read_10x_selected_genes(DATA_DIR / folder / "filtered_feature_bc_matrix.h5", genes)

    omega = pd.read_csv(OMEGA_DIR / f"mouse_reference_nnls_omega_{section}.csv")
    frame = omega.merge(expr, on="spot_id", how="inner", validate="one_to_one")
    scale = json.loads((DATA_DIR / folder / "spatial" / "scalefactors_json.json").read_text())["tissue_hires_scalef"]
    image = mpimg.imread(DATA_DIR / folder / "spatial" / "tissue_hires_image.png")
    return frame, image, float(scale)


def percentile(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def crop_bounds(frame: pd.DataFrame, scale: float, image: np.ndarray) -> tuple[float, float, float, float]:
    x = frame["x"].to_numpy(dtype=float) * scale
    y = frame["y"].to_numpy(dtype=float) * scale
    margin = 35.0
    return (
        max(0.0, float(x.min() - margin)),
        min(float(image.shape[1]), float(x.max() + margin)),
        max(0.0, float(y.min() - margin)),
        min(float(image.shape[0]), float(y.max() + margin)),
    )


def composition_adjusted_values(frame: pd.DataFrame, gene: str, cell_type: str) -> dict[str, np.ndarray | float]:
    expression = frame[gene].to_numpy(dtype=float)
    weight = frame[cell_type].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(frame)), weight])
    coefficients, *_ = np.linalg.lstsq(design, expression, rcond=None)
    fitted = design @ coefficients
    residual = expression - fitted
    residual_sd = float(residual.std(ddof=1))
    residual_z = residual / residual_sd if residual_sd > 0 else np.zeros_like(residual)
    legacy_proxy = expression * weight
    return {
        "expression": expression,
        "weight": weight,
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "fitted": fitted,
        "residual": residual,
        "residual_z": residual_z,
        "residual_percentile": percentile(residual),
        "legacy_proxy": legacy_proxy,
    }


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def moran_knn(x: np.ndarray, y: np.ndarray, values: np.ndarray, seed: int) -> tuple[float, float]:
    coordinates = np.column_stack([x, y])
    k = min(6, len(coordinates) - 1)
    neighbours = cKDTree(coordinates).query(coordinates, k=k + 1)[1][:, 1:]
    centered = values - values.mean()
    denominator = float(np.square(centered).sum())
    observed = float(len(values) / (len(values) * k) * np.sum(centered[:, None] * centered[neighbours]) / denominator)
    rng = np.random.default_rng(seed)
    null = np.empty(999, dtype=float)
    for index in range(len(null)):
        permuted = rng.permutation(centered)
        null[index] = float(len(values) / (len(values) * k) * np.sum(permuted[:, None] * permuted[neighbours]) / denominator)
    p_value = float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))
    return observed, p_value


def export_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 450},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(FIG_DIR / f"{stem}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_qa_notes(moran: pd.DataFrame, similarity: pd.DataFrame) -> None:
    grouped = similarity.groupby("section")[["legacy_proxy_pearson_r", "adjusted_residual_pearson_r"]].mean()
    correlation_lines = [
        f"  - {section}: legacy mean Pearson r={row['legacy_proxy_pearson_r']:.3f}; "
        f"adjusted mean Pearson r={row['adjusted_residual_pearson_r']:.3f}."
        for section, row in grouped.iterrows()
    ]
    notes = """# Mouse region-overlay QA notes

- The final figure uses each Visium section's own raw H&E image and full-resolution spot coordinates.
- For every displayed gene--cell-type pair, log-CP10K expression is regressed on that cell-type's reference-derived NNLS weight within the section. Only the top within-section decile of the resulting residual is colored; all other spots are transparent.
- The former expression-times-weight proxy is retained only as an audit comparator and is not used for the displayed hotspots.
- Mean pairwise map correlations before and after composition adjustment are:
""" + "\n".join(correlation_lines) + f"""
- Across the 12 displayed residual maps, six-nearest-neighbour Moran's I ranges from {moran['morans_i'].min():.3f} to {moran['morans_i'].max():.3f}; the largest BH-adjusted q-value from 999 one-sided permutations is {moran['bh_q_across_12_maps'].max():.3g}.
- Moran's I uses a directed, non-symmetrized binary 6-NN graph from scipy.spatial.cKDTree.query(k=7), with self dropped and distance ties following the returned source-row index order. Each row has six unit weights, so uniform row standardization would leave I unchanged.
- The BH adjustment is conditional on the retained four-pair display panel and does not account for candidate selection.
- The anterior, posterior, and anterior-replicate sagittal planes are interpreted independently. No coordinate normalization, map warping, or cross-section spatial correlation is used.
- Candidate anatomical labels are broad morphology-guided interpretations checked against the Allen adult mouse brain reference atlas and gene-distribution literature.
- No deformable registration to the Allen Common Coordinate Framework was performed. Labels therefore are candidate territories, not exact atlas annotations or ground-truth region calls.
- Olfm1 and Cck show gene-level regional agreement with published distributions. Enpp2 and Bex2 show only partial region-level agreement, and their displayed cell-type pairings are not presented as canonical marker validation.
- SVG/PDF retain vector text; PNG/TIFF are high-resolution raster exports. The H&E images are displayed without contrast enhancement or geometric transformation.
"""
    (FIG_DIR / "fig6_mouse_region_qa_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    genes = [gene for gene, _ in PAIRS]
    loaded = {section: load_section(section, folder, genes) for section, folder in SECTIONS.items()}

    adjusted = {}
    moran_rows = []
    source_frames = []
    for section_index, (section, (frame, _, _)) in enumerate(loaded.items()):
        x = frame["x"].to_numpy(dtype=float)
        y = frame["y"].to_numpy(dtype=float)
        for pair_index, (gene, cell_type) in enumerate(PAIRS):
            values = composition_adjusted_values(frame, gene, cell_type)
            adjusted[(section, gene)] = values
            observed, p_value = moran_knn(
                x,
                y,
                np.asarray(values["residual_z"]),
                seed=20260807 + 100 * section_index + pair_index,
            )
            moran_rows.append(
                {
                    "gene": gene,
                    "cell_type": cell_type,
                    "section": section,
                    "n_spots": len(frame),
                    "knn_k": 6,
                    "permutations": 999,
                    "weight_definition": "directed binary kNN; w_ij=1 when j is returned among i's k nearest neighbours; not symmetrized",
                    "row_standardized": False,
                    "tie_breaking": "scipy.spatial.cKDTree.query return order on released source-row coordinates after dropping self",
                    "scipy_version": scipy.__version__,
                    "morans_i": observed,
                    "one_sided_permutation_p": p_value,
                }
            )
            source_frames.append(
                pd.DataFrame(
                    {
                        "gene": gene,
                        "cell_type": cell_type,
                        "section": section,
                        "spot_id": frame["spot_id"],
                        "x": frame["x"],
                        "y": frame["y"],
                        "library_size": frame["library_size"],
                        "log_cp10k": values["expression"],
                        "cell_type_weight": values["weight"],
                        "composition_intercept": values["intercept"],
                        "composition_slope": values["slope"],
                        "fitted_composition_component": values["fitted"],
                        "composition_adjusted_residual": values["residual"],
                        "composition_adjusted_z": values["residual_z"],
                        "residual_percentile": values["residual_percentile"],
                        "legacy_weighted_expression_proxy": values["legacy_proxy"],
                    }
                )
            )

    moran = pd.DataFrame(moran_rows)
    moran["bh_q_across_12_maps"] = bh_adjust(moran["one_sided_permutation_p"].to_numpy(dtype=float))
    moran_lookup = moran.set_index(["section", "gene"])["morans_i"].to_dict()

    similarity_rows = []
    for section in SECTIONS:
        for left_index, (left_gene, _) in enumerate(PAIRS):
            for right_gene, _ in PAIRS[left_index + 1 :]:
                left = adjusted[(section, left_gene)]
                right = adjusted[(section, right_gene)]
                similarity_rows.append(
                    {
                        "section": section,
                        "gene_left": left_gene,
                        "gene_right": right_gene,
                        "legacy_proxy_pearson_r": np.corrcoef(left["legacy_proxy"], right["legacy_proxy"])[0, 1],
                        "legacy_proxy_spearman_r": pd.Series(left["legacy_proxy"]).corr(pd.Series(right["legacy_proxy"]), method="spearman"),
                        "adjusted_residual_pearson_r": np.corrcoef(left["residual_z"], right["residual_z"])[0, 1],
                        "adjusted_residual_spearman_r": pd.Series(left["residual_z"]).corr(pd.Series(right["residual_z"]), method="spearman"),
                    }
                )
    similarity = pd.DataFrame(similarity_rows)

    rows = []
    fig, axes = plt.subplots(len(PAIRS), len(SECTIONS), figsize=(7.2, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.91, top=0.92, bottom=0.035, wspace=0.05, hspace=0.15)
    cmap = mpl.colormaps["magma"].copy()
    cmap.set_under((0, 0, 0, 0))
    scatter = None

    for column, section in enumerate(SECTIONS):
        frame, image, scale = loaded[section]
        x = frame["x"].to_numpy(dtype=float) * scale
        y = frame["y"].to_numpy(dtype=float) * scale
        xmin, xmax, ymin, ymax = crop_bounds(frame, scale, image)
        for row, (gene, cell_type) in enumerate(PAIRS):
            axis = axes[row, column]
            values = np.asarray(adjusted[(section, gene)]["residual_percentile"])
            axis.imshow(image, origin="upper")
            axis.scatter(x, y, s=2.0, c="#777777", alpha=0.10, linewidths=0, rasterized=True)
            scatter = axis.scatter(
                x,
                y,
                s=7.0,
                c=values,
                cmap=cmap,
                vmin=0.90,
                vmax=1.0,
                alpha=np.where(values >= 0.90, 0.92, 0.0),
                linewidths=0,
                rasterized=True,
            )
            axis.set_xlim(xmin, xmax)
            axis.set_ylim(ymax, ymin)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#BDBDBD")
                spine.set_linewidth(0.45)
            if row == 0:
                axis.set_title(SECTION_LABELS[section], fontsize=6.8, fontweight="bold", pad=3)
            if column == 0:
                axis.text(-0.09, 0.63, gene, transform=axis.transAxes, ha="right", va="center", fontsize=7.0, fontweight="bold")
                axis.text(-0.09, 0.46, cell_type, transform=axis.transAxes, ha="right", va="center", fontsize=5.4, color="#4D4D4D")

            axis.text(
                0.98,
                0.97,
                f"I={moran_lookup[(section, gene)]:.2f}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=4.6,
                color="#202020",
                bbox={"boxstyle": "square,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )

            region, confidence, evidence = REGION_INTERPRETATIONS[(gene, section)]
            axis.text(
                0.02,
                0.035,
                f"Candidate: {region}",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=4.6,
                color="#202020",
                bbox={"boxstyle": "square,pad=0.22", "facecolor": "white", "edgecolor": "#BDBDBD", "alpha": 0.86, "linewidth": 0.35},
            )

            top = values >= 0.90
            rows.append(
                {
                    "gene": gene,
                    "cell_type": cell_type,
                    "section": section,
                    "top_decile_n": int(top.sum()),
                    "top_decile_centroid_x_fullres": float(frame.loc[top, "x"].mean()),
                    "top_decile_centroid_y_fullres": float(frame.loc[top, "y"].mean()),
                    "candidate_region": region,
                    "annotation_confidence": confidence,
                    "literature_anchor": evidence,
                    "morans_i": moran_lookup[(section, gene)],
                }
            )

    colorbar_axis = fig.add_axes([0.93, 0.35, 0.012, 0.30])
    colorbar = fig.colorbar(scatter, cax=colorbar_axis)
    colorbar.set_label("Composition-adjusted residual percentile", fontsize=5.8)
    colorbar.ax.tick_params(labelsize=5.2, length=2)
    fig.suptitle("Section-specific composition-adjusted hotspots and candidate anatomical territories", fontsize=8.4, fontweight="bold", y=0.985)

    pd.DataFrame(rows).to_csv(SOURCE_DIR / "fig6_mouse_region_hotspot_summary.csv", index=False)
    pd.concat(source_frames, ignore_index=True).to_csv(SOURCE_DIR / "fig6_mouse_spot_values.csv", index=False)
    similarity.to_csv(SOURCE_DIR / "fig6_mouse_row_similarity_audit.csv", index=False)
    moran.to_csv(SOURCE_DIR / "fig6_mouse_moran_audit.csv", index=False)
    manifest_path = SOURCE_DIR / "cross_section_maps_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update(
        {
            "mouse_sections": list(SECTIONS),
            "mouse_display": "native H&E with top within-section decile of residual from log(CP10K+1) regressed on the paired NNLS cell-type weight; no image warping or cross-section alignment",
            "mouse_spatial_audit": "6-nearest-neighbor Moran's I with 999 one-sided permutations and BH correction across the 12 displayed residual maps",
            "moran_weight_definition": "directed non-symmetrized binary 6-NN from scipy.spatial.cKDTree.query(k=7), self dropped; ties follow returned source-row index order; fixed out-degree means uniform row standardization leaves I unchanged",
            "legacy_mouse_diagnostics_not_used_in_manuscript": [
                "fig6_mouse_spatial_surfaces.csv (legacy smoothed weighted-proxy surface)",
                "fig6_mouse_candidate_concordance.csv (legacy cross-section concordance selection)",
            ],
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_qa_notes(moran, similarity)
    export_figure(fig, "fig6_mouse_spatial_coherence")


if __name__ == "__main__":
    main()
