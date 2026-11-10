"""Export existing mouse counts and genuine reference inputs for official R tools."""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

from run_completion import OUT, ROOT, old, write_manifest, initialize


def main():
    initialize()
    target = OUT / "official_mouse/inputs"
    target.mkdir(parents=True, exist_ok=True)
    reference_dir = ROOT / "04_数值实验/reference_data/mouse_brain"
    labels = old.read(reference_dir / "GSE71585_Clustering_Results.csv.gz")
    labels = labels[labels.pass_qc_checks.astype(str).str.upper() == "Y"].set_index("sample_title")
    counts = old.read(reference_dir / "GSE71585_RefSeq_counts.csv.gz", index_col=0)
    if counts.index.has_duplicates:
        counts = counts.groupby(counts.index, sort=False).sum()
    cells = [name for name in counts.columns if name in labels.index]
    labels = labels.loc[cells]
    label_values = labels.broad_type.fillna("Unclassified").astype(str)
    valid_types = label_values[~label_values.str.lower().str.contains("unclassified")].value_counts()
    valid_types = sorted(valid_types[valid_types >= 10].index)
    cells = label_values.index[label_values.isin(valid_types)]
    counts = counts.loc[:, cells]
    library = counts.sum(0)
    assert np.all(library >= 100)
    normalized = counts.divide(library, axis=1)
    profiles = pd.DataFrame({ct: normalized.loc[:, label_values.loc[cells] == ct].mean(1) for ct in valid_types})
    profiles.index.name = "gene"
    profiles.to_csv(target / "reference_profiles.csv")
    pd.DataFrame(dict(cell_id=cells, cell_type=label_values.loc[cells], nUMI=library)).to_csv(target / "reference_labels.csv", index=False)
    if not (target / "reference_counts_rsem.mtx").exists():
        mmwrite(target / "reference_counts_rsem.mtx", sparse.csc_matrix(counts.to_numpy()), field="real")
    pd.Series(counts.index, name="gene").to_csv(target / "reference_genes.csv", index=False)
    sections = {"Anterior": "V1_Mouse_Brain_Sagittal_Anterior", "Posterior": "V1_Mouse_Brain_Sagittal_Posterior", "Anterior2": "V1_Mouse_Brain_Sagittal_Anterior_Section_2"}
    audit = []
    for short, full in sections.items():
        directory = target / short
        directory.mkdir(exist_ok=True)
        path = old.track(ROOT / "04_数值实验/data" / full / "filtered_feature_bc_matrix.h5")
        with h5py.File(path, "r") as f:
            m = f["matrix"]
            matrix = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
            genes = pd.Index([x.decode() for x in m["features/name"][:]])
            barcodes = pd.Index([x.decode() for x in m["barcodes"][:]])
        omega = old.read(ROOT / "04_数值实验/figures/mouse_reference_deconv" / f"mouse_reference_nnls_omega_{short}.csv").set_index("spot_id")
        columns = barcodes.get_indexer(omega.index)
        assert np.all(columns >= 0)
        unique = pd.Index(genes.unique())
        mapping = sparse.csr_matrix((np.ones(len(genes)), (unique.get_indexer(genes), np.arange(len(genes)))), shape=(len(unique), len(genes)))
        collapsed = (mapping @ matrix[:, columns]).tocsc()
        totals = np.asarray(collapsed.sum(0)).ravel()
        if not (directory / "counts.mtx").exists():
            mmwrite(directory / "counts.mtx", collapsed, field="integer")
        pd.Series(unique, name="gene").to_csv(directory / "genes.csv", index=False)
        meta = omega[["x", "y"]].copy()
        meta["nUMI"] = totals
        meta.to_csv(directory / "spots.csv")
        omega[valid_types].to_csv(directory / "nnls_weights.csv")
        audit.append(dict(section=short, n_spots=len(columns), n_genes=len(unique),
                          duplicate_symbols_collapsed=len(genes) - len(unique), min_library=int(totals.min())))
    panel = old.read(ROOT / "04_数值实验/figures/mouse_reference_deconv/mouse_reference_nnls_gene_panel.csv")
    panel.iloc[:300].to_csv(target / "test_gene_panel.csv", index=False)
    pd.DataFrame(audit).to_csv(target / "input_audit.csv", index=False)
    manifest = dict(reference_cells=len(cells), reference_genes=len(counts), cell_type_counts=label_values.loc[cells].value_counts().to_dict(),
                    reference_count_scale="fractional RSEM estimated read counts, not UMI/TPM/RPKM",
                    reference_matrix="reference_counts_rsem.mtx",
                    excluded_debug_matrix="reference_counts.mtx used integer export; aborted before results; never use",
                    profile_rule="mean per-cell count / full reference-cell library; official get_cell_type_info convention",
                    reference_scope="visual cortex, not anatomically matched to every sagittal region",
                    spatial_rule="all retained spots; raw counts; duplicate gene symbols summed", test_genes=300,
                    files={str(p.relative_to(target)): old.digest(p) for p in target.rglob("*") if p.is_file() and p.name != "manifest.json"})
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_manifest()
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
