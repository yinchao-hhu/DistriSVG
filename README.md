# DistriSVG

Research code for distributed sparse estimation and inference of shared cell-type-specific spatially variable genes.

## Data

Data are organized into Simulation, PDAC and Mouse brain cases in the [data directory](https://drive.google.com/drive/folders/11DNp5epBPN97qs5Yerl3f8h0AiVf1yDr). Download each case archive and `shared_provenance.zip`. To restore the original paths, run:

```sh
python scripts/restore_data.py /absolute/path/to/downloaded/data
```

The shared archive contains the original cross-dataset summary tables and run identities. Filtered case-specific views are included for convenient inspection; they do not replace the original tables. Per-file SHA-256 inventories accompany all archives.

## Code Organization

The original relative paths are retained to avoid changing analysis inputs or import behavior.

| Directory under `04_数值实验/code/` | Role |
| --- | --- |
| `distrisvg_revision_2026_09_18/` | Canonical negative-binomial model, AP-FedAdam and FedAvg, sequencing-depth sensitivity and evaluation metrics |
| `distrisvg_completion_2026_09_25/` | All-variable joint refit, nuisance-adjusted covariance, calibration and full-panel real-data analyses |
| `workflow_pooled_2026_10_26/` | Matched pooled inference replay on the same complete-input calibration data |
| `figure_scripts/` | Figure generation, reference deconvolution, official-method adapters and retained historical analyses |
| `new_test1-5/`, `missing_celltype_test/` | Retained large-scale simulation generators and fitting scripts |
| `pdac_data_to_yc/` | Archived PDAC fitting and postprocessing |

The numerical optimizer, inferential refit and descriptive spatial maps are separate analyses. A plot regenerated from saved CSVs is not a refitted model. Scripts retained for historical comparisons are identified in `docs/FIGURE_MAP.md`.

## Environment and Checks

The recorded Python package versions are in `environment/requirements-frozen.txt`. The explicit R environment is specific to macOS Apple Silicon. R session information and pinned upstream repository revisions are included separately. These records describe the environment used for the retained results, not a verified universal installer.

```sh
python -m unittest discover -s 04_数值实验/code/distrisvg_revision_2026_09_18 -p 'test_*.py' -v
python -m unittest discover -s 04_数值实验/code/distrisvg_completion_2026_09_25 -p 'test_*.py' -v
python 04_数值实验/code/workflow_pooled_2026_10_26/test_pooled.py
```

The pooled tests require the restored simulation source data. See `docs/REPRODUCIBILITY.md` before running result-writing or computationally expensive scripts.

## Provenance

`provenance/SOURCE_MANIFEST.csv` links original and publication-copy source hashes. Ordinary explanatory comments have been reduced without changing executable Python syntax. The numerical cores used to validate cached results and the pinned R sources remain byte-identical. Licenses, attribution, relevant function documentation and necessary numerical explanations are retained.

Upstream software is not relicensed or bundled as newly authored code. See `provenance/UPSTREAM_SOURCES.csv`. An author-approved reuse license for this repository has not been selected; public access alone does not grant a software license.

## Interpretation

The implementation emulates distributed likelihood calculations locally. It is not a transport-security or privacy implementation. Complete-input inferential calibration does not establish universal false-discovery control. Real-data rankings and composition-adjusted expression maps do not establish cell-type ground truth. Mouse sections are unregistered and the reference is from visual cortex, not a matched whole-brain atlas.
