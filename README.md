# DistriSVG

Research code for distributed sparse estimation and inference of shared cell-type-specific spatially variable genes.

## Data

Scientific data are distributed separately in the [Simulation, PDAC and Mouse brain data folders](https://drive.google.com/drive/folders/11DNp5epBPN97qs5Yerl3f8h0AiVf1yDr). This repository contains code, environment records and data checksums, not raw expression matrices or collaborator-transfer packages.

Download the case archives and `shared_provenance.zip`, keeping all `*.zip.part-*` pieces together, then run:

```sh
python scripts/restore_data.py /absolute/path/to/downloaded/data
```

The restore tool joins multipart archives, checks SHA-256 values and restores their frozen input paths. It also installs local compatibility copies of the pinned R adapters. Restored data and generated results are excluded from version control. `provenance/UPLOAD_PARTS.csv` lists the download pieces.

## Code Organization

| Directory | Purpose |
| --- | --- |
| `analysis/code/estimation/` | Negative-binomial model, AP-FedAdam and FedAvg, depth sensitivity and evaluation |
| `analysis/code/inference/` | All-variable refit, nuisance-adjusted covariance, calibration and real-data analyses |
| `analysis/code/pooled_comparison/` | Matched pooled inference on the complete-input calibration data |
| `analysis/code/simulation_complete/` | Large-scale simulation generator and complete-panel fitting |
| `analysis/code/simulation_missing/` | Large-scale incomplete-panel fitting |
| `analysis/code/pdac/` | Archived PDAC fitting across the five basis families |
| `analysis/code/figures/` | Figures, reference deconvolution and additional benchmark entrypoints |
| `analysis/adapters/` | Pinned R adapters used for official-method comparisons |

See [the manuscript display map](docs/FIGURE_MAP.md) for figure and table sources, [analysis instructions](docs/ANALYSES.md) for entrypoints, and [reproduction notes](docs/REPRODUCIBILITY.md) for input and environment requirements. All five spatial basis families are retained.

## Environment and Tests

Python package versions are recorded in `environment/requirements-frozen.txt`. The explicit R environment targets macOS Apple Silicon. R session records and upstream revisions are included separately. These describe the retained results' environment, not a universal installer.

```sh
python -m unittest discover -s analysis/code/estimation -p 'test_*.py' -v
python -m unittest discover -s analysis/code/inference -p 'test_*.py' -v
python analysis/code/pooled_comparison/test_pooled.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

The pooled tests require restored simulation data. Result-writing scripts can overwrite local results, so rerun them in a separate working copy.

## Provenance and Scope

`provenance/SOURCE_MANIFEST.csv` records the canonical source paths and source hashes. The optimizer, joint refit and weighted-information numerical cores are byte-identical to the archived versions. Pinned R adapters and upstream attribution are preserved. The public directory cleanup changes file organization and input-path wiring, not the statistical calculations or released data.

Upstream revisions are listed in `provenance/UPSTREAM_SOURCES.csv`. An author-approved software reuse license has not been selected. Public access alone does not grant a software license.

The implementation evaluates distributed likelihood calculations locally. It does not implement secure transport. Optimization, independent statistical inference and descriptive spatial maps are distinct components. Real-data rankings and composition-adjusted expression maps are not cell-type ground truth. Mouse sections are unregistered and the single-cell reference is from visual cortex rather than a matched whole-brain atlas.
