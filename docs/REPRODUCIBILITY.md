# Reproduction Notes

## Data Restoration

Download the six logical archives from the data directory, including `shared_provenance.zip` and all multipart pieces. Run `scripts/restore_data.py` from the repository. The tool verifies the joined ZIP against `provenance/DATA_PACKAGES.json`, verifies every restored member against its SHA-256 inventory and refuses to overwrite a different existing file.

The download packages preserve their original scientific paths under `04_数值实验/` and `05_论文草稿/`. The public source has been reorganized under `analysis/`. Code paths have been adjusted to read those unchanged inputs. Original data filenames and archived execution manifests have not been rewritten. These local data directories are ignored by Git.

The restoration command creates the internal figure alias `04_数值实验/overleaf` pointing to `05_论文草稿/latex`. Pinned R adapters in `analysis/adapters/` are copied into the historical launch locations under the restored results directory. These are independent working copies, so generating a new adapter cannot overwrite its pinned source.

## Numerical Identity

The following numerical cores are preserved byte-for-byte because cached fits validate their exact hashes:

- `analysis/code/estimation/nb_core.py`
- `analysis/code/inference/joint_refit.py`
- `analysis/code/inference/weighted_information.py`

The directory cleanup only adjusts source locations and path wiring in callers. Other earlier publication-copy changes reduced ordinary comments or module help text. Function documentation, licenses and attribution are retained. R sources are byte-identical. The source inventory records both original and current source hashes. Archived result manifests retain their historical identities.

## Running and Checking

See `ANALYSES.md` for the analysis entrypoints. Running a fit or plotting command can overwrite a restored result or figure. Use a separate working copy for such runs. Unit tests check focused numerical and file-restoration behavior. They do not rerun every manuscript experiment.

Expensive analyses use checkpoints. Input-array, configuration or numerical-core mismatches produce explicit failures instead of silently reusing incompatible fits.

Official R methods require the upstream revisions recorded in `provenance/UPSTREAM_SOURCES.csv`. The launch scripts expect an isolated R environment and source directories under `08_其他/运行环境/distrisvg_2026_09_25/`. That runtime is not bundled. The environment and R session records describe the original setup.

The earlier integer-field reference export and initialization-only RDS files are excluded from the public data. Reference-dependent results use the real-valued `reference_counts_rsem.mtx`. Historical manifests can still mention excluded debugging files. Those entries do not identify missing scientific inputs.
