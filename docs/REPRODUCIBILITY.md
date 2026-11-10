# Reproduction Notes

## Restoring Data

Download all six data archives, including every `*.zip.part-*` piece and `shared_provenance.zip`, and run `scripts/restore_data.py` from this repository. Large archives are divided only for transfer. The restore tool joins their parts, checks the complete archive hash against `provenance/DATA_PACKAGES.json`, checks each member against its SHA-256 inventory and refuses to overwrite a different existing file. It ignores archive metadata under `_release/`.

The scientific scripts retain their original paths. The restore tool creates the internal alias `04_数值实验/overleaf` to `05_论文草稿/latex` for historical plotting scripts. The alias points within the repository, not to an author-specific external directory.

## Numerical Components

The following source files are preserved byte-for-byte because result caches identify their exact hashes:

- `distrisvg_revision_2026_09_18/nb_core.py`
- `distrisvg_completion_2026_09_25/joint_refit.py`
- `distrisvg_completion_2026_09_25/weighted_information.py`

Other publication-copy Python files differ only in ordinary comments or shortened module help text. Their executable ASTs are checked against the original sources. Function docstrings, pragmas, licenses and attribution are retained. R adapters and third-party/pinned R files remain unchanged.

The archived manifests keep their original source hashes. They have not been rewritten to misrepresent historical execution. The source manifest provides the original-to-publication mapping.

## Running Analyses

The code records original output locations. Running an analysis or plotting command can overwrite a restored result or figure. Make a separate work copy before rerunning result-writing entrypoints. Unit tests do not run the complete retained experiments.

The main calibration entrypoint is `distrisvg_completion_2026_09_25/run_completion.py`. Its `archive`, `calibration` and `real` subcommands fit different targets. Expensive analyses use checkpoints; input or numerical-core mismatches cause an explicit failure rather than silent reuse.

For official R methods, clone the recorded upstream sources at the commits listed in `provenance/UPSTREAM_SOURCES.csv`. The original launch scripts expect the corresponding isolated R environment and source directories under `08_其他/运行环境/distrisvg_2026_09_25/`. That environment is not vendored in this repository. Follow the original analysis READMEs and environment records rather than assuming a system R installation is equivalent.

The earlier integer-field reference export and initialization-only RDS files are excluded from the public data packages. Valid reference-dependent results use the real-valued `reference_counts_rsem.mtx`. The original manifest also lists excluded debug files for historical provenance; this is not a missing scientific input.
