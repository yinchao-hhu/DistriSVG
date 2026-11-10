# Joint-refit and official-method completion

Version: 2026-09-25. This directory adds versioned experiments without replacing
the earlier fitted results, figures or manuscripts. The analysis contract is
`.ai-collab/rounds/round-016-plan.md` at the project root.

## Scientific scope

- Canonical NB2 means, all-variable unpenalized gene-wise refits, estimated
  log-dispersion, full nuisance covariance and two-coordinate Wald blocks.
- Exact local sums reproduce the same weighted pooled criterion at a common
  parameter. This is a local protocol emulator, not network/security software.
- Both information-based and independent-spot HC1 covariances are reported.
  Unequal section weights require squared weights in the covariance meat.
- Five-family Cauchy and Bonferroni combinations, followed by BH or BY, are all
  retained. Dispersion-boundary, nonstationary and nonidentifiable fits are
  flagged with conservative p=1, not silently removed.
- Missing compact inputs delete proportions after generating counts. Keeping
  that site's incomplete likelihood is misspecification, not absent cells.
- The new 100 null and 50 sparse response replicates are conditional on one
  retained coordinate/composition design. They do not model spatially correlated
  noise or reference-weight uncertainty and do not prove universal FDR control.
- Official reference workflows preserve the GSE71585 fractional RSEM read-count
  scale. These are not UMI reference counts, TPM or pseudo-cells. The visual-
  cortex reference is not anatomically matched to every sagittal territory.
- Old maps remain descriptive adjusted observed expression, not fitted effects
  or registered anatomy. No real-data accuracy follows from candidate counts.

## Main entry points

Run commands from the project root. The existing Python environment is
`08_其他/运行环境/.venv/bin/python`; the isolated R executable is
`08_其他/运行环境/distrisvg_2026_09_25/r/bin/Rscript`.

1. Run `python -m unittest discover -s <this-directory> -p 'test_*.py' -v`.
2. Run `python <this-directory>/run_completion.py archive --workers 2`.
3. Run `python <this-directory>/run_completion.py calibration --workers 2`.
4. Run `python <this-directory>/run_completion.py real --workers 2`.
5. Once real fits finish, run
   `python <this-directory>/postprocess_weighted_real.py --workers 2`.
   This uses cached fitted parameters; it does not repeat or change optimization.
6. Run `python <this-directory>/prepare_mouse_official.py`, then
   `Rscript --vanilla <this-directory>/run_official_mouse.R rctd`, and similarly
   replace `rctd` by `cside` and `mmm`. An optional second argument is a comma-
   separated list of `Anterior,Posterior,Anterior2` for resumable section runs.
7. Run `python <this-directory>/prepare_competitors.py`, then
   `python <this-directory>/run_competitors.py`. This calls unchanged official
   statistical routines through recorded copies of the archived adapters.
   Run `audit_spvc_warnings.py` to retain immediate warning output and the
   unseeded replay differences, then `run_seeded_spvc.py` for the prespecified
   date-seeded run and identical replay used in the final comparison table.
8. Run `python <this-directory>/summarize_completion.py` and
   `python <this-directory>/finalize_results.py real`, `competitors`, `official`
   only after the respective output audits show complete runs.
9. `audit_large_generator.py` replays current generator defaults without calling
   the original output-writing entrypoint. It does not replace archived counts.

Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and
`VECLIB_MAXIMUM_THREADS=1` for these runs. For R, also put the isolated `r/bin`
directory first on PATH and set `RGL_USE_NULL=TRUE`. This is needed when official
adapters compile Rcpp source. No global R/Python environment is modified.
`DISTRISVG_MMM_WORKERS` changes only the official MMM worker count (default 1).
CTSV uses two workers after C-SIDE completes; all concurrent workflows together
must remain within six single-thread compute workers on this host.
The R conda explicit package list and Python package snapshot are in the results
directory. Official repositories and source-package installation logs are
under the isolated runtime. The core Python files also depend on the preserved
`distrisvg_revision_2026_09_18` helpers, so this directory alone is not a complete
standalone source package.

## Output contract

Outputs are under `04_数值实验/results/distrisvg_completion_2026-09-25`.

| Files | Meaning |
|---|---|
| `run_manifest.json`, `refits/*.json` | Input, source and configuration identities |
| `archive_refit_*` | All original 160-gene fits; separately re-adjusted common-40 panel |
| `calibration/*_input.npz` | Actual generated counts and baseline/spatial truth |
| `calibration/*_audit.csv` | Every estimated-dispersion gene/family fit, including failures |
| `refits/*knownpsi*` | Additional known-dispersion linear checks and their audits |
| `calibration_metrics.csv`, `calibration_summary.csv` | Per-response metrics and Monte Carlo uncertainty |
| `ap_fits/*` | Full-panel canonical AP coefficients, histories and exact input identities |
| `real_both_*` | All full-panel scores and diagnostics; inverse-Hessian arm is audit-only for unequal weights |
| `weighted_information/*`, `real_weighted_*` | Corrected unequal-weight information covariance and scores |
| `official_mouse/inputs` | Genuine reference, integer spatial counts, labels and provenance |
| `official_mouse/rctd` | Formal fitted weights, coverage and agreement, not composition truth |
| `official_mouse/cside`, `official_mouse/mmm` | Official per-section tests and eligibility/failure records |
| `official_competitors` | Pinned source versions, adapters, new official outputs and logs |
| `large_generator_provenance_*` | Current-source versus historical-input audit; no overwritten inputs |

The literal CSV value `null` is a scenario name. Pandas treats it as missing by
default; analysis code explicitly restores `signal.fillna('null')` before
grouping. Empty numerical entries elsewhere represent undefined metrics, not
zero. `real_all_metrics.csv` contains both inferential candidates and preserved
diagnostic arms, so always filter by method, covariance, offset and adjustment.

## Numerical and input safeguards

The refit uses at most 500 L-BFGS-B iterations; final projected gradient must be
below 2e-5. Dispersion-boundary solutions do not receive interior Wald values.
Cached chunks validate their inputs and mathematical core hashes. Coordinate
normalization and full-library offsets follow the frozen new-run convention;
they are not retroactive statements about the old real-data fits.

The first reference MatrixMarket export incorrectly used an integer field. Its
RCTD run was interrupted before completed scientific output, and its debug
matrix/log/initial object are retained only for audit. The valid pipeline reads
`reference_counts_rsem.mtx` with real-valued entries and checks library totals
to relative tolerance 1e-10. Never use `reference_counts.mtx` for a result.
Archived NNLS row sums differ from one by at most a small epsilon; official
input validation uses the same 1e-6 tolerance as `spacexr::import_weights`.

Manuscript tables and the calibration figure are generated from completed
machine-readable outputs. Compilation/render checks are separate from numerical
validation. Independent theory review, resolved missing-input inference and
public archival are not supplied by these scripts.
