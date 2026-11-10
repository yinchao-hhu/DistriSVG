# DistriSVG revision experiments, 2026-09-18

## Scope

This directory adds experiments; it does not overwrite any archived fit, input,
figure, or dated manuscript. The new canonical NB mean implementation is separate
from the legacy inverse-mean PyTorch code. The old implementation is not silently
relabelled as corrected.

Completed:

- Recomputed detection metrics for every retained G=1000 generating pattern
  (linear, Gaussian, cosine), complete/missing inputs, and joint/matched scores.
- Recomputed CTSV, STANCE, Celina, spVC and AP-FedAdam metrics on the same compact
  40-gene panel. Complete inputs: five methods, three repeats. Missing inputs:
  the two methods with available archived outputs, three repeats.
- 198 fresh simulation fits on existing 160-gene inputs: complete/missing
  five-family fits and linear 90-degree coordinate stress, two offset choices,
  AP-FedAdam/FedAvg/pooled, three repeats. This produces 54 pair-level evaluations.
- 20 fresh real-data sensitivity fits: PDAC/mouse, five families, two offsets,
  fixed 160-gene panels. These are bounded pilots, not replacements for the full
  real-data analyses and not new deconvolutions.
- Eight metric/distribution/gradient/block-availability unit tests.

Not completed: new official R-method runs, C-SIDE, MMM, full-scale real-data
replacement, null calibration, or post-selection-valid refitting. The official
replay preflight reports the missing R environment. No artificial reference or
single-cell labels are manufactured to make an incompatible method run.

## Reproduction

Run from the project root using Python 3.10 and the versions recorded in
`04_数值实验/results/distrisvg_revision_2026-09-18/run_manifest.json`.
The existing project Python is `08_其他/运行环境/.venv/bin/python`.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/private/tmp/distrisvg-mpl
PY=08_其他/运行环境/.venv/bin/python
CODE=04_数值实验/code/distrisvg_revision_2026_09_18
$PY "$CODE/test_revision.py"
$PY "$CODE/run_revision.py" --resume
$PY "$CODE/run_real_offset.py" --resume
$PY "$CODE/build_outputs.py"
$PY "$CODE/competing_methods.py"
```

`--resume` verifies the numerical-core hash, fit settings, and array hashes before
reusing a coefficient file. It does not refit completed jobs or relabel cached
time as fresh execution time. Omitting it refuses to overwrite existing fits.
Use a separate versioned output directory for changed configurations.

The old R adapters can be replayed into a new `official_replay/<method>` folder
after installing R and their official dependencies. Example:

```bash
$PY "$CODE/competing_methods.py" --run CTSV
$PY "$CODE/competing_methods.py" --run STANCE --official-source /path/to/STANCE
```

The replay branch has not been executed on this machine because R is absent.
It is an execution adapter, not a claim of a new successful official run.
The preflight alone was executed and saved. It never installs packages or
overwrites the original official outputs. For Celina and STANCE the source path
is the repository root; for spVC it is the repository root containing `R/`.

## Input / output contract

Simulation inputs are the existing `repeat*_counts.csv` (genes by spots),
`coordinates.csv` (spot IDs, client labels, x, y), `omega.csv` (spots by types),
and `truth.csv` (gene/type IDs, two coefficients, binary truth). Spot order and
the common pair universe are checked explicitly. Raw source SHA-256 values are
recorded. No counts are generated in this revision.

`pair_metrics` uses q<0.05, power, precision, empirical FDP, specificity, F1,
tie-aware AUROC and non-interpolated average precision. Undefined power,
precision, F1 or AUC is NaN; zero-discovery FDP is zero. Invalid numeric values
raise an error. Metrics are summarized over independent repeats, never over
dependent gene/type pairs as though they were independent datasets.

Core design order is `[omega, omega*h1, omega*h2]`; coefficient rows follow the
same order. Fixed dispersion is 12. The optional offset is
`log(full_input_library / local_median_library)`, computed before gene-panel
selection. Local means/SDs of counts are not substituted for an audited external
exposure. Simulation counts had no separately imposed exposure variation.

Real-data panels: PDAC is a seed-20260918 uniform selection from 4740 genes;
mouse is the first 160 genes of the existing reference-selected list. No genes
were selected for attractive output. All 1819 PDAC and 8875 mouse spots are used.
Both pilot arms normalize omega rows, use global coordinate standardization,
fixed dispersion, equal site weights and 150x5 updates. These pilot settings
differ from the archived real-data fits; only the offset differs *between arms*.

Files in the results directory include:

- `archive_*metrics*.csv`: recomputation of retained official/large-scale outputs.
- `offset_simulation_pairs.csv`, `offset_simulation_per_repeat.csv`, and summary:
  54 evaluations from the new simulation fits.
- `real_offset_pairs.csv`, summary, rank sensitivity, gene panels and input audit.
- `fits/*.npz`: all 218 new coefficient matrices.
- `fits/*.json`: settings, array/core hashes, training checks and measured time.
- `run_manifest.json`: source-file hashes, versions and configuration.
- `competitor_preflight.json`: explicit runtime/input gaps.

New tables and figure exports are under the dated `tables/` and `figs/`
subdirectories in the LaTeX project. The eight unit tests are implementation
checks, not mathematical proof review or evidence of FDR control.

## Interpretation

The framework name does not change the model or create a new discovery guarantee.
Cauchy aggregation integrates evidence; it is not a majority vote or a requirement
for repeated support across all bases. A pooled fixed-budget fit is a reference,
not an oracle. The observed-library offset changes real-data rankings, but does
not by itself calibrate penalised reciprocal-diagonal Wald diagnostics.

The new real-data pilot has five families; the archived main mouse result still
has only one cosine family. Do not confuse these analyses. No source atlas,
registration, privacy mechanism, or independent biological truth was added.
