# Analysis Entrypoints

All paths below are relative to the repository root. Restore the data first. Commands that fit models, prepare adapters or generate figures write to the frozen analysis locations. Run them in a separate working copy to preserve the downloaded results.

## Parameter Estimation

```sh
python analysis/code/estimation/run_revision.py --resume
python analysis/code/estimation/run_real_offset.py --resume
python analysis/code/estimation/build_outputs.py
```

This benchmark compares AP-FedAdam and FedAvg, evaluates sequencing-depth sensitivity and summarizes the archived method results. The `--resume` option validates fit settings, array hashes and the negative-binomial numerical-core hash.

## Independent Inference

```sh
python analysis/code/inference/run_completion.py archive --workers 2
python analysis/code/inference/run_completion.py calibration --workers 2
python analysis/code/inference/run_completion.py real --workers 2
python analysis/code/inference/postprocess_weighted_real.py --workers 2
python analysis/code/inference/summarize_completion.py
python analysis/code/inference/finalize_results.py real
```

`archive` refits the retained compact simulation, `calibration` generates and fits the independent calibration responses, and `real` fits the PDAC and mouse panels. The weighted-information postprocessing corrects covariance calculations for unequal client weights. The joint refit is not a Wald test applied to thresholded optimizer coefficients.

The complete-input calibration assesses detection power and empirical false-discovery proportions under the retained data-generating design. Incomplete cell-type input analyses and real-data sensitivity are separate results.

## Official Comparisons

```sh
python analysis/code/inference/prepare_competitors.py
python analysis/code/inference/run_competitors.py
python analysis/code/inference/run_seeded_spvc.py
python analysis/code/inference/finalize_results.py competitors
python analysis/code/inference/finalize_results.py official
```

The method wrappers use the recorded CTSV, STANCE, Celina and spVC sources and preserve fit failures in their audit outputs. `analysis/adapters/` contains the pinned executed adapters. Preparing adapters creates new working copies for the current runtime.

Mouse reference-dependent comparisons begin with `analysis/code/inference/prepare_mouse_official.py` and `analysis/code/inference/run_official_mouse.R`. Consult the recorded R environments before invoking those routines.

## Archived Large-Scale Simulation and PDAC Fits

The renamed entrypoints are `analysis/code/simulation_complete/generate.py`, `analysis/code/simulation_complete/fit.py`, `analysis/code/simulation_missing/fit.py` and `analysis/code/pdac/fit.py`. They retain the original current-working-directory convention. Run them with the corresponding restored data directory as the working directory:

| Entrypoint | Working directory |
| --- | --- |
| Complete simulation generator or fit | `04_数值实验/code/new_test1-5/` |
| Incomplete-panel simulation fit | `04_数值实验/code/missing_celltype_test/` |
| Archived PDAC fit | `04_数值实验/code/pdac_data_to_yc/` |

These archived fits are distinct from the canonical refit calibration. The generator audit in `analysis/code/inference/audit_large_generator.py` compares source and historical inputs without overwriting the latter.

## Figures

`FIGURE_MAP.md` identifies current displays and their source data. `analysis/code/figures/` also retains historical analyses needed by those workflows. Historical figure numbers in script names are not current manuscript numbering. The latest workflow illustration is artwork, not an executable statistical analysis, and obsolete schematic drafts are not included here.
