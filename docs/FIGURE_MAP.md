# Manuscript and Data Map

This map follows the current local NC manuscript labels, not historical figure numbers embedded in filenames. The manuscript sources themselves are not included in the code repository. Code entries are relative to `analysis/code/`. Unprefixed inference scripts in the table are under `inference/`.

| Display or analysis | Source data | Main code |
| --- | --- | --- |
| Workflow schematic, `fig:distrisvg_workflow` | Schematic maps; coefficient MAE from the revision benchmark; inference metrics from the matched pooled replay | Illustrative artwork is not a measured data product. Numerical source files are retained in the Simulation case. |
| Inferential calibration, `fig:refit-calibration` | `results/distrisvg_completion_2026-09-25/calibration/`, `calibration_metrics.csv`, `calibration_summary.csv` | `inference/run_completion.py`, `joint_refit.py`, `summarize_completion.py` |
| Main common-panel method table | `fresh_competitor_pairs.csv`, `fresh_competitor_metrics.csv`, `fresh_competitor_summary.csv`, `official_competitors/` | `prepare_competitors.py`, `run_competitors.py`, `run_seeded_spvc.py`, `finalize_results.py` |
| PDAC marker ranking and enrichment | `pdac_basis_consensus_candidates.csv`, `pdac_marker_annotation_summary.csv`, `pdac_top20_marker_annotations.csv`, all five archived PDAC basis outputs | `figures/plot_pdac_validation.py`; the retained ranking is exploratory |
| PDAC spatial figure, `fig:pdac-spatial` | `cross_section_spatial_maps/source_data/fig5_pdac_spot_values.csv`, `fig5_pdac_moran_audit.csv`, PDAC processed expression and SPOTlight proportions | `figures/make_cross_section_spatial_maps.py` |
| Mouse spatial figure, `fig:mouse-brain` | `fig6_mouse_spot_values.csv`, `fig6_mouse_moran_audit.csv`, `fig6_mouse_region_hotspot_summary.csv`, raw Visium counts, coordinates and histology | `figures/make_mouse_region_overlays.py` |
| Mouse NNLS reference weights | Three Visium sections, two GSE71585 reference files, `mouse_reference_deconv/` | `figures/run_mouse_reference_deconvolution.py` |
| Supplementary FedAvg and coordinate sensitivity | `simulation_robustness/source_data/`, `centralized_baseline/` | `figures/run_compact_benchmark.py`, `figures/run_centralized_baseline.py`, `figures/replot_fig3_fig4.py` |
| Supplementary depth sensitivity | `results/distrisvg_revision_2026-09-18/` | `estimation/run_revision.py`, `estimation/run_real_offset.py`, `estimation/build_outputs.py` |
| Supplementary full-panel real-data sensitivity | Per-case AP fits, refits and weighted-information files; original `real_*` tables in the shared archive | `inference/run_completion.py`, `postprocess_weighted_real.py`, `finalize_results.py` |
| Official mouse comparisons | `official_mouse/inputs/`, `rctd/`, `cside/`, `mmm/` | `prepare_mouse_official.py`, `run_official_mouse.R` |
| Matched pooled inference replay | `results/workflow_pooled_2026-10-26/` | `pooled_comparison/run_pooled.py` |
| Historical G=1000 coefficient and ROC displays | All CSVs in `new_test1-5/` and `missing_celltype_test/`, `simulation_detection/`, all five basis families | `figures/regen_recovery_figs_G1000.py`, `figures/run_simulation_detection_roc.py`; these are not the current main calibration figure |

Paths beginning with `results/` are under `04_数值实验/`; figure source-data paths are under `05_论文草稿/latex/figs/`. Current main spatial figures have historical filenames `fig5_pdac_spatial_coherence` and `fig6_mouse_spatial_coherence`.
