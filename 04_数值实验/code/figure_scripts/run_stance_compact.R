#!/usr/bin/env Rscript

# Run the official STANCE two-stage testing code on the compact benchmark.
# Stage 1 is fitted to all 160 genes. Stage 2 is evaluated on the fixed 40-gene
# panel; panel genes failing Stage 1 receive p=1 for every cell type.

suppressPackageStartupMessages({
  library(gaston)
  library(KRLS)
  library(Rcpp)
  library(RcppArmadillo)
})

args <- commandArgs(trailingOnly = TRUE)
stance_source <- if (length(args) >= 1) args[[1]] else "/private/tmp/method-src/STANCE-master"
source_dir <- if (length(args) >= 2) args[[2]] else file.path(
  "05_论文草稿", "latex", "figs", "simulation_method_comparison", "source_data"
)
repeat_ids <- if (length(args) >= 3) {
  as.integer(strsplit(args[[3]], ",")[[1]])
} else {
  0:2
}

Rcpp::sourceCpp(
  file.path(stance_source, "src", "matrix_inversion.cpp"), rebuild = FALSE
)
for (source_file in c(
  "set_class.R", "createSTANCEobject.R", "data_preprocess.R",
  "build_kernelMatrix.R", "runTest1.R", "runTest2.R"
)) {
  source(file.path(stance_source, "R", source_file), local = globalenv())
}

panel <- read.csv(
  file.path(source_dir, "official_method_gene_panel.csv"),
  check.names = FALSE
)$gene_id

score_summary <- function(frame, repeat_id, elapsed, stage1_selected) {
  active <- as.logical(frame$active)
  score <- -log10(pmax(frame$pair_p, 1e-300))
  selected <- frame$pair_bh_q < 0.05
  positives <- sum(active)
  negatives <- sum(!active)
  ordering <- order(score, decreasing = TRUE)
  ranked_active <- active[ordering]
  tp_curve <- cumsum(ranked_active)
  fp_curve <- cumsum(!ranked_active)
  tpr <- tp_curve / positives
  fpr <- fp_curve / negatives
  auroc <- sum(
    diff(c(0, fpr, 1)) *
      (head(c(0, tpr, 1), -1) + tail(c(0, tpr, 1), -1)) / 2
  )
  precision <- tp_curve / seq_along(tp_curve)
  auprc <- sum(precision[ranked_active]) / positives
  true_positive <- sum(selected & active)
  false_positive <- sum(selected & !active)
  data.frame(
    method = "STANCE (pooled)",
    test = "official two-stage distance kernel",
    repeat_id = repeat_id,
    scenario = "complete",
    panel_genes = length(unique(frame$gene_id)),
    tested_pairs = nrow(frame),
    active_pairs = positives,
    stage1_selected_all_genes = stage1_selected,
    auroc = auroc,
    auprc = auprc,
    selected_pairs = sum(selected),
    selected_genes = length(unique(frame$gene_id[selected])),
    true_positives = true_positive,
    false_positives = false_positive,
    false_negatives = sum(!selected & active),
    empirical_fdr = false_positive / max(true_positive + false_positive, 1),
    elapsed_seconds = elapsed
  )
}

all_summaries <- list()
for (repeat_id in repeat_ids) {
  prefix <- sprintf("repeat%d_complete", repeat_id)
  counts_frame <- read.csv(
    file.path(source_dir, paste0(prefix, "_counts.csv")),
    check.names = FALSE
  )
  coordinates_frame <- read.csv(
    file.path(source_dir, paste0(prefix, "_coordinates.csv")),
    check.names = FALSE
  )
  omega_frame <- read.csv(
    file.path(source_dir, paste0(prefix, "_omega.csv")),
    check.names = FALSE
  )
  truth <- read.csv(
    file.path(source_dir, paste0(prefix, "_truth.csv")),
    check.names = FALSE
  )

  counts <- as.matrix(counts_frame[, -1, drop = FALSE])
  rownames(counts) <- counts_frame$gene_id
  spot_names <- colnames(counts)
  library_size <- colSums(counts)
  size_factor <- library_size / mean(library_size)
  normalized_counts <- log2(sweep(counts, 2, size_factor, "/") + 1)

  coordinates <- as.matrix(coordinates_frame[, c("x", "y")])
  rownames(coordinates) <- coordinates_frame$spot_id
  coordinates <- coordinates[spot_names, , drop = FALSE]
  omega <- as.matrix(omega_frame[, paste0("CT", 0:4), drop = FALSE])
  rownames(omega) <- omega_frame$spot_id
  omega <- omega[spot_names, , drop = FALSE]
  omega <- omega / rowSums(omega)

  object <- creatSTANCEobject(
    counts = normalized_counts,
    pos = coordinates,
    prop = omega,
    covariates = NULL
  )
  object <- data_preprocess(object, normalized = TRUE)
  started <- proc.time()[["elapsed"]]
  object <- build_kernelMatrix(object, bw.method = "Silverman")
  object <- runTest1(object, correction = FALSE, pv.adjust = "BY")
  stage1_genes <- rownames(object@Test_1)[object@Test_1$p_value_adj < 0.05]
  stage2_genes <- intersect(panel, stage1_genes)

  if (length(stage2_genes) > 0) {
    object <- runTest2(
      object,
      Cell_types_to_test = paste0("CT", 0:4),
      Genes_to_test = stage2_genes,
      correction = FALSE,
      ncores = 4
    )
  }

  rows <- list()
  row_index <- 1
  for (cell_type in paste0("CT", 0:4)) {
    for (gene_id in panel) {
      p_value <- 1
      if (gene_id %in% stage2_genes) {
        p_value <- object@Test_2[[cell_type]][gene_id, "p_value"]
      }
      active <- truth$active[
        truth$gene_id == gene_id & truth$cell_type == cell_type
      ]
      rows[[row_index]] <- data.frame(
        method = "STANCE (pooled)",
        repeat_id = repeat_id,
        scenario = "complete",
        gene_id = gene_id,
        cell_type = cell_type,
        active = active,
        stage1_pass = gene_id %in% stage1_genes,
        pair_p = as.numeric(p_value)
      )
      row_index <- row_index + 1
    }
  }
  pairs <- do.call(rbind, rows)
  pairs$pair_p[!is.finite(pairs$pair_p) | pairs$pair_p < 0 | pairs$pair_p > 1] <- 1
  pairs$pair_bh_q <- p.adjust(pairs$pair_p, method = "BH")
  elapsed <- proc.time()[["elapsed"]] - started
  write.csv(
    pairs,
    file.path(source_dir, paste0(prefix, "_stance_pairs.csv")),
    row.names = FALSE
  )
  all_summaries[[length(all_summaries) + 1]] <- score_summary(
    pairs, repeat_id, elapsed, length(stage1_genes)
  )
}

summary <- do.call(rbind, all_summaries)
write.csv(
  summary,
  file.path(source_dir, "compact_stance_detection_summary.csv"),
  row.names = FALSE
)
print(summary)
