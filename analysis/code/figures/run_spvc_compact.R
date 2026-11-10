#!/usr/bin/env Rscript

# Run spVC's documented full-model option on the compact benchmark and extract
# p-values for the five cell-type-specific varying coefficients.

suppressPackageStartupMessages({
  library(BPST)
  library(Matrix)
  library(MGLM)
  library(mgcv)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
spvc_source <- if (length(args) >= 1) args[[1]] else "/private/tmp/spVC-src/R"
source_dir <- if (length(args) >= 2) args[[2]] else file.path(
  "05_论文草稿", "latex", "figs", "simulation_method_comparison", "source_data"
)
repeat_ids <- if (length(args) >= 3) {
  as.integer(strsplit(args[[3]], ",")[[1]])
} else {
  0:2
}

for (source_file in list.files(spvc_source, pattern = "\\.R$", full.names = TRUE)) {
  source(source_file, local = globalenv())
}

panel <- read.csv(
  file.path(source_dir, "official_method_gene_panel.csv"),
  check.names = FALSE
)$gene_id

regular_mesh <- function(grid_size = 6) {
  values <- seq(0, 1, length.out = grid_size)
  vertices <- as.matrix(expand.grid(x = values, y = values))
  vertex_id <- matrix(seq_len(grid_size^2), nrow = grid_size, ncol = grid_size)
  triangles <- list()
  index <- 1
  for (row in seq_len(grid_size - 1)) {
    for (column in seq_len(grid_size - 1)) {
      lower_left <- vertex_id[row, column]
      lower_right <- vertex_id[row + 1, column]
      upper_left <- vertex_id[row, column + 1]
      upper_right <- vertex_id[row + 1, column + 1]
      triangles[[index]] <- c(lower_left, lower_right, upper_right)
      triangles[[index + 1]] <- c(lower_left, upper_right, upper_left)
      index <- index + 2
    }
  }
  list(V = vertices, Tr = do.call(rbind, triangles))
}

score_summary <- function(frame, repeat_id, failed_genes, elapsed) {
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
    method = "spVC (pooled)",
    test = "official full varying-coefficient model",
    repeat_id = repeat_id,
    scenario = "complete",
    panel_genes = length(unique(frame$gene_id)),
    tested_pairs = nrow(frame),
    active_pairs = positives,
    auroc = auroc,
    auprc = auprc,
    selected_pairs = sum(selected),
    selected_genes = length(unique(frame$gene_id[selected])),
    true_positives = true_positive,
    false_positives = false_positive,
    false_negatives = sum(!selected & active),
    empirical_fdr = false_positive / max(true_positive + false_positive, 1),
    failed_genes = failed_genes,
    elapsed_seconds = elapsed
  )
}

mesh <- regular_mesh(6)
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
  size_factors <- colSums(counts) / median(colSums(counts))
  coordinates <- as.matrix(coordinates_frame[, c("x", "y")])
  rownames(coordinates) <- coordinates_frame$spot_id
  coordinates <- coordinates[spot_names, , drop = FALSE]
  omega <- as.matrix(omega_frame[, paste0("CT", 0:4), drop = FALSE])
  rownames(omega) <- omega_frame$spot_id
  omega <- omega[spot_names, , drop = FALSE]
  omega <- omega / rowSums(omega)

  started <- proc.time()[["elapsed"]]
  fit <- test.spVC(
    Y = counts[panel, , drop = FALSE],
    X = omega,
    S = coordinates,
    V = mesh$V,
    Tr = mesh$Tr,
    para.cores = 4,
    scaleX = FALSE,
    subset = seq_along(panel),
    p.adjust.method = "BH",
    p.adjust.thresh = 0.05,
    linear.fit = FALSE,
    reduced.only = FALSE,
    twostep = FALSE,
    filter.min.nonzero = 0,
    filter.spot.counts = 0,
    fix.constant = NULL,
    fix.varying = NULL,
    size.factors = size_factors
  )
  full_results <- fit$results.full
  failed_genes <- sum(!panel %in% names(full_results))
  component_audit <- do.call(
    rbind,
    lapply(names(full_results), function(gene_id) {
      fitted_p <- full_results[[gene_id]]$p.value
      data.frame(
        gene_id = gene_id,
        component = names(fitted_p),
        p_value = as.numeric(fitted_p)
      )
    })
  )
  write.csv(
    component_audit,
    file.path(source_dir, paste0(prefix, "_spvc_component_audit.csv")),
    row.names = FALSE
  )
  rows <- list()
  row_index <- 1
  cell_types <- paste0("CT", 0:4)
  varying_components <- setNames(paste0("gamma_X", 1:5), cell_types)
  for (cell_type in cell_types) {
    p_name <- varying_components[[cell_type]]
    for (gene_id in panel) {
      p_value <- 1
      if (gene_id %in% names(full_results)) {
        fitted_p <- full_results[[gene_id]]$p.value
        if (p_name %in% names(fitted_p)) {
          p_value <- as.numeric(fitted_p[[p_name]])
        }
      }
      active <- truth$active[
        truth$gene_id == gene_id & truth$cell_type == cell_type
      ]
      rows[[row_index]] <- data.frame(
        method = "spVC (pooled)",
        repeat_id = repeat_id,
        scenario = "complete",
        gene_id = gene_id,
        cell_type = cell_type,
        active = active,
        pair_p = p_value
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
    file.path(source_dir, paste0(prefix, "_spvc_pairs.csv")),
    row.names = FALSE
  )
  all_summaries[[length(all_summaries) + 1]] <- score_summary(
    pairs, repeat_id, failed_genes, elapsed
  )
}

summary <- do.call(rbind, all_summaries)
write.csv(
  summary,
  file.path(source_dir, "compact_spvc_detection_summary.csv"),
  row.names = FALSE
)
print(summary)
