#!/usr/bin/env Rscript

# Run CELINA's default 11-kernel test on the compact benchmark. The 40-gene
# evaluation panel is fixed before fitting, so reference-based gene filtering is
# bypassed while the official null model, kernel tests and ACAT combination are
# retained.

suppressPackageStartupMessages({
  library(CompQuadForm)
  library(dplyr)
  library(fields)
  library(Matrix)
  library(pbmcapply)
  library(pdist)
  library(pracma)
  library(Rcpp)
  library(RcppArmadillo)
  library(RSpectra)
  library(tidyr)
})

args <- commandArgs(trailingOnly = TRUE)
celina_source <- if (length(args) >= 1) args[[1]] else "/private/tmp/method-src/CELINA-master"
source_dir <- if (length(args) >= 2) args[[2]] else file.path(
  "05_论文草稿", "latex", "figs", "simulation_method_comparison", "source_data"
)
repeat_ids <- if (length(args) >= 3) {
  as.integer(strsplit(args[[3]], ",")[[1]])
} else {
  0:2
}

Rcpp::sourceCpp(file.path(celina_source, "src", "CELINA.cpp"), rebuild = FALSE)
source(file.path(celina_source, "R", "CELINA.R"), local = globalenv())

panel <- read.csv(
  file.path(source_dir, "official_method_gene_panel.csv"),
  check.names = FALSE
)$gene_id

score_summary <- function(frame, repeat_id, failed_fits, elapsed) {
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
    method = "Celina (pooled)",
    test = "official 11-kernel ACAT",
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
    failed_fits = failed_fits,
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
  normalized_panel <- log2(
    sweep(counts[panel, , drop = FALSE], 2, size_factor, "/") + 1
  )

  coordinates <- as.matrix(coordinates_frame[, c("x", "y")])
  rownames(coordinates) <- coordinates_frame$spot_id
  coordinates <- coordinates[spot_names, , drop = FALSE]
  omega <- as.matrix(omega_frame[, paste0("CT", 0:4), drop = FALSE])
  rownames(omega) <- omega_frame$spot_id
  omega <- omega[spot_names, , drop = FALSE]
  omega <- omega / rowSums(omega)

  object <- Create_Celina_Object(
    celltype_mat = t(omega),
    gene_expression_mat = counts[panel, , drop = FALSE],
    location = coordinates,
    scaling_method = "separate",
    covariates = NULL,
    project = paste0("AP-FedAdam-compact-repeat", repeat_id)
  )
  object@gene_expression_mat <- normalized_panel
  object@genes_list <- setNames(rep(list(panel), ncol(omega)), colnames(omega))

  started <- proc.time()[["elapsed"]]
  object <- Calculate_Kernel(object, approximation = FALSE)
  failed_fits <- 0
  error_rows <- list()
  rows <- list()
  row_index <- 1
  for (cell_type in colnames(omega)) {
    target_index <- which(rownames(object@celltype_mat) == cell_type)
    other_weights <- t(object@celltype_mat[-target_index, , drop = FALSE])
    target_weight <- as.numeric(object@celltype_mat[target_index, ])
    for (gene_id in panel) {
      response <- as.numeric(object@gene_expression_mat[gene_id, ])
      residual_response <- stats::lm(response ~ other_weights)$residuals
      p_value <- tryCatch(
        {
          kernel_p <- Testing_interaction_multi_kernels(
            residual_response,
            target_weight,
            covariates = NULL,
            kernel_mat = object@kernelmat
          )
          as.numeric(kernel_p[, "CombinedPvals"])
        },
        error = function(error) {
          error_rows[[length(error_rows)+1]] <<- data.frame(gene_id=gene_id,cell_type=cell_type,message=conditionMessage(error))
          failed_fits <<- failed_fits + 1
          1
        }
      )
      active <- truth$active[
        truth$gene_id == gene_id & truth$cell_type == cell_type
      ]
      rows[[row_index]] <- data.frame(
        method = "Celina (pooled)",
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
  write.csv(if(length(error_rows)) do.call(rbind,error_rows) else data.frame(gene_id=character(),cell_type=character(),message=character()), file.path(source_dir,paste0(prefix,"_celina_errors.csv")),row.names=FALSE)
  pairs <- do.call(rbind, rows)
  pairs$pair_p[!is.finite(pairs$pair_p) | pairs$pair_p < 0 | pairs$pair_p > 1] <- 1
  pairs$pair_bh_q <- p.adjust(pairs$pair_p, method = "BH")
  elapsed <- proc.time()[["elapsed"]] - started
  write.csv(
    pairs,
    file.path(source_dir, paste0(prefix, "_celina_pairs.csv")),
    row.names = FALSE
  )
  all_summaries[[length(all_summaries) + 1]] <- score_summary(
    pairs, repeat_id, failed_fits, elapsed
  )
}

summary <- do.call(rbind, all_summaries)
write.csv(
  summary,
  file.path(source_dir, "compact_celina_detection_summary.csv"),
  row.names = FALSE
)
print(summary)
