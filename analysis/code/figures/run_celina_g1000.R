#!/usr/bin/env Rscript

# Run the official CELINA testing core on the pre-specified G=1000 simulation
# panel. Gene filtering is bypassed because the panel was fixed before fitting;
# the spatial test, null fit and low-rank Gaussian kernel are unchanged.

suppressPackageStartupMessages({
  library(CompQuadForm)
  library(dplyr)
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

Rcpp::sourceCpp(file.path(celina_source, "src", "CELINA.cpp"), rebuild = FALSE)
source(file.path(celina_source, "R", "CELINA.R"), local = globalenv())

data_dir <- file.path("04_数值实验", "code", "new_test1-5", "sim_data")
count_path <- file.path(
  data_dir, "AllClients_Y_8_G_1000_L1_2e-07_Round1000_h_linear.csv"
)
input_path <- file.path(
  data_dir, "Input_Omega_S_8_G_1000_L1_2e-07_Round1000_h_linear.csv"
)

panel <- read.csv(file.path(source_dir, "official_g1000_gene_panel.csv"), check.names = FALSE)
truth_pairs <- read.csv(
  file.path(source_dir, "g1000_ap_fedadam_panel_pairs.csv"), check.names = FALSE
)
truth_pairs <- truth_pairs[
  truth_pairs$scenario == "complete",
  c("gene_id", "cell_type", "active")
]

count_frame <- read.csv(count_path, check.names = FALSE)
input_frame <- read.csv(input_path, check.names = FALSE)
stopifnot(identical(count_frame$Client_ID, input_frame$Client_ID))
stopifnot(identical(count_frame$Spot_ID, input_frame$Spot_ID))

all_gene_columns <- grep("^Gene_", names(count_frame), value = TRUE)
panel_genes <- panel$gene_id
spot_names <- sprintf(
  "client%s_spot%s", count_frame$Client_ID, count_frame$Spot_ID
)
full_library_size <- rowSums(count_frame[, all_gene_columns, drop = FALSE])
size_factor <- full_library_size / mean(full_library_size)
raw_panel <- t(as.matrix(count_frame[, panel_genes, drop = FALSE]))
normalized_panel <- log2(sweep(raw_panel, 2, size_factor, "/") + 1)
colnames(normalized_panel) <- spot_names

coordinates <- as.matrix(input_frame[, c("Coord_X", "Coord_Y")])
rownames(coordinates) <- spot_names
weights <- as.matrix(input_frame[, grep("^Omega_Cell_", names(input_frame)), drop = FALSE])
weights <- weights / rowSums(weights)
colnames(weights) <- paste0("CT", seq_len(ncol(weights)) - 1)
rownames(weights) <- spot_names

object <- Create_Celina_Object(
  celltype_mat = t(weights),
  gene_expression_mat = raw_panel,
  location = coordinates,
  scaling_method = "separate",
  covariates = NULL,
  project = "AP-FedAdam-G1000-fixed-panel"
)
object@gene_expression_mat <- normalized_panel
object@genes_list <- setNames(
  rep(list(panel_genes), ncol(weights)), colnames(weights)
)

started <- proc.time()[["elapsed"]]
object <- Calculate_Kernel(
  object,
  approximation = TRUE,
  bandwidthtype = "Silverman",
  sparseKernel_tol = 1e-6,
  sparseKernel_ncore = 4
)
kernel_seconds <- proc.time()[["elapsed"]] - started

pair_rows <- list()
row_index <- 1
failed_fits <- 0
for (cell_type in colnames(weights)) {
  target_index <- which(rownames(object@celltype_mat) == cell_type)
  other_weights <- t(object@celltype_mat[-target_index, , drop = FALSE])
  target_weight <- as.numeric(object@celltype_mat[target_index, ])
  for (gene_id in panel_genes) {
    response <- as.numeric(object@gene_expression_mat[gene_id, ])
    residual_response <- stats::lm(response ~ other_weights)$residuals
    p_value <- tryCatch(
      Testing_interaction(
        residual_response,
        target_weight,
        covariates = NULL,
        kernelmat_approx_U = object@kernelmat_approx_U
      ),
      error = function(error) {
        failed_fits <<- failed_fits + 1
        1
      }
    )
    pair_rows[[row_index]] <- data.frame(
      method = "Celina (pooled; approximate kernel)",
      scenario = "complete",
      data_mode = "linear",
      gene_id = gene_id,
      cell_type = cell_type,
      pair_p = as.numeric(p_value)
    )
    row_index <- row_index + 1
  }
}

pairs <- do.call(rbind, pair_rows)
pairs <- merge(
  pairs, truth_pairs,
  by = c("gene_id", "cell_type"), all.x = TRUE, sort = FALSE
)
pairs$pair_p[!is.finite(pairs$pair_p) | pairs$pair_p < 0 | pairs$pair_p > 1] <- 1
pairs$pair_bh_q <- p.adjust(pairs$pair_p, method = "BH")
write.csv(
  pairs,
  file.path(source_dir, "g1000_complete_linear_celina_pairs.csv"),
  row.names = FALSE
)

score_summary <- function(frame) {
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
    method = "Celina (pooled; approximate kernel)",
    test = "official low-rank Gaussian kernel",
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
    bandwidth = object@bandwidth,
    kernel_seconds = kernel_seconds,
    total_seconds = proc.time()[["elapsed"]] - started
  )
}

summary <- score_summary(pairs)
write.csv(
  summary,
  file.path(source_dir, "g1000_complete_linear_celina_summary.csv"),
  row.names = FALSE
)
print(summary)
