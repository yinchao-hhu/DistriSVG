#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(CTSV))
suppressPackageStartupMessages(library(SpatialExperiment))

args <- commandArgs(trailingOnly = TRUE)
source_dir <- if (length(args) >= 1) args[[1]] else file.path(
  "05_论文草稿", "latex", "figs", "simulation_method_comparison", "source_data"
)
repeats <- if (length(args) >= 2) as.integer(strsplit(args[[2]], ",")[[1]]) else 0:2
conditions <- if (length(args) >= 3) strsplit(args[[3]], ",")[[1]] else c("complete", "missing")

acat <- function(p_values) {
  p_values <- pmin(pmax(as.numeric(p_values), 1e-15), 1 - 1e-15)
  statistic <- mean(tan((0.5 - p_values) * pi))
  pmin(pmax(0.5 - atan(statistic) / pi, 1e-15), 1)
}

score_summary <- function(frame) {
  active <- as.logical(frame$active)
  score <- -log10(pmax(frame$pair_p, 1e-300))
  selected <- frame$pair_bh_q < 0.05
  positives <- sum(active)
  negatives <- sum(!active)
  ordering <- order(score, decreasing = TRUE)
  ranked_active <- active[ordering]
  tp <- cumsum(ranked_active)
  fp <- cumsum(!ranked_active)
  tpr <- tp / positives
  fpr <- fp / negatives
  auroc <- sum(diff(c(0, fpr, 1)) * (head(c(0, tpr, 1), -1) + tail(c(0, tpr, 1), -1)) / 2)
  precision <- tp / seq_along(tp)
  auprc <- sum(precision[ranked_active]) / positives
  true_positive <- sum(selected & active)
  false_positive <- sum(selected & !active)
  data.frame(
    auroc = auroc,
    auprc = auprc,
    selected_pairs = sum(selected),
    selected_genes = length(unique(frame$gene_id[selected])),
    true_positives = true_positive,
    false_positives = false_positive,
    false_negatives = sum(!selected & active),
    empirical_fdr = false_positive / max(true_positive + false_positive, 1)
  )
}

all_summaries <- list()

for (repeat_id in repeats) {
  for (condition in conditions) {
    prefix <- sprintf("repeat%d_%s", repeat_id, condition)
    count_frame <- read.csv(file.path(source_dir, paste0(prefix, "_counts.csv")), check.names = FALSE)
    coordinate_frame <- read.csv(file.path(source_dir, paste0(prefix, "_coordinates.csv")), check.names = FALSE)
    omega_frame <- read.csv(file.path(source_dir, paste0(prefix, "_omega.csv")), check.names = FALSE)
    truth <- read.csv(file.path(source_dir, paste0(prefix, "_truth.csv")), check.names = FALSE)
    gene_panel <- read.csv(file.path(source_dir, "official_method_gene_panel.csv"), check.names = FALSE)$gene_id

    full_gene_ids <- count_frame$gene_id
    gene_ids <- gene_panel
    count_matrix <- as.matrix(count_frame[, -1, drop = FALSE])
    storage.mode(count_matrix) <- "integer"
    rownames(count_matrix) <- full_gene_ids
    spot_ids <- colnames(count_matrix)
    stopifnot(identical(spot_ids, coordinate_frame$spot_id))
    stopifnot(identical(spot_ids, omega_frame$spot_id))

    coordinates <- as.matrix(coordinate_frame[, c("x", "y")])
    rownames(coordinates) <- spot_ids
    omega <- as.matrix(omega_frame[, grep("^CT", names(omega_frame)), drop = FALSE])
    rownames(omega) <- spot_ids
    omega <- omega / rowSums(omega)
    Y <- t(count_matrix)
    selected_gene_indices <- match(gene_ids, colnames(Y))
    stopifnot(!anyNA(selected_gene_indices))

    standardized_coordinates <- t(coordinates) - colMeans(coordinates)
    standardized_coordinates <- t(
      standardized_coordinates / apply(standardized_coordinates, 1, sd)
    )
    quantiles <- c(0.4, 0.6)
    psi1 <- quantile(abs(standardized_coordinates[, 1]), quantiles)
    psi2 <- quantile(abs(standardized_coordinates[, 2]), quantiles)
    patterns <- c("linear", "gau1", "gau2", "cos1", "cos2")
    p_value_array <- array(
      NA_real_,
      dim = c(length(gene_ids), 2 * ncol(omega), length(patterns)),
      dimnames = list(gene_ids, NULL, patterns)
    )

    started <- proc.time()[["elapsed"]]
    for (pattern_index in seq_along(patterns)) {
      fit_pattern <- patterns[[pattern_index]]
      cache_path <- file.path(source_dir, sprintf("%s_ctsv_%s_axis_pval.csv", prefix, fit_pattern))
      if (file.exists(cache_path)) {
        cached <- read.csv(cache_path, check.names = FALSE)
        matrix_result <- as.matrix(cached[, -1, drop = FALSE])
        rownames(matrix_result) <- cached$gene_id
      } else {
        if (fit_pattern == "gau1") {
          h1 <- exp(-standardized_coordinates[, 1]^2 / (2 * psi1[[1]]^2))
          h2 <- exp(-standardized_coordinates[, 2]^2 / (2 * psi2[[1]]^2))
        } else if (fit_pattern == "gau2") {
          h1 <- exp(-standardized_coordinates[, 1]^2 / (2 * psi1[[2]]^2))
          h2 <- exp(-standardized_coordinates[, 2]^2 / (2 * psi2[[2]]^2))
        } else if (fit_pattern == "cos1") {
          h1 <- cos(2 * pi * standardized_coordinates[, 1] / psi1[[1]])
          h2 <- cos(2 * pi * standardized_coordinates[, 2] / psi2[[1]])
        } else if (fit_pattern == "cos2") {
          h1 <- cos(2 * pi * standardized_coordinates[, 1] / psi1[[2]])
          h2 <- cos(2 * pi * standardized_coordinates[, 2] / psi2[[2]])
        } else {
          h1 <- standardized_coordinates[, 1]
          h2 <- standardized_coordinates[, 2]
        }
        design <- cbind(omega * h1, omega * h2, omega)
        matrix_result <- do.call(
          rbind,
          parallel::mclapply(
            selected_gene_indices,
            CTSV:::.P_gene,
            Y = Y,
            Tmp = design,
            h1 = h1,
            h2 = h2,
            mc.cores = 2,
            mc.preschedule = TRUE
          )
        )
        rownames(matrix_result) <- gene_ids
        write.csv(
          data.frame(gene_id = gene_ids, matrix_result, check.names = FALSE),
          cache_path,
          row.names = FALSE
        )
      }
      p_value_array[, , pattern_index] <- matrix_result[gene_ids, , drop = FALSE]
      message(prefix, ": completed CTSV pattern ", fit_pattern)
    }
    p_value_array[is.na(p_value_array) | p_value_array < 0] <- 1
    cauchy_terms <- tan((0.5 - p_value_array) * pi)
    result_pval <- 0.5 - atan(apply(cauchy_terms, c(1, 2), mean)) / pi
    rownames(result_pval) <- gene_ids
    elapsed <- proc.time()[["elapsed"]] - started
    cell_types <- colnames(omega)
    pair_rows <- vector("list", length(gene_ids) * length(cell_types))
    index <- 1
    for (cell_index in seq_along(cell_types)) {
      pair_p <- apply(
        cbind(result_pval[, cell_index], result_pval[, length(cell_types) + cell_index]),
        1,
        acat
      )
      for (gene_index in seq_along(gene_ids)) {
        truth_row <- truth[
          truth$gene_id == gene_ids[[gene_index]] & truth$cell_type == cell_types[[cell_index]],
          ,
          drop = FALSE
        ]
        pair_rows[[index]] <- data.frame(
          repeat_id = repeat_id,
          condition = condition,
          gene_id = gene_ids[[gene_index]],
          cell_type = cell_types[[cell_index]],
          active = truth_row$active,
          pair_p = pair_p[[gene_index]]
        )
        index <- index + 1
      }
    }
    pairs <- do.call(rbind, pair_rows)
    pairs$pair_bh_q <- p.adjust(pairs$pair_p, method = "BH")
    write.csv(pairs, file.path(source_dir, paste0(prefix, "_ctsv_pairs.csv")), row.names = FALSE)

    summary <- score_summary(pairs)
    summary$method <- "CTSV (pooled)"
    summary$repeat_id <- repeat_id
    summary$condition <- condition
    summary$elapsed_seconds <- elapsed
    all_summaries[[length(all_summaries) + 1]] <- summary
    message(prefix, ": AUROC=", sprintf("%.4f", summary$auroc),
            ", AUPRC=", sprintf("%.4f", summary$auprc),
            ", selected=", summary$selected_pairs,
            ", TP=", summary$true_positives,
            ", FP=", summary$false_positives)
  }
}

write.csv(
  do.call(rbind, all_summaries),
  file.path(source_dir, "official_ctsv_detection_summary.csv"),
  row.names = FALSE
)
