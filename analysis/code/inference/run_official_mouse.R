#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(Matrix); library(spacexr); library(MCube)})
args <- commandArgs(trailingOnly=TRUE)
stage <- if (length(args)) args[[1]] else "rctd"
sections <- if (length(args)>1) strsplit(args[[2]], ",")[[1]] else c("Anterior", "Posterior", "Anterior2")
mmm_workers <- as.integer(Sys.getenv("DISTRISVG_MMM_WORKERS", "1"))
stopifnot(length(mmm_workers)==1, !is.na(mmm_workers), mmm_workers>=1)
root <- normalizePath(".")
out <- file.path(root, "04_数值实验/results/distrisvg_completion_2026-09-25/official_mouse")
input <- file.path(out, "inputs")
dir.create(file.path(out, stage), showWarnings=FALSE)
set.seed(20260925)
writeLines(capture.output(sessionInfo()), file.path(out, stage, "sessionInfo.txt"))
panel <- read.csv(file.path(input, "test_gene_panel.csv"), check.names=FALSE)[[1]]
labels <- read.csv(file.path(input, "reference_labels.csv"), check.names=FALSE)
ref_genes <- read.csv(file.path(input, "reference_genes.csv"), check.names=FALSE)[[1]]
ref_counts <- as(readMM(file.path(input, "reference_counts_rsem.mtx")), "CsparseMatrix")
dimnames(ref_counts) <- list(ref_genes, labels$cell_id)
cell_types <- factor(labels$cell_type); names(cell_types) <- labels$cell_id
lib_ref <- labels$nUMI; names(lib_ref) <- labels$cell_id
stopifnot(max(abs(colSums(ref_counts)-lib_ref)/lib_ref)<1e-10)
# RSEM expected read counts are fractional and are not UMI measurements.
reference <- Reference(ref_counts, cell_types, lib_ref, require_int=FALSE)
profiles <- read.csv(file.path(input, "reference_profiles.csv"), row.names=1, check.names=FALSE)
audit_file <- file.path(out,stage,"run_audit.csv")
records <- if(file.exists(audit_file)) split(read.csv(audit_file,check.names=FALSE),
    read.csv(audit_file,check.names=FALSE)$section) else list()

read_section <- function(section) {
  path <- file.path(input, section)
  spots <- read.csv(file.path(path,"spots.csv"), check.names=FALSE)
  genes <- read.csv(file.path(path,"genes.csv"), check.names=FALSE)[[1]]
  counts <- as(readMM(file.path(path,"counts.mtx")), "CsparseMatrix")
  dimnames(counts) <- list(genes, spots$spot_id)
  coords <- spots[,c("x","y")]; rownames(coords) <- spots$spot_id
  umi <- spots$nUMI; names(umi) <- spots$spot_id
  weights <- as.matrix(read.csv(file.path(path,"nnls_weights.csv"), row.names=1, check.names=FALSE))
  # The archived NNLS normalizer adds epsilon; keep those values unchanged.
  stopifnot(identical(rownames(weights), spots$spot_id), all(is.finite(weights)),
            all(weights>=0), all(abs(rowSums(weights)-1)<1e-6))
  list(counts=counts, coords=coords, umi=umi, weights=weights)
}

export_pairs <- function(pvalues, section, status_by_pair=NULL) {
  types <- colnames(profiles)
  grid <- expand.grid(gene_id=panel, cell_type=types, stringsAsFactors=FALSE)
  grid$pair_p <- 1; grid$status <- "not_eligible_or_not_returned"
  for (ct in names(pvalues)) {
    values <- pvalues[[ct]]
    idx <- which(grid$cell_type==ct & grid$gene_id %in% names(values))
    grid$pair_p[idx] <- as.numeric(values[grid$gene_id[idx]])
    grid$status[idx] <- "returned"
  }
  bad <- !is.finite(grid$pair_p) | grid$pair_p<0 | grid$pair_p>1
  grid$status[bad] <- "invalid_p"; grid$pair_p[bad] <- 1
  grid$pair_bh_q <- p.adjust(grid$pair_p, "BH")
  grid$section <- section; grid$method <- stage; grid$weights <- "original_NNLS"
  write.csv(grid, file.path(out, stage, paste0(section,"_pairs.csv")), row.names=FALSE)
  grid
}

for (section in sections) {
  started <- proc.time()[["elapsed"]]
  final <- file.path(out, stage, paste0(section,"_done.rds"))
  if (file.exists(final)) { message(section, ": cached ", stage); next }
  message(section, ": starting ", stage)
  result <- tryCatch({
    data <- read_section(section)
    if (stage %in% c("rctd", "cside")) {
      base <- file.path(out, paste0(section,"_",stage,"_initialized_rsem.rds"))
      if (file.exists(base)) obj <- readRDS(base) else {
        puck <- SpatialRNA(data$coords, data$counts, data$umi)
        obj <- create.RCTD(puck, reference, max_cores=1, CELL_MIN_INSTANCE=10)
        saveRDS(obj, base)
      }
      if (stage=="rctd") {
        obj <- run.RCTD(obj, doublet_mode="full")
        raw <- as.matrix(obj@results$weights)
        weights <- raw/rowSums(raw)
        write.csv(raw, file.path(out,stage,paste0(section,"_raw_weights.csv")))
        write.csv(weights, file.path(out,stage,paste0(section,"_weights.csv")))
        omitted <- setdiff(colnames(data$counts), rownames(weights))
        write.csv(data.frame(spot_id=omitted), file.path(out,stage,paste0(section,"_omitted_spots.csv")), row.names=FALSE)
        agreement <- data.frame(cell_type=colnames(weights),
          pearson=sapply(colnames(weights), function(ct) cor(weights[,ct],data$weights[rownames(weights),ct])),
          spearman=sapply(colnames(weights), function(ct) cor(weights[,ct],data$weights[rownames(weights),ct],method="spearman")))
        write.csv(agreement, file.path(out,stage,paste0(section,"_nnls_agreement.csv")), row.names=FALSE)
      } else {
        weights <- data$weights[colnames(obj@originalSpatialRNA@counts),,drop=FALSE]
        obj <- import_weights(obj, weights)
        present <- intersect(panel, rownames(obj@originalSpatialRNA@counts))
        # Keep measured full-library totals while restricting only the tested genes.
        obj@originalSpatialRNA@counts <- obj@originalSpatialRNA@counts[present,,drop=FALSE]
        barcodes <- rownames(weights)
        xy <- data$coords[barcodes,,drop=FALSE]
        xy <- apply(xy,2,function(x) (x-min(x))/(max(x)-min(x)))
        X <- cbind(intercept=1,x=xy[,1],y=xy[,2]); rownames(X) <- barcodes
        obj <- run.CSIDE(obj, X, barcodes, doublet_mode=FALSE, test_mode="individual",
          params_to_test=2:3, weight_threshold=.8, fdr=.05, logs=FALSE)
        all <- obj@de_results$all_gene_list
        p <- lapply(all, function(d) setNames(d$p_val, rownames(d)))
        for (ct in names(all)) write.csv(all[[ct]], file.path(out,stage,paste0(section,"_",make.names(ct),"_audit.csv")))
        export_pairs(p,section)
      }
    } else if (stage=="mmm") {
      counts <- as.matrix(t(data$counts[panel,,drop=FALSE]))
      ref <- t(as.matrix(profiles[panel,colnames(data$weights),drop=FALSE]))
      obj <- createMCube(counts=counts, coordinates=as.matrix(data$coords), proportions=data$weights,
        library_sizes=data$umi, reference=ref, used_for_deconvolution=rep(TRUE,length(panel)))
      write.csv(obj@celltype_gene_test_pairs, file.path(out,stage,paste0(section,"_eligible_before_fit.csv")),row.names=FALSE)
      null_file <- file.path(out,stage,paste0(section,"_null.rds"))
      if (file.exists(null_file)) obj <- readRDS(null_file) else {
        obj <- mcubeFitNull(obj,num_workers=mmm_workers,num_threads=1)
        saveRDS(obj,null_file)
      }
      write.csv(obj@celltype_gene_test_pairs, file.path(out,stage,paste0(section,"_eligible_after_fit.csv")),row.names=FALSE)
      obj <- mcubeTest(obj,num_workers=mmm_workers,num_threads=1,shared_memory=FALSE)
      write.csv(obj@celltype_gene_test_pairs, file.path(out,stage,paste0(section,"_eligible_after_test.csv")),row.names=FALSE)
      p <- lapply(obj@pvalues, function(d) setNames(d$combined_pvalue,rownames(d)))
      for (ct in names(obj@pvalues)) write.csv(obj@pvalues[[ct]],file.path(out,stage,paste0(section,"_",make.names(ct),"_kernel_p.csv")))
      export_pairs(p,section)
    } else stop("Unknown stage")
    saveRDS(obj,final)
    data.frame(section=section,stage=stage,status="completed",message="",elapsed_seconds=proc.time()[["elapsed"]]-started)
  }, error=function(e) {
    message(section, ": FAILED: ",conditionMessage(e))
    data.frame(section=section,stage=stage,status="failed",message=conditionMessage(e),elapsed_seconds=proc.time()[["elapsed"]]-started)
  })
  records[[section]] <- result
  write.csv(do.call(rbind,records),file.path(out,stage,"run_audit.csv"),row.names=FALSE)
}
if (length(records) && any(vapply(records,function(x)x$status!="completed",logical(1)))) quit(status=2)
