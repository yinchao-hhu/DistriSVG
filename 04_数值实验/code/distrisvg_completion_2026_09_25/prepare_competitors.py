"""Copy the frozen benchmark and make auditable, resource-only R adapters."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
OLD = CODE.parent / "figure_scripts"
DATA = ROOT / "05_论文草稿/latex/figs/simulation_method_comparison/source_data"
OUT = ROOT / "04_数值实验/results/distrisvg_completion_2026-09-25/official_competitors"
SOURCES = ROOT / "08_其他/运行环境/distrisvg_2026_09_25/sources"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    inputs = OUT / "source_data"
    adapters = OUT / "adapters"
    inputs.mkdir(parents=True, exist_ok=True)
    adapters.mkdir(exist_ok=True)
    hashes = {}
    for pattern in ("official_method_gene_panel.csv", "repeat*_complete_counts.csv",
                    "repeat*_complete_coordinates.csv", "repeat*_complete_omega.csv",
                    "repeat*_complete_truth.csv", "repeat*_missing_counts.csv",
                    "repeat*_missing_coordinates.csv", "repeat*_missing_omega.csv",
                    "repeat*_missing_truth.csv"):
        for path in DATA.glob(pattern):
            shutil.copy2(path, inputs / path.name)
            hashes[path.name] = sha(path)
    for method, name in (("CTSV", "run_official_ctsv_benchmark.R"),
                         ("STANCE", "run_stance_compact.R"),
                         ("CELINA", "run_celina_compact.R"),
                         ("spVC", "run_spvc_compact.R")):
        path = OLD / name
        text = path.read_text()
        changes = []
        if method == "CTSV":
            original = "suppressPackageStartupMessages(library(CTSV))\nsuppressPackageStartupMessages(library(SpatialExperiment))"
            replacement = ("suppressPackageStartupMessages(library(pscl))\n"
                           f'source("{SOURCES / "CTSV/R/CTSV.R"}")')
            assert original in text
            text = text.replace(original, replacement).replace("CTSV:::.P_gene", ".P_gene")
            changes.append("Source identical official .P_gene; bypass unused SpatialExperiment wrapper; 2 workers after C-SIDE completed")
        else:
            text = text.replace("ncores = 4", "ncores = 1").replace("para.cores = 4", "para.cores = 1")
            changes.append("1 worker; source and output paths passed explicitly")
        if method == "CELINA":
            text = text.replace("  rows <- list()", "  error_rows <- list()\n  rows <- list()")
            text = text.replace("          failed_fits <<- failed_fits + 1",
                '          error_rows[[length(error_rows)+1]] <<- data.frame(gene_id=gene_id,cell_type=cell_type,message=conditionMessage(error))\n          failed_fits <<- failed_fits + 1')
            text = text.replace("  pairs <- do.call(rbind, rows)",
                '  write.csv(if(length(error_rows)) do.call(rbind,error_rows) else data.frame(gene_id=character(),cell_type=character(),message=character()), file.path(source_dir,paste0(prefix,"_celina_errors.csv")),row.names=FALSE)\n  pairs <- do.call(rbind, rows)')
            changes.append("Save caught per-pair error messages; do not alter p-values")
        target = adapters / name
        target.write_text(text)
        commit = subprocess.check_output(["git", "-C", str(SOURCES / method), "rev-parse", "HEAD"], text=True).strip()
        hashes[method] = dict(upstream_commit=commit, old_adapter_sha256=sha(path),
                              adapter_sha256=sha(target), changes=changes)
    (OUT / "adapter_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
