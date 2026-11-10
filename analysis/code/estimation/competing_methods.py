#!/usr/bin/env python3
"""Preflight and non-destructive replay of retained official R adapters.

Never substitutes a Python approximation for an unavailable official method.
C-SIDE/MMM are explicit input-contract gaps, not zero-discovery results.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from run_revision import ROOT, SOURCE, OUTPUT, digest

SCRIPTS = ROOT / "analysis/code/figures"
METHODS = {
    "CTSV": dict(url="https://github.com/jingeyu/CTSV", script="run_official_ctsv_benchmark.R",
                 packages=["CTSV", "SpatialExperiment"], required_source=None, conditions=["complete", "missing"]),
    "STANCE": dict(url="https://github.com/Cui-STT-Lab/STANCE", script="run_stance_compact.R",
                   packages=["gaston", "KRLS", "Rcpp", "RcppArmadillo"], required_source="src/matrix_inversion.cpp", conditions=["complete"]),
    "Celina": dict(url="https://github.com/pekjoonwu/CELINA", script="run_celina_compact.R",
                   packages=["CompQuadForm", "dplyr", "fields", "Matrix", "pbmcapply", "pdist", "pracma", "Rcpp", "RcppArmadillo", "RSpectra", "tidyr"],
                   required_source="src/CELINA.cpp", conditions=["complete"]),
    "spVC": dict(url="https://github.com/shanyu-stat/spVC", script="run_spvc_compact.R",
                 packages=["BPST", "Matrix", "MGLM", "mgcv", "parallel"], required_source="R", conditions=["complete"]),
}


def preflight():
    r = shutil.which("Rscript")
    report = {"Rscript": r, "official_methods": {}, "not_run": {
        "C-SIDE": "Needs a validated RCTD object with matching profiles/platform likelihoods; import_weights is supported, but weights alone are insufficient here.",
        "MMM": "Needs validated reference expression/profile input; no matched simulation reference has been frozen.",
        "ctSVG": "Not applicable directly to mixed-cell spot inputs; requires single-cell-resolution expression and labels."}}
    for method, settings in METHODS.items():
        status = dict(script=settings["script"], url=settings["url"],
                      archived_outputs="available; evaluated in this revision", new_execution="not_run")
        if r:
            expr = "p <- c(" + ",".join(json.dumps(x) for x in settings["packages"]) + "); cat(p[!vapply(p, requireNamespace, logical(1), quietly=TRUE)], sep='\\n')"
            check = subprocess.run([r, "--vanilla", "-e", expr], text=True, capture_output=True, check=True)
            status["missing_packages"] = check.stdout.splitlines()
        else:
            status["blocker"] = "Rscript is unavailable on PATH; no fresh official R-method fitting was performed"
        report["official_methods"][method] = status
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "competitor_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def replay(method, source_path):
    report = preflight()
    if not report["Rscript"]:
        raise RuntimeError("Install the official R environment before replay; no substitute was run")
    if report["official_methods"][method].get("missing_packages"):
        raise RuntimeError("Required official-method R packages are missing; inspect competitor_preflight.json")
    settings = METHODS[method]
    if settings["required_source"] and (source_path is None or not (source_path / settings["required_source"]).exists()):
        raise ValueError("Supply --official-source with the matching official source checkout")
    target = OUTPUT / "official_replay" / method
    target.mkdir(parents=True, exist_ok=False)
    inputs = []
    for rep in range(3):
        for condition in settings["conditions"]:
            for kind in ("counts", "coordinates", "omega", "truth"):
                inputs.append(SOURCE / f"repeat{rep}_{condition}_{kind}.csv")
    inputs.append(SOURCE / "official_method_gene_panel.csv")
    for path in inputs:
        shutil.copy2(path, target / path.name)
    adapter = SCRIPTS / settings["script"]
    command = [report["Rscript"], str(adapter)]
    if method == "CTSV":
        command += [str(target), "0,1,2", "complete,missing"]
    else:
        source_arg = source_path / "R" if method == "spVC" else source_path
        command += [str(source_arg.resolve()), str(target), "0,1,2"]
    (target / "replay_manifest.json").write_text(json.dumps(dict(command=command, adapter_sha256=digest(adapter),
        inputs={p.name: digest(p) for p in inputs}, status="started"), indent=2) + "\n")
    with (target / "run.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    (target / "COMPLETED").write_text("Official adapter exited successfully; inspect pair coverage and failure audit before comparing.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=list(METHODS))
    parser.add_argument("--official-source", type=Path)
    args = parser.parse_args()
    if args.run:
        replay(args.run, args.official_source)
    else:
        preflight()
