#!/usr/bin/env python3
"""Non-destructive metric audit and matched-input NB offset sensitivity runs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from metrics import acat, bh, pair_metrics
from nb_core import Site, diagonal_score_p, fit

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "05_论文草稿/latex/figs"
SOURCE = FIG / "simulation_method_comparison/source_data"
OUTPUT = ROOT / "04_数值实验/results/distrisvg_revision_2026-09-18"
FAMILIES = ("linear", "gau1", "gau2", "cos1", "cos2")
CONFIG = dict(version="2026-09-18-v1", repeats=[0, 1, 2], families=FAMILIES,
              rounds=150, local_steps=5, dispersion=12.0, l1=2e-6,
              ap_lr=0.035, fedavg_lr=0.12, history_scale=0.05,
              conditions=["complete", "missing", "rotation90"],
              offsets=["none", "library"], offset_rule="log(full-input library / local median)",
              rotation="client1, 90 degrees about (0.5, 0.5), linear fit only",
              real_panel_seed=20260918, real_panel_genes=160, real_rounds=150,
              real_families=FAMILIES, real_l1=2e-6,
              inference="penalised diagonal score-square diagnostic, not calibrated discovery",
              missing="existing supplied incomplete omega; counts unchanged",
              numerics="stable canonical NB mean, no predictor clipping; fixed dispersion",
              timestamp_policy="run-specific wall seconds, never inferred from cached outputs")
INPUTS = {}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def track(path):
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    return path


def read(path, **kwargs):
    return pd.read_csv(track(path), **kwargs)


def write_manifest():
    manifest_path = OUTPUT / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for name, value in existing["inputs"].items():
            if name in INPUTS and INPUTS[name] != value:
                raise ValueError(f"Input changed during revision: {name}")
            INPUTS[name] = value
    code = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "scipy", "scikit-learn", "torch", "h5py", "matplotlib", "Pillow")}
    manifest = dict(config=CONFIG, inputs=INPUTS, code_sha256=code, packages=versions,
                    python=sys.version, platform=platform.platform(),
                    numpy=np.__version__, pandas=pd.__version__,
                    scope="Existing data only; metric recomputation and separate sensitivity fits.")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def metric_audit():
    large = read(FIG / "simulation_detection/source_data/simulation_detection_pair_results.csv")
    rows = []
    for (scenario, mode), frame in large.groupby(["scenario", "data_mode"], sort=False):
        if frame.duplicated(["Gene_ID", "Cell_Type"]).any() or len(frame) != 5000:
            raise ValueError("Invalid large-scale pair universe")
        for name in ("joint_all_bases", "matched_basis"):
            row = dict(scenario=scenario, data_mode=mode, procedure=name, repeats=1)
            row.update(pair_metrics(frame.truth_active, frame[f"p_{name}"], frame[f"q_{name}"]))
            row["selected_genes"] = frame.loc[frame[f"q_{name}"] < .05, "Gene_ID"].nunique()
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT / "archive_g1000_detection_metrics.csv", index=False)

    panel = set(read(SOURCE / "official_method_gene_panel.csv").gene_id)
    rows, pairs = [], []
    for condition in ("complete", "missing"):
        methods = ["ap_fedadam", "ctsv", "stance", "celina", "spvc"] if condition == "complete" else ["ap_fedadam", "ctsv"]
        for rep in range(3):
            truth = read(SOURCE / f"repeat{rep}_{condition}_truth.csv")
            truth = truth[truth.gene_id.isin(panel)].set_index(["gene_id", "cell_type"]).sort_index()
            for method in methods:
                frame = read(SOURCE / f"repeat{rep}_{condition}_{method}_pairs.csv")
                frame = frame[frame.gene_id.isin(panel)].set_index(["gene_id", "cell_type"]).sort_index()
                if len(frame) != 200 or not frame.index.equals(truth.index) or not np.array_equal(frame.active, truth.active):
                    raise ValueError(f"Incompatible paired panel: {condition}/{rep}/{method}")
                q = bh(frame.pair_p.to_numpy())
                row = dict(method=method, condition=condition, repeat=rep,
                           **pair_metrics(frame.active, frame.pair_p, q))
                row["selected_genes"] = len(set(frame.index.get_level_values(0)[q < .05]))
                rows.append(row)
                frame = frame[["active", "pair_p"]].reset_index()
                frame["q_common"], frame["method"], frame["condition"], frame["repeat"] = q, method, condition, rep
                pairs.append(frame)
    per_repeat = pd.DataFrame(rows)
    per_repeat.to_csv(OUTPUT / "archive_competitor_metrics_per_repeat.csv", index=False)
    summarize(per_repeat, ["condition", "method"]).to_csv(OUTPUT / "archive_competitor_metrics_summary.csv", index=False)
    pd.concat(pairs).to_csv(OUTPUT / "archive_competitor_pairs.csv", index=False)
    print("Archived metrics verified: all 3 generating patterns, 5 complete-panel methods, 2 missing-panel methods", flush=True)


def summarize(frame, keys):
    numeric = frame.select_dtypes("number").columns.difference(keys + ["repeat"])
    grouped = frame.groupby(keys, sort=False)
    out = grouped[numeric].agg(["mean", "std"])
    out.columns = [f"{x}_{y}" for x, y in out.columns]
    out["n_runs"] = grouped.size()
    return out.reset_index()


def load_sim(rep, condition):
    input_condition = "missing" if condition == "missing" else "complete"
    prefix = f"repeat{rep}_{input_condition}"
    counts = read(SOURCE / f"{prefix}_counts.csv").set_index("gene_id")
    coords = read(SOURCE / f"{prefix}_coordinates.csv").set_index("spot_id")
    omega = read(SOURCE / f"{prefix}_omega.csv").set_index("spot_id")
    if list(counts.columns) != list(coords.index) or not coords.index.equals(omega.index):
        raise ValueError("Spot-order mismatch")
    truth = read(SOURCE / f"{prefix}_truth.csv")
    cell_types, genes = list(omega.columns), list(counts.index)
    truth = truth.set_index(["cell_type", "gene_id"]).loc[pd.MultiIndex.from_product([cell_types, genes])]
    beta = np.stack([truth.true_beta_h1.to_numpy(), truth.true_beta_h2.to_numpy()]).reshape(2, len(cell_types), len(genes))
    clients = []
    for client_id in sorted(coords.client.unique()):
        mask = coords.client == client_id
        xy = coords.loc[mask, ["x", "y"]].to_numpy()
        if condition == "rotation90" and client_id == 1:
            xy = (xy - .5) @ np.array([[0., 1.], [-1., 0.]]) + .5
        y = counts.loc[:, mask.index[mask]].to_numpy(dtype=float).T
        clients.append(dict(y=y, coords=xy, omega=omega.loc[mask].to_numpy(), library=y.sum(1)))
    return clients, genes, cell_types, beta


def make_sites(clients, family, offset):
    xy = np.concatenate([c["coords"] for c in clients])
    sd = xy.std(axis=0, ddof=1)
    z = (xy - xy.mean(0)) / sd
    scales = np.quantile(np.abs(z), [.4, .6], axis=0)
    if family == "linear":
        h = z
    elif family.startswith("gau"):
        h = np.exp(-z ** 2 / (2 * scales[int(family[-1]) - 1] ** 2))
    elif family.startswith("cos"):
        h = np.cos(2 * np.pi * z / scales[int(family[-1]) - 1])
    else:
        raise ValueError(family)
    sites, start = [], 0
    for client in clients:
        n, k = client["omega"].shape
        om = client["omega"]
        if np.any(om < 0) or not np.allclose(om.sum(1), 1, atol=1e-6):
            raise ValueError("Weights must be non-negative and unit-sum")
        x = np.concatenate([om, om * h[start:start+n, 0, None], om * h[start:start+n, 1, None]], axis=1)
        available = np.tile(np.any(om > 0, axis=0), 3)
        lib = client["library"]
        if np.any(lib <= 0):
            raise ValueError("Zero library; no implicit pseudocount")
        log_offset = np.log(lib / np.median(lib)) if offset == "library" else np.zeros(n)
        sites.append(Site(x, client["y"], log_offset, available))
        start += n
    return sites, sd


def pooled_sites(sites):
    return [Site(np.concatenate([s.x for s in sites]), np.concatenate([s.y for s in sites]),
                 np.concatenate([s.offset for s in sites]), np.any([s.available for s in sites], axis=0))]


def run_fit(key, sites, k, method, rounds=150, l1=2e-6, resume=False):
    cache = OUTPUT / "fits" / f"{key}.npz"
    meta = cache.with_suffix(".json")
    (OUTPUT / "fits").mkdir(exist_ok=True)
    fit_config = dict(method=method, rounds=rounds, steps=5, l1=l1, psi=12.,
                      lr=.12 if method == "FedAvg" else .035)
    input_hash = hashlib.sha256()
    for site in sites:
        for array in (site.x, site.y, site.offset, site.available):
            input_hash.update(np.ascontiguousarray(array).tobytes())
    identity = dict(fit_config=fit_config, input_sha256=input_hash.hexdigest(), core_sha256=digest(Path(__file__).with_name("nb_core.py")))
    if cache.exists():
        if not resume:
            raise FileExistsError(f"Use --resume to verify and reuse this versioned fit: {cache}")
        saved = json.loads(meta.read_text())
        if saved["identity"] != identity:
            raise ValueError(f"Cached input/config/code changed: {cache}")
        return np.load(cache)["coef"], saved["seconds"]
    coef, history, seconds = fit(sites, k, **fit_config)
    np.savez_compressed(cache, coef=coef)
    meta.write_text(json.dumps(dict(identity=identity, seconds=seconds, history=history), indent=2) + "\n")
    print(f"{key}: {seconds:.2f}s, NLL={history[-1]['nb_nll']:.4f}", flush=True)
    return coef, seconds


def simulation(resume=False):
    rows, pairs = [], []
    for rep in range(3):
        for condition in CONFIG["conditions"]:
            clients, genes, cell_types, truth = load_sim(rep, condition)
            k = len(cell_types)
            families = ["linear"] if condition == "rotation90" else FAMILIES
            for offset in CONFIG["offsets"]:
                for method in ("AP-FedAdam", "FedAvg", "Pooled"):
                    values, seconds, coefficient_metrics = [], 0., {}
                    for family in families:
                        sites, sd = make_sites(clients, family, offset)
                        model_sites = pooled_sites(sites) if method == "Pooled" else sites
                        key = f"sim_r{rep}_{condition}_{offset}_{method}_{family}"
                        coef, elapsed = run_fit(key, model_sites, k, method, resume=resume)
                        seconds += elapsed
                        values.append(acat(diagonal_score_p(coef, model_sites, k), axis=0))
                        if family == "linear":
                            reference, _, _, _ = load_sim(rep, "complete")
                            reference_sd = np.concatenate([c["coords"] for c in reference]).std(0, ddof=1)
                            target = truth * (np.sqrt(12) * reference_sd)[:, None, None]
                            estimate = coef[k:].reshape(truth.shape) * (reference_sd / sd)[:, None, None]
                            coefficient_metrics = dict(mae=float(np.abs(estimate - target).mean()),
                                coefficient_r=float(np.corrcoef(estimate.ravel(), target.ravel())[0, 1]))
                    p = acat(np.stack(values), axis=0).ravel()
                    q = bh(p)
                    active = np.any(truth != 0, axis=0).ravel().astype(int)
                    row = dict(repeat=rep, condition=condition, offset=offset, method=method,
                               families=len(families), elapsed_seconds=seconds,
                               **coefficient_metrics, **pair_metrics(active, p, q))
                    rows.append(row)
                    pair = pd.DataFrame(dict(gene_id=np.tile(genes, k), cell_type=np.repeat(cell_types, len(genes)),
                                             active=active, pair_p=p, pair_q=q))
                    for name in ("repeat", "condition", "offset", "method"):
                        pair[name] = row[name]
                    pairs.append(pair)
                    pd.DataFrame(rows).to_csv(OUTPUT / "offset_simulation_per_repeat.csv", index=False)
                    pd.concat(pairs).to_csv(OUTPUT / "offset_simulation_pairs.csv", index=False)
    frame = pd.DataFrame(rows)
    summarize(frame, ["condition", "offset", "method"]).to_csv(OUTPUT / "offset_simulation_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["metrics", "simulation", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT / "run_manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if json.loads(json.dumps(CONFIG)) != old["config"]:
            raise ValueError("Existing output directory has a different experiment configuration")
        INPUTS.update(old["inputs"])
    write_manifest()
    if args.stage in ("metrics", "all"):
        metric_audit()
    if args.stage in ("simulation", "all"):
        simulation(args.resume)
    write_manifest()


if __name__ == "__main__":
    main()
