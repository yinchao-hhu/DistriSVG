"""Frozen, resumable inference-calibration and full-panel completion runs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import spearmanr
import h5py

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[2]
sys.path.insert(0, str(CODE.parent / "estimation"))
import run_revision as old
from metrics import acat, bh, pair_metrics
from nb_core import fit, diagonal_score_p
from joint_refit import gene_sites, refit_gene

OUT = ROOT / "04_数值实验/results/distrisvg_completion_2026-09-25"
CONFIG = dict(version="2026-09-25-v1", null_repeats=100, alternative_repeats=50,
              calibration_genes=40, calibration_seed=20260925, psi=12.,
              families=list(old.FAMILIES), refit="unpenalized full coefficient and log-dispersion",
              covariance="observed joint Hessian; spot-independent HC1 sandwich",
              failures="flagged p=1; retained in denominator", q=.05,
              multiplicity=["acat_bh", "acat_by", "bonferroni_bh", "bonferroni_by"],
              scenarios=["complete", "missing_all", "missing_complete_sites"],
              real_panels=dict(PDAC=4740, Mouse=300), real_offsets=["none", "library"],
              clipping=False, l1=2e-6, rounds=150, steps=5, ap_lr=.035)


def sha(path):
    return old.digest(path)


def array_hash(sites):
    h = hashlib.sha256()
    for site in sites:
        for array in (site.x, site.y, site.offset, site.available):
            h.update(str(array.shape).encode())
            h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "run_manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["config"] != CONFIG:
            raise ValueError("Versioned experiment configuration changed")
    write_manifest()


def write_manifest():
    path = OUT / "run_manifest.json"
    with (OUT / ".manifest.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        inputs = json.loads(path.read_text()).get("inputs", {}) if path.exists() else {}
        for name, value in old.INPUTS.items():
            if name in inputs and inputs[name] != value:
                raise ValueError(f"Input hash changed: {name}")
            inputs[name] = value
        versions = {p: importlib.metadata.version(p) for p in
                    ("numpy", "scipy", "pandas", "statsmodels", "joblib", "scikit-learn", "h5py")}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(config=CONFIG, inputs=inputs, packages=versions,
                        code_sha256={p.name: sha(p) for p in CODE.glob("*.py")}, python=sys.version), indent=2) + "\n")
        temporary.replace(path)


def cached_refit(key, sites, k, genes, workers=4, estimate=True):
    directory = OUT / "refits"
    directory.mkdir(exist_ok=True)
    identity = dict(input_sha256=array_hash(sites), core_sha256=sha(CODE / "joint_refit.py"),
                    estimated_dispersion=estimate, fixed_psi=12., genes=list(genes))
    records, parameters, pm, ps = [], [], [], []
    for start in range(0, len(genes), 80):
        end = min(start + 80, len(genes))
        file = directory / f"{key}_g{start:04d}-{end:04d}.npz"
        meta = file.with_suffix(".json")
        audit = file.with_suffix(".csv")
        chunk_identity = dict(identity, gene_start=start, gene_end=end)
        if file.exists():
            if json.loads(meta.read_text())["identity"] != chunk_identity:
                raise ValueError(f"Cached identity mismatch: {file}")
            data = np.load(file)
            parameters.append(data["parameter"])
            pm.append(data["p_model"])
            ps.append(data["p_sandwich"])
            records.append(pd.read_csv(audit))
            continue
        results = Parallel(n_jobs=workers, max_nbytes="2M", temp_folder="/private/tmp")(
            delayed(refit_gene)(gene_sites(sites, gene), k, estimate_dispersion=estimate)
            for gene in range(start, end))
        parameter = np.stack([x["parameter"] for x in results])
        p_model = np.stack([x["p_model"] for x in results])
        p_sandwich = np.stack([x["p_sandwich"] for x in results])
        detail = pd.DataFrame([{name: value for name, value in result.items()
                                if name not in ("parameter", "p_model", "p_sandwich")}
                               for result in results])
        detail.insert(0, "gene_id", genes[start:end])
        detail.to_csv(audit, index=False)
        np.savez_compressed(file, parameter=parameter, p_model=p_model, p_sandwich=p_sandwich)
        meta.write_text(json.dumps(dict(identity=chunk_identity), indent=2) + "\n")
        parameters.append(parameter)
        pm.append(p_model)
        ps.append(p_sandwich)
        records.append(detail)
        print(f"{key}: genes {start}-{end}; valid {int(detail.valid.sum())}/{len(detail)}", flush=True)
    return np.concatenate(parameters), np.concatenate(pm), np.concatenate(ps), pd.concat(records, ignore_index=True)


def combine(pvalues):
    p = np.stack(pvalues)
    cauchy = acat(p, axis=0).ravel()
    bonferroni = np.minimum(1., p.shape[0] * np.min(p, axis=0)).ravel()
    output = {}
    for name, raw in (("acat", cauchy), ("bonferroni", bonferroni)):
        q = bh(raw)
        harmonic = np.sum(1 / np.arange(1, len(raw) + 1))
        output[name + "_bh"] = (raw, q)
        output[name + "_by"] = (raw, np.minimum(1., q * harmonic))
    return output


def archived(workers):
    metrics, pair_tables, audits = [], [], []
    for rep in range(3):
        for condition in ("complete", "missing", "rotation90"):
            clients, genes, types, truth = old.load_sim(rep, condition)
            k = len(types)
            families = ["linear"] if condition == "rotation90" else old.FAMILIES
            policies = ["all"] + (["complete_sites"] if condition == "missing" else [])
            for policy in policies:
                selected = [c for c in clients if np.all(np.any(c["omega"] > 0, axis=0))] if policy == "complete_sites" else clients
                values = {"model": [], "sandwich": []}
                for family in families:
                    sites, _ = old.make_sites(clients, family, "none")
                    if policy == "complete_sites":
                        sites = [s for s in sites if s.available.all()]
                    key = f"archive_r{rep}_{condition}_{policy}_{family}"
                    _, model, robust, audit = cached_refit(key, sites, k, genes, workers)
                    values["model"].append(model)
                    values["sandwich"].append(robust)
                    audit = audit.assign(repeat=rep, condition=condition, policy=policy, family=family)
                    audits.append(audit)
                active = np.any(truth != 0, axis=0).T.ravel().astype(int)
                for covariance, family_values in values.items():
                    for adjustment, (p, q) in combine(family_values).items():
                        common = dict(repeat=rep, condition=condition, policy=policy,
                                      covariance=covariance, adjustment=adjustment)
                        metrics.append(dict(common, **pair_metrics(active, p, q)))
                        pair_tables.append(pd.DataFrame(dict(common, gene_id=np.repeat(genes, k),
                            cell_type=np.tile(types, len(genes)), active=active, pair_p=p, pair_q=q)))
                pd.DataFrame(metrics).to_csv(OUT / "archive_refit_metrics.csv", index=False)
                pd.concat(pair_tables).to_csv(OUT / "archive_refit_pairs.csv", index=False)
                pd.concat(audits).to_csv(OUT / "archive_refit_audit.csv", index=False)
    write_manifest()


def calibration(workers, maximum=None):
    base, _, types, _ = old.load_sim(0, "complete")
    k, g = len(types), CONFIG["calibration_genes"]
    genes = [f"calib_{i:03d}" for i in range(g)]
    rows, audits, p_tables = [], [], []
    out = OUT / "calibration"
    out.mkdir(exist_ok=True)
    for signal, number in (("null", CONFIG["null_repeats"]), ("sparse", CONFIG["alternative_repeats"])):
        number = min(number, maximum) if maximum is not None else number
        for rep in range(number):
            checkpoint = out / f"{signal}_{rep:03d}_metrics.csv"
            if checkpoint.exists():
                rows.append(pd.read_csv(checkpoint, keep_default_na=False, na_values=[""]))
                continue
            rng = np.random.default_rng(np.random.SeedSequence([CONFIG["calibration_seed"], signal == "sparse", rep]))
            alpha = .4 + .1 * rng.normal(size=(k, g))
            beta = np.zeros((2, k, g))
            if signal == "sparse":
                for ct in range(k):
                    active_genes = rng.choice(g, 4, replace=False)
                    beta[:, ct, active_genes] = rng.choice([-1., 1.], size=(2, 4)) * rng.uniform(2.2, 3.8, (2, 4))
            generated = []
            for c in base:
                h = np.sqrt(12.) * (c["coords"] - .5)
                x = np.concatenate([c["omega"], c["omega"] * h[:, :1], c["omega"] * h[:, 1:]], axis=1)
                mu = np.exp(x @ np.vstack([alpha, beta.reshape(2 * k, g)]))
                y = rng.negative_binomial(12., 12. / (12. + mu)).astype(float)
                generated.append(dict(c, y=y, library=y.sum(1)))
            np.savez_compressed(out / f"{signal}_{rep:03d}_input.npz", y=np.concatenate([c["y"] for c in generated]),
                                alpha=alpha, beta=beta)
            active = np.any(beta != 0, axis=0).T.ravel().astype(int)
            repeat_rows, repeat_audit, repeat_pairs = [], [], []
            for policy in CONFIG["scenarios"]:
                clients = [dict(c, omega=c["omega"].copy()) for c in generated]
                if policy != "complete":
                    clients[1]["omega"][:, [1, 3]] = 0
                    clients[1]["omega"] /= clients[1]["omega"].sum(1, keepdims=True)
                values = {"model": [], "sandwich": []}
                known = []
                for family in old.FAMILIES:
                    sites, _ = old.make_sites(clients, family, "none")
                    if policy == "missing_complete_sites":
                        sites = [s for s in sites if s.available.all()]
                    key = f"calib_{signal}_r{rep}_{policy}_{family}"
                    _, model, robust, audit = cached_refit(key, sites, k, genes, workers)
                    values["model"].append(model)
                    values["sandwich"].append(robust)
                    repeat_audit.append(audit.assign(signal=signal, repeat=rep, policy=policy, family=family))
                    if family == "linear":
                        _, known_model, known_robust, known_audit = cached_refit(key + "_knownpsi", sites, k, genes, workers, estimate=False)
                        for name, raw in (("knownpsi_model_linear", known_model), ("knownpsi_sandwich_linear", known_robust)):
                            p = raw.ravel()
                            q = bh(p)
                            repeat_rows.append(dict(signal=signal, repeat=rep, policy=policy,
                                covariance=name, adjustment="bh", **pair_metrics(active, p, q),
                                null_p_below05=float(np.mean(p[active == 0] < .05)),
                                any_discovery=bool(np.any(q < .05))))
                for cov, family_values in values.items():
                    for adjustment, (p, q) in combine(family_values).items():
                        common = dict(signal=signal, repeat=rep, policy=policy, covariance=cov, adjustment=adjustment)
                        repeat_rows.append(dict(common, **pair_metrics(active, p, q),
                            null_p_below05=float(np.mean(p[active == 0] < .05)), any_discovery=bool(np.any(q < .05))))
                        repeat_pairs.append(pd.DataFrame(dict(common, gene_id=np.repeat(genes, k),
                            cell_type=np.tile(types, g), active=active, pair_p=p, pair_q=q)))
                print(f"calibration {signal} repeat {rep}: {policy} complete", flush=True)
            pd.DataFrame(repeat_rows).to_csv(checkpoint, index=False)
            pd.concat(repeat_audit).to_csv(out / f"{signal}_{rep:03d}_audit.csv", index=False)
            pd.concat(repeat_pairs).to_csv(out / f"{signal}_{rep:03d}_pairs.csv", index=False)
            rows.append(pd.DataFrame(repeat_rows))
            pd.concat(rows).to_csv(OUT / "calibration_metrics.csv", index=False)
    pd.concat(rows).to_csv(OUT / "calibration_metrics.csv", index=False)
    write_manifest()


def load_real(dataset):
    if dataset == "PDAC":
        clients, genes, types = [], None, None
        root = ROOT / "04_数值实验/code/pdac_data_to_yc"
        for name in (f"PDAC_{p}_ST{i}" for p in "AB" for i in range(1, 4)):
            expr = old.read(root / "preprocessed_data" / f"{name}_processed_data_mincell5.csv").set_index("Spot_ID")
            omega = old.read(root / "spotlight_out" / f"{name}_SPOTlight_W_omega_with_xy.csv").set_index("spot")
            names = [c for c in expr.columns if c not in ("x", "y", "X", "Y")]
            if genes is None:
                genes = names
                types = [c for c in omega.columns if c not in ("x", "y")]
            assert names == genes and expr.index.is_unique and omega.index.is_unique
            ids = expr.index[expr.index.isin(omega.index)]
            expr, omega = expr.loc[ids], omega.loc[ids]
            om = omega[types].to_numpy(float)
            om /= om.sum(1, keepdims=True)
            clients.append(dict(y=expr[genes].to_numpy(float), omega=om,
                coords=omega[["x", "y"]].to_numpy(float), library=expr[genes].sum(1).to_numpy(float), section=name))
    else:
        root = ROOT / "04_数值实验"
        panel = old.read(root / "figures/mouse_reference_deconv/mouse_reference_nnls_gene_panel.csv")
        genes = panel.gene.iloc[:300].tolist()
        types = ["Astrocyte", "Endothelial Cell", "GABA-ergic Neuron", "Glutamatergic Neuron", "Microglia", "Oligodendrocyte", "Oligodendrocyte Precursor Cell"]
        clients = []
        for name, section in {"Anterior": "V1_Mouse_Brain_Sagittal_Anterior", "Posterior": "V1_Mouse_Brain_Sagittal_Posterior", "Anterior2": "V1_Mouse_Brain_Sagittal_Anterior_Section_2"}.items():
            path = old.track(root / "data" / section / "filtered_feature_bc_matrix.h5")
            with h5py.File(path, "r") as f:
                m = f["matrix"]
                counts = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
                symbols = pd.Index([x.decode() for x in m["features/name"][:]])
                barcodes = pd.Index([x.decode() for x in m["barcodes"][:]])
            omega = old.read(root / "figures/mouse_reference_deconv" / f"mouse_reference_nnls_omega_{name}.csv").set_index("spot_id")
            cols = barcodes.get_indexer(omega.index)
            rows = [int(np.flatnonzero(symbols == gene)[0]) for gene in genes]
            assert np.all(cols >= 0)
            om = omega[types].to_numpy(float)
            om /= om.sum(1, keepdims=True)
            clients.append(dict(y=counts[rows][:, cols].T.toarray().astype(float), omega=om,
                coords=omega[["x", "y"]].to_numpy(float), library=np.asarray(counts[:, cols].sum(0)).ravel(), section=name))
    if len(genes) != CONFIG["real_panels"][dataset]:
        raise ValueError("Unexpected full-panel size")
    return clients, genes, types


def real_data(workers, which="both"):
    metrics, all_pairs, all_audits = [], [], []
    for dataset in (["PDAC", "Mouse"] if which == "both" else [which]):
        clients, genes, types = load_real(dataset)
        k = len(types)
        pd.DataFrame({"gene_id": genes}).to_csv(OUT / f"{dataset}_full_gene_panel.csv", index=False)
        for offset in CONFIG["real_offsets"]:
            values = {"AP_diagonal": [], "refit_model": [], "refit_sandwich": []}
            for family in old.FAMILIES:
                sites, _ = old.make_sites(clients, family, offset)
                key = f"real_{dataset}_{offset}_{family}"
                fit_dir = OUT / "ap_fits"
                fit_dir.mkdir(exist_ok=True)
                file = fit_dir / (key + ".npz")
                identity = dict(input_sha256=array_hash(sites), core_sha256=sha(Path(old.__file__).with_name("nb_core.py")),
                                config={n: CONFIG[n] for n in ("rounds", "steps", "ap_lr", "l1", "psi")})
                if file.exists():
                    if json.loads(file.with_suffix(".json").read_text())["identity"] != identity:
                        raise ValueError("AP cache differs")
                    coef = np.load(file)["coef"]
                else:
                    coef, history, seconds = fit(sites, k, rounds=150, steps=5, lr=.035, l1=2e-6, psi=12.)
                    np.savez_compressed(file, coef=coef)
                    file.with_suffix(".json").write_text(json.dumps(dict(identity=identity, history=history, seconds=seconds), indent=2) + "\n")
                values["AP_diagonal"].append(acat(diagonal_score_p(coef, sites, k), axis=0).T)
                _, model, robust, audit = cached_refit(key, sites, k, genes, workers)
                values["refit_model"].append(model)
                values["refit_sandwich"].append(robust)
                all_audits.append(audit.assign(dataset=dataset, offset=offset, family=family))
                pd.concat(all_audits).to_csv(OUT / f"real_{which}_fit_audit.csv", index=False)
            for method, family_p in values.items():
                for adjustment, (p, q) in combine(family_p).items():
                    common = dict(dataset=dataset, offset=offset, method=method, adjustment=adjustment)
                    pairs = pd.DataFrame(dict(common, gene_id=np.repeat(genes, k), cell_type=np.tile(types, len(genes)), pair_p=p, pair_q=q))
                    all_pairs.append(pairs)
                    metrics.append(dict(common, n_genes=len(genes), n_pairs=len(p),
                        selected_pairs=int(np.sum(q < .05)), selected_genes=pairs.loc[q < .05, "gene_id"].nunique()))
            pd.concat(all_pairs).to_csv(OUT / f"real_{which}_pairs.csv", index=False)
            pd.DataFrame(metrics).to_csv(OUT / f"real_{which}_metrics.csv", index=False)
            write_manifest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["archive", "calibration", "real"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum", type=int, help="Debug only: truncate calibration repeats, not final evaluation")
    parser.add_argument("--dataset", choices=["both", "PDAC", "Mouse"], default="both")
    args = parser.parse_args()
    initialize()
    if args.stage == "archive":
        archived(args.workers)
    elif args.stage == "calibration":
        calibration(args.workers, args.maximum)
    else:
        real_data(args.workers, args.dataset)


if __name__ == "__main__":
    main()
