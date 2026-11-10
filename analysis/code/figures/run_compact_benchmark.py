"""Compact controlled CTSV benchmark (NumPy, torch-free), used for the experiments
that require many repeated federated fits: coordinate-frame sensitivity, the
FedAvg optimizer comparison, and the centralized baseline.

The benchmark uses sparse coefficients with signal magnitudes comparable to the
large-scale recovery experiment. Dispersion is fixed at its data-generating value.

Outputs (overwrite the simulation source_data so figures/tables stay consistent):
  simulation_summary.csv, simulation_training_history.csv,
  selected_coefficient_matrices.csv
plus centralized_baseline summary and the LaTeX tables.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "04_数值实验"
SIM_SRC = ROOT / "overleaf" / "figs" / "simulation_robustness" / "source_data"
CENT_DIR = ROOT / "overleaf" / "figs" / "centralized_baseline"
TABLES = ROOT / "overleaf" / "tables"

CLIENTS, SPOTS, G, K, L = 3, 100, 160, 5, 2
ROUNDS, E = 150, 5
LR, L1, DRIFT, PSI = 0.035, 2e-6, 0.05, 12.0
FEDAVG_LR_GRID = [0.02, 0.035, 0.05, 0.08, 0.12]
SEED, REPEATS = 20260711, 3
TUNING_SEED = 20260710
B1, B2, EPS = 0.9, 0.99, 1e-8
TARGET = 1
MISSING_TYPES = [1, 3]
ROT_ANGLES = [30.0, 60.0, 90.0]
CANDID = [91, 3, 87, 157, 23]

METHODS = {"AP-FedAdam": dict(adaptive=True, prox=True, drift=True),
           "FedAvg": dict(adaptive=False, prox=False, drift=False)}


def basis(coords):
    return math.sqrt(12.0) * (coords - 0.5)


def rotate(coords, deg):
    if deg == 0:
        return coords.copy()
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.array([0.5, 0.5])
    return (coords - c) @ R.T + c


def make_beta(rng):
    beta = np.zeros((L, K, G)); active = max(12, G // 12)
    for k in range(K):
        cand = rng.permutation(G)[:active]
        mag = 2.2 + 1.6 * rng.random((L, active))
        sign = np.where(rng.random((L, active)) > 0.5, 1.0, -1.0)
        beta[:, k, cand] = mag * sign
    return beta


def gen(rng):
    theta = 0.4 + 0.1 * rng.standard_normal((1, K, G))
    tb = make_beta(rng); conc = np.full(K, 0.65)
    clients = []
    for _ in range(CLIENTS):
        coords = rng.random((SPOTS, 2)); om = rng.dirichlet(conc, size=SPOTS)
        sp = np.einsum('nl,lkg->nkg', basis(coords), tb)
        lm = np.sum(om[:, :, None] * (theta[0][None] + sp), 1)
        mu = np.exp(np.clip(lm, -4, 4)); p = np.clip(PSI / (PSI + mu), 1e-6, 1 - 1e-6)
        y = rng.negative_binomial(PSI, p).astype(float)
        clients.append({"y": y, "coords": coords, "om": om, "n": SPOTS})
    return clients, tb


def apply_condition(clients, cond):
    out = []
    active = np.ones((CLIENTS, K))
    for i, c in enumerate(clients):
        coords = c["coords"].copy(); om = c["om"].copy()
        if cond.startswith("rotation") and i == TARGET:
            coords = rotate(coords, float(cond.replace("rotation", "")))
        if cond == "missing" and i == TARGET:
            om[:, MISSING_TYPES] = 0.0
            om = om / om.sum(1, keepdims=True).clip(1e-8)
            active[i, MISSING_TYPES] = 0.0
        out.append({"y": c["y"], "h": basis(coords), "om": om, "n": c["n"]})
    return out, active


def init_state(clients):
    y = np.concatenate([c["y"] for c in clients], 0)
    theta = np.log(y.mean(0) + 0.2)[None, None, :].repeat(K, 1)
    return {"theta": theta.copy(), "beta": np.zeros((L, K, G))}


def grads(s, c):
    sp = np.einsum('nl,lkg->nkg', c["h"], s["beta"])
    lm = np.sum(c["om"][:, :, None] * (s["theta"][0][None] + sp), 1)
    mu = np.exp(np.clip(lm, -7, 6)); r = PSI * (c["y"] - mu) / (PSI + mu); sc = 1.0 / (c["n"] * G)
    return {"theta": -sc * np.einsum('nk,ng->kg', c["om"], r)[None],
            "beta": -sc * np.einsum('nl,nk,ng->lkg', c["h"], c["om"], r)}


def local(s0, c, cfg, drift, active_row, lr):
    s = {k: v.copy() for k, v in s0.items()}
    m = {k: np.zeros_like(v) for k, v in s.items()}; v = {k: np.zeros_like(x) for k, x in s.items()}
    for e in range(E):
        g = grads(s, c)
        if active_row is not None:
            drop = np.where(active_row == 0)[0]
            if len(drop):
                g["theta"][:, drop, :] = 0.0; g["beta"][:, drop, :] = 0.0
        for nm in s:
            gg = g[nm]
            if cfg["adaptive"]:
                m[nm] = B1 * m[nm] + (1 - B1) * gg; v[nm] = B2 * v[nm] + (1 - B2) * gg ** 2
                mh = m[nm] / (1 - B1 ** (e + 1)); vh = v[nm] / (1 - B2 ** (e + 1)); d = mh / (np.sqrt(vh) + EPS)
            else:
                vh = np.ones_like(s[nm]); d = gg
            if cfg["drift"] and drift is not None:
                d = d + DRIFT * drift[nm]
            p = s[nm] - lr * (0.98 ** e) * d
            if cfg["prox"] and nm == "beta":
                thr = lr * (0.98 ** e) * L1 / (np.sqrt(vh) + EPS)
                s[nm] = np.sign(p) * np.maximum(np.abs(p) - thr, 0)
            else:
                s[nm] = p
    return {k: s[k] - s0[k] for k in s0}


def _loss(clients, s):
    tot = 0.0
    for c in clients:
        sp = np.einsum('nl,lkg->nkg', c["h"], s["beta"]); lm = np.sum(c["om"][:, :, None] * (s["theta"][0][None] + sp), 1)
        mu = np.exp(np.clip(lm, -7, 6)); denom = PSI + mu + 1e-8
        logp = PSI * (np.log(PSI + 1e-8) - np.log(denom)) + c["y"] * (np.log(mu + 1e-8) - np.log(denom))
        tot += -logp.mean()
    return tot / len(clients)


def run(clients, active, cfg, lr=LR, record_loss=False):
    s = init_state(clients); drift = {k: np.zeros_like(v) for k, v in s.items()}
    w = np.array([c["n"] for c in clients], float); w /= w.sum()
    losses = []
    update_norm = 0.0
    for rd in range(ROUNDS):
        deltas = [local(s, c, cfg, drift, active[i], lr) for i, c in enumerate(clients)]
        agg = {}
        for k in s:
            if k in ("theta", "beta"):
                num = sum(w[i] * deltas[i][k] * active[i][None, :, None] for i in range(len(clients)))
                den = sum(w[i] * active[i][None, :, None] for i in range(len(clients)))
                agg[k] = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
            else:
                agg[k] = sum(w[i] * deltas[i][k] for i in range(len(clients)))
        s = {k: s[k] + agg[k] for k in s}; drift = agg
        update_norm = float(np.sqrt(sum(np.sum(delta ** 2) for delta in agg.values())))
        if record_loss:
            losses.append(_loss(clients, s))
    return s, losses, {"final_loss": float(_loss(clients, s)), "final_update_norm": update_norm}


def metrics(bh, tb):
    p, t = bh.ravel(), tb.ravel(); d = p - t
    return dict(mae=float(np.mean(np.abs(d))), rmse=float(np.sqrt(np.mean(d ** 2))),
               pearson=float(np.corrcoef(p, t)[0, 1]),
               support_recall=float(np.mean(np.abs(p[np.abs(t) > 1e-8]) > 0.10)))


CONDITIONS = ["complete", "missing", "rotation30", "rotation60", "rotation90"]


def select_fedavg_lr():
    """Tune FedAvg on an independent complete-input simulation draw."""
    base, _ = gen(np.random.default_rng(TUNING_SEED))
    clients, active = apply_condition(base, "complete")
    results = []
    for lr in FEDAVG_LR_GRID:
        state, _, _ = run(clients, active, METHODS["FedAvg"], lr=lr)
        results.append({"learning_rate": float(lr), "final_nb_loss": float(_loss(clients, state))})
    best = min(results, key=lambda x: (x["final_nb_loss"], x["learning_rate"]))
    return best["learning_rate"], results


def write_manifest(fedavg_lr, tuning_results, candidates):
    manifest = {
        "benchmark": "compact_nb_ctsv",
        "clients": CLIENTS,
        "spots_per_client": SPOTS,
        "genes": G,
        "cell_types": K,
        "spatial_bases": L,
        "rounds": ROUNDS,
        "local_steps": E,
        "repeats": REPEATS,
        "seed": SEED,
        "dispersion": PSI,
        "ap_lr": LR,
        "fedavg_lr": fedavg_lr,
        "fedavg_lr_grid": FEDAVG_LR_GRID,
        "fedavg_tuning_seed": TUNING_SEED,
        "fedavg_tuning_criterion": "minimum final NB working loss on an independent complete-input draw",
        "fedavg_tuning_results": tuning_results,
        "l1_lambda": L1,
        "drift_scale": DRIFT,
        "candidate_genes": candidates,
        "cell_type_labels": [f"CT{k}" for k in range(K)],
        "conditions": [
            {"key": "complete", "label": "Complete", "missing_types": [], "rotation_deg": 0.0},
            {"key": "missing", "label": "Client 1 missing CT1 + CT3", "missing_types": MISSING_TYPES, "rotation_deg": 0.0},
            {"key": "rotation30", "label": "Client 1 rotated 30 degrees", "missing_types": [], "rotation_deg": 30.0},
            {"key": "rotation60", "label": "Client 1 rotated 60 degrees", "missing_types": [], "rotation_deg": 60.0},
            {"key": "rotation90", "label": "Client 1 rotated 90 degrees", "missing_types": [], "rotation_deg": 90.0},
        ],
    }
    (SIM_SRC / "simulation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    SIM_SRC.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    fedavg_lr, tuning_results = select_fedavg_lr()
    print(f"FedAvg selected learning rate: {fedavg_lr:g}")
    summ, betarows, histrows, centrows = [], [], [], []
    FED_CONDS = {"complete", "missing", "rotation90"}
    HIST_CONDS = {"complete", "missing", "rotation90"}
    candid = None
    for rep in range(REPEATS):
        rng = np.random.default_rng(SEED + rep)
        base, tb = gen(rng)
        if candid is None:
            strength = np.sum(np.abs(tb), axis=0)
            candid = []
            for k in range(K):
                for g in np.argsort(strength[k])[::-1]:
                    if int(g) not in candid:
                        candid.append(int(g)); break
            globals()["CANDID"] = candid
        for cond in CONDITIONS:
            clients, active = apply_condition(base, cond)
            for method in ["AP-FedAdam", "FedAvg"]:
                if method == "FedAvg" and cond not in FED_CONDS:
                    continue
                rl = (rep == 0 and cond in HIST_CONDS)
                lr = LR if method == "AP-FedAdam" else fedavg_lr
                s, losses, diagnostics = run(clients, active, METHODS[method], lr=lr, record_loss=rl)
                mt = metrics(s["beta"], tb)
                mt.update(diagnostics)
                summ.append(dict(method=method, **mt, repeat=rep, seed=SEED + rep, condition=cond,
                                 condition_label=cond, rotation_deg=(float(cond.replace("rotation", "")) if cond.startswith("rotation") else 0.0),
                                 missing_cell_types=(",".join(str(x + 1) for x in MISSING_TYPES) if cond == "missing" else "")))
                if rl:
                    for rd, lv in enumerate(losses, 1):
                        histrows.append(dict(round=rd, loss=float(lv), update_norm=0.0, repeat=0, seed=SEED,
                                             condition=cond, method=method))
                if rep == 0 and method == "AP-FedAdam":
                    for g in candid:
                        for l in range(L):
                            for k in range(K):
                                betarows.append(dict(condition=cond, method=method, repeat=rep, gene=g,
                                                     basis=l + 1, cell_type=k,
                                                     true_beta=float(tb[l, k, g]), learned_beta=float(s["beta"][l, k, g])))
        cc, ac = apply_condition(base, "complete")
        pooled = [{"y": np.concatenate([c["y"] for c in cc]), "h": np.concatenate([c["h"] for c in cc]),
                   "om": np.concatenate([c["om"] for c in cc]), "n": CLIENTS * SPOTS}]
        sc, _, diagnostics = run(pooled, np.ones((1, K)), METHODS["AP-FedAdam"], lr=LR)
        mt = metrics(sc["beta"], tb)
        mt.update(diagnostics)
        centrows.append(dict(repeat=rep, method="Centralized", **mt))
        print(f"rep {rep} done")

    dfs = pd.DataFrame(summ); dfs.to_csv(SIM_SRC / "simulation_summary.csv", index=False)
    pd.DataFrame(betarows).to_csv(SIM_SRC / "selected_coefficient_matrices.csv", index=False)
    pd.DataFrame(histrows).to_csv(SIM_SRC / "simulation_training_history.csv", index=False)

    cent = pd.DataFrame(centrows)
    fed = dfs[dfs.condition == "complete"]
    def ms(df, col):
        return f"${df[col].mean():.4f}\\pm{df[col].std():.4f}$"
    print("\n=== complete-input summary (mean+/-sd) ===")
    for name, sub in [("Centralized (pooled)", cent),
                      ("AP-FedAdam (federated)", fed[fed.method == 'AP-FedAdam']),
                      ("FedAvg (federated)", fed[fed.method == 'FedAvg'])]:
        print(f"{name:24s} MAE {sub.mae.mean():.4f} r {sub.pearson.mean():.3f} recall {sub.support_recall.mean():.3f}")
    CENT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([cent.assign(setting='Centralized'),
               fed[fed.method == 'AP-FedAdam'].assign(setting='AP-FedAdam'),
               fed[fed.method == 'FedAvg'].assign(setting='FedAvg')]).to_csv(CENT_DIR / "centralized_baseline_repeats.csv", index=False)
    summary_rows = []
    for setting, sub in [("Centralized", cent),
                         ("AP-FedAdam", fed[fed.method == "AP-FedAdam"]),
                         ("FedAvg", fed[fed.method == "FedAvg"])]:
        summary = {"method": setting, "n_repeats": int(len(sub))}
        for metric in ["mae", "pearson", "support_recall"]:
            summary[f"{metric}_mean"] = float(sub[metric].mean())
            summary[f"{metric}_sd"] = float(sub[metric].std())
        summary_rows.append(summary)
    pd.DataFrame(summary_rows).to_csv(CENT_DIR / "centralized_baseline_summary.csv", index=False)

    def row(label, sub):
        return f"{label} & {ms(sub,'mae')} & {ms(sub,'pearson')} & {ms(sub,'support_recall')} \\\\"
    cent_tex = r"""\begin{table}[htbp]
\centering
\small
\caption{\textbf{Matched centralized-versus-federated comparison on the compact complete-input CTSV benchmark.} Centralized pools all spots into a single node; the federated methods partition the same spots across three clients with server aggregation. All rows share the identical data-generating process, model, optimizer schedule, and compute budget, with dispersion fixed at its data-generating value. Values are mean $\pm$ s.d. across three independently generated data sets.}
\label{tab:centralized-baseline}
\begin{tabular}{lrrr}
\toprule
Setting & Coefficient MAE & Pearson $r$ & Support recall \\
\midrule
""" + row("Centralized (pooled)", cent) + "\n" + \
        row("AP-FedAdam (federated)", fed[fed.method == 'AP-FedAdam']) + "\n" + \
        row("FedAvg (federated)", fed[fed.method == 'FedAvg']) + "\n" + \
        r"""\bottomrule
\end{tabular}
\end{table}
"""
    (TABLES / "table_centralized_baseline.tex").write_text(cent_tex)

    ap = dfs[dfs.method == 'AP-FedAdam']
    def cell(cond, col):
        s = ap[ap.condition == cond]
        return f"${s[col].mean():.4f}\\pm{s[col].std():.4f}$"
    rot_tex = r"""\begin{table}[htbp]
\centering
\small
\caption{\textbf{Coordinate-frame sensitivity of AP-FedAdam on the compact benchmark.} One of three clients was fitted after the indicated rotation while counts were generated in the common canonical coordinate frame. Values are mean $\pm$ s.d. across three independently generated data sets.}
\label{tab:coordinate-rotation}
\begin{tabular}{lrrr}
\toprule
Rotation angle & Coefficient MAE & Pearson $r$ & Support recall \\
\midrule
""" + \
        f"$0^\\circ$ & {cell('complete','mae')} & {cell('complete','pearson')} & {cell('complete','support_recall')} \\\\\n" + \
        f"$30^\\circ$ & {cell('rotation30','mae')} & {cell('rotation30','pearson')} & {cell('rotation30','support_recall')} \\\\\n" + \
        f"$60^\\circ$ & {cell('rotation60','mae')} & {cell('rotation60','pearson')} & {cell('rotation60','support_recall')} \\\\\n" + \
        f"$90^\\circ$ & {cell('rotation90','mae')} & {cell('rotation90','pearson')} & {cell('rotation90','support_recall')} \\\\\n" + \
        r"""\bottomrule
\end{tabular}
\end{table}
"""
    (TABLES / "table_coordinate_rotation.tex").write_text(rot_tex)

    def cc(cond, method, col):
        s = dfs[(dfs.condition == cond) & (dfs.method == method)]
        return f"${s[col].mean():.4f}\\pm{s[col].std():.4f}$"
    rows_ = [("Complete", "complete"), ("Missing CT1/CT3", "missing"), ("Rotation $90^\\circ$", "rotation90")]
    fed_caption = ("\\textbf{AP-FedAdam versus FedAvg on the compact three-client CTSV benchmark.} "
                   "Values are mean $\\pm$ s.d. across three independently generated data sets. "
                   "FedAvg used learning rate $%.3g$, selected by minimum final NB working loss on an "
                   "independent complete-input tuning draw from the grid $\\{0.02,0.035,0.05,0.08,0.12\\}$. "
                   "One client had a missing CT1 and CT3 input panel in the incomplete condition. The "
                   "rotation row is reported for transparency; it tests coordinate registration rather than "
                   "an optimizer advantage." % fedavg_lr)
    fed_tex = r"""\begin{table}[htbp]
\centering
\small
\caption{""" + fed_caption + r"""}
\label{tab:simulation-federated-comparison}
\begin{tabular}{lrrrr}
\toprule
Condition & AP-FedAdam MAE & FedAvg MAE & AP-FedAdam $r$ & FedAvg $r$ \\
\midrule
""" + "\n".join(
        f"{lab} & {cc(cd,'AP-FedAdam','mae')} & {cc(cd,'FedAvg','mae')} & {cc(cd,'AP-FedAdam','pearson')} & {cc(cd,'FedAvg','pearson')} \\\\"
        for lab, cd in rows_) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
"""
    (TABLES / "table_simulation_ap_vs_fedavg.tex").write_text(fed_tex)
    write_manifest(fedavg_lr, tuning_results, candid)
    print("wrote source-data CSVs, manifest, centralized summary, and 3 tables")


if __name__ == "__main__":
    main()
