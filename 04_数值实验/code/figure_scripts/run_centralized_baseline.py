"""Matched centralized-vs-federated CTSV baseline (NumPy, torch-free).

Purpose: quantify how close federated AP-FedAdam is to a pooled-data (centralized)
fit under an identical model, data set, optimizer, and compute budget.  The only
difference between the "Centralized" and "Federated" runs is whether the spots are
pooled into one node or partitioned across three clients with server aggregation.

Notes:
- This is a self-contained reproduction of the synthetic NB-CTSV benchmark of
  run_simulation_robustness.py, ported to NumPy so it runs without PyTorch.
- Dispersion psi is fixed at the data-generating value (the CTSV target of
  inference is the spatial coefficient tensor beta, not psi); this also removes a
  digamma dependency.  Absolute MAE is therefore on this experiment's own scale
  and is not meant to equal the PyTorch Table-2 numbers; the comparison of
  interest is Centralized vs Federated on identical data.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[2] / "overleaf" / "figs" / "centralized_baseline"

CLIENTS = 3
SPOTS = 100
GENES = 160
CELL_TYPES = 5
ROUNDS = 100
LOCAL_STEPS = 3
LR = 0.035
L1 = 0.0002
DRIFT_SCALE = 0.08
PSI_TRUE = 12.0
SEED = 20260711
REPEATS = 3

BETA1, BETA2, EPS = 0.9, 0.99, 1e-8


def basis_of(coords: np.ndarray) -> np.ndarray:
    return math.sqrt(12.0) * (coords - 0.5)


def make_sparse_beta(rng: np.random.Generator) -> np.ndarray:
    beta = np.zeros((2, CELL_TYPES, GENES))
    active = max(12, GENES // 12)
    for k in range(CELL_TYPES):
        cand = rng.permutation(GENES)[:active]
        mag = 0.70 + 0.45 * rng.random((2, active))
        sign = np.where(rng.random((2, active)) > 0.5, 1.0, -1.0)
        beta[:, k, cand] = mag * sign
    return beta


def generate_clients(rng: np.random.Generator):
    theta = 0.75 + 0.15 * rng.standard_normal((1, CELL_TYPES, GENES))
    true_beta = make_sparse_beta(rng)
    conc = np.full(CELL_TYPES, 0.65)
    clients = []
    for _ in range(CLIENTS):
        coords = rng.random((SPOTS, 2))
        omega = rng.dirichlet(conc, size=SPOTS)
        basis = basis_of(coords)
        spatial = np.einsum("nl,lkg->nkg", basis, true_beta)
        log_mu = np.sum(omega[:, :, None] * (theta[0][None] + spatial), axis=1)
        mu = np.exp(np.clip(log_mu, -4.0, 4.0))
        p = np.clip(PSI_TRUE / (PSI_TRUE + mu), 1e-6, 1 - 1e-6)
        y = rng.negative_binomial(PSI_TRUE, p).astype(float)
        clients.append({"y": y, "coords": coords, "omega": omega, "basis": basis, "n": SPOTS})
    return clients, true_beta


def pooled_client(clients):
    return [{
        "y": np.concatenate([c["y"] for c in clients], 0),
        "coords": np.concatenate([c["coords"] for c in clients], 0),
        "omega": np.concatenate([c["omega"] for c in clients], 0),
        "basis": np.concatenate([c["basis"] for c in clients], 0),
        "n": sum(c["n"] for c in clients),
    }]


def init_state(clients):
    y = np.concatenate([c["y"] for c in clients], 0)
    theta = np.log(y.mean(0) + 0.2)[None, None, :].repeat(CELL_TYPES, axis=1)
    return {"theta": theta.copy(), "beta": np.zeros((2, CELL_TYPES, GENES))}


def grads(state, c):
    theta, beta = state["theta"], state["beta"]
    spatial = np.einsum("nl,lkg->nkg", c["basis"], beta)
    log_mu = np.sum(c["omega"][:, :, None] * (theta[0][None] + spatial), axis=1)
    mu = np.exp(np.clip(log_mu, -7.0, 6.0))
    r = PSI_TRUE * (c["y"] - mu) / (PSI_TRUE + mu)
    scale = 1.0 / (c["n"] * GENES)
    gtheta = -scale * np.einsum("nk,ng->kg", c["omega"], r)[None]
    gbeta = -scale * np.einsum("nl,nk,ng->lkg", c["basis"], c["omega"], r)
    return {"theta": gtheta, "beta": gbeta}


def local_update(state, c, cfg, drift):
    s = {k: v.copy() for k, v in state.items()}
    m = {k: np.zeros_like(v) for k, v in s.items()}
    v = {k: np.zeros_like(val) for k, val in s.items()}
    for step in range(LOCAL_STEPS):
        g = grads(s, c)
        step_lr = LR * (0.98 ** step)
        for name in s:
            grad = g[name]
            if cfg["adaptive"]:
                m[name] = BETA1 * m[name] + (1 - BETA1) * grad
                v[name] = BETA2 * v[name] + (1 - BETA2) * grad ** 2
                mhat = m[name] / (1 - BETA1 ** (step + 1))
                vhat = v[name] / (1 - BETA2 ** (step + 1))
                direction = mhat / (np.sqrt(vhat) + EPS)
            else:
                vhat = np.ones_like(s[name])
                direction = grad
            if cfg["drift"] and drift is not None:
                direction = direction + DRIFT_SCALE * drift[name]
            proposal = s[name] - step_lr * direction
            if cfg["prox"] and name == "beta":
                thr = step_lr * L1 / (np.sqrt(vhat) + EPS)
                s[name] = np.sign(proposal) * np.maximum(np.abs(proposal) - thr, 0.0)
            else:
                s[name] = proposal
            s[name] = np.nan_to_num(s[name], nan=0.0, posinf=25.0, neginf=-25.0)
    delta = {k: s[k] - state[k] for k in state}
    return delta


def run(clients, cfg):
    state = init_state(clients)
    drift = {k: np.zeros_like(v) for k, v in state.items()}
    weights = np.array([c["n"] for c in clients], float)
    weights /= weights.sum()
    for _ in range(ROUNDS):
        deltas = [local_update(state, c, cfg, drift) for c in clients]
        agg = {k: sum(w * d[k] for w, d in zip(weights, deltas)) for k in state}
        state = {k: state[k] + agg[k] for k in state}
        drift = agg
    return state


def metrics(beta_hat, truth):
    pred, tgt = beta_hat.ravel(), truth.ravel()
    diff = pred - tgt
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "pearson": float(np.corrcoef(pred, tgt)[0, 1]),
        "support_recall": float(np.mean(np.abs(pred[np.abs(tgt) > 1e-8]) > 0.10)),
    }


CFG = {
    "Centralized": {"adaptive": True, "prox": True, "drift": True},
    "AP-FedAdam": {"adaptive": True, "prox": True, "drift": True},
    "FedAvg": {"adaptive": False, "prox": False, "drift": False},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(args.repeats):
        rng = np.random.default_rng(SEED + rep)
        clients, truth = generate_clients(rng)
        pooled = pooled_client(clients)
        for method in ("Centralized", "AP-FedAdam", "FedAvg"):
            data = pooled if method == "Centralized" else clients
            state = run(data, CFG[method])
            mt = metrics(state["beta"], truth)
            rows.append({"repeat": rep, "method": method, **mt})
            print(f"rep={rep} {method:12s} MAE={mt['mae']:.4f} r={mt['pearson']:.3f} recall={mt['support_recall']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "centralized_baseline_repeats.csv", index=False)
    summ = df.groupby("method").agg(["mean", "std"]).reset_index()
    tidy = df.groupby("method")[["mae", "pearson", "support_recall"]].agg(["mean", "std"])
    tidy.to_csv(OUT_DIR / "centralized_baseline_summary.csv")
    print("\n=== SUMMARY (mean +/- sd over", args.repeats, "repeats) ===")
    for method in ("Centralized", "AP-FedAdam", "FedAvg"):
        sub = df[df.method == method]
        print(f"{method:12s} MAE={sub.mae.mean():.4f}+/-{sub.mae.std():.4f}  "
              f"r={sub.pearson.mean():.3f}+/-{sub.pearson.std():.3f}  "
              f"recall={sub.support_recall.mean():.3f}+/-{sub.support_recall.std():.3f}")
    (OUT_DIR / "centralized_baseline_manifest.json").write_text(json.dumps({
        "clients": CLIENTS, "spots_per_client": SPOTS, "genes": GENES, "cell_types": CELL_TYPES,
        "rounds": ROUNDS, "local_steps": LOCAL_STEPS, "lr": LR, "l1_lambda": L1,
        "drift_scale": DRIFT_SCALE, "psi_fixed": PSI_TRUE, "seed": SEED, "repeats": args.repeats,
        "note": "Centralized = AP-FedAdam optimizer on a single pooled client; dispersion fixed at truth.",
    }, indent=2))


if __name__ == "__main__":
    main()
