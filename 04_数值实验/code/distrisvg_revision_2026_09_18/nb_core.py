"""Canonical NB mean regression for the versioned depth-sensitivity experiment.

This module does not modify the archived inverse-mean PyTorch implementation.
The stable log-likelihood uses mu=exp(X @ coefficients + offset), without
clipping the predictor. Dispersion is fixed, as in the compact benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.special import expit, gammaln


@dataclass
class Site:
    x: np.ndarray
    y: np.ndarray
    offset: np.ndarray
    available: np.ndarray


def log_probability(y, eta, psi):
    log_odds = eta - np.log(psi)
    return (gammaln(y + psi) - gammaln(psi) - gammaln(y + 1)
            + y * log_odds - (y + psi) * np.logaddexp(0, log_odds))


def loss_gradient(coef, site, psi):
    eta = site.x @ coef + site.offset[:, None]
    score = (site.y + psi) * expit(eta - np.log(psi)) - site.y
    loss = -log_probability(site.y, eta, psi).mean()
    gradient = site.x.T @ score / site.y.size
    return float(loss), gradient


def fit(sites, k, method="AP-FedAdam", rounds=150, steps=5, lr=0.035,
        l1=2e-6, psi=12.0, drift_scale=0.05, equal_sites=True):
    started = time.perf_counter()
    g = sites[0].y.shape[1]
    d = sites[0].x.shape[1]
    if any(s.x.shape[1] != d or s.y.shape[1] != g for s in sites):
        raise ValueError("All sites must use the same coefficient/feature order")
    weights = np.ones(len(sites)) if equal_sites else np.array([len(s.y) for s in sites])
    weights /= weights.sum()
    mu0 = sum(w * s.y.mean(0) for w, s in zip(weights, sites))
    exposure0 = sum(w * np.exp(s.offset).mean() for w, s in zip(weights, sites))
    coef = np.zeros((d, g))
    coef[:k] = np.log(mu0 / exposure0 + 0.2)
    previous = np.zeros_like(coef)
    adaptive = method != "FedAvg"
    history = []
    for rd in range(rounds):
        numerator = np.zeros_like(coef)
        denominator = np.zeros(d)
        for weight, site in zip(weights, sites):
            local = coef.copy()
            first = np.zeros_like(coef)
            second = np.zeros_like(coef)
            for step in range(steps):
                eta = site.x @ local + site.offset[:, None]
                residual = (site.y + psi) * expit(eta - np.log(psi)) - site.y
                gradient = site.x.T @ residual / site.y.size
                gradient *= site.available[:, None]
                learning_rate = lr * 0.98 ** step
                if adaptive:
                    first = 0.9 * first + 0.1 * gradient
                    second = 0.99 * second + 0.01 * gradient ** 2
                    metric = np.sqrt(second / (1 - 0.99 ** (step + 1))) + 1e-8
                    direction = first / (1 - 0.9 ** (step + 1)) / metric
                    proposal = local - learning_rate * (direction + drift_scale * previous)
                    threshold = learning_rate * l1 / metric[k:]
                    proposal[k:] = np.sign(proposal[k:]) * np.maximum(np.abs(proposal[k:]) - threshold, 0)
                else:
                    proposal = local - learning_rate * gradient
                local = np.where(site.available[:, None], proposal, coef)
            numerator += weight * (local - coef) * site.available[:, None]
            denominator += weight * site.available
        previous = np.divide(numerator, denominator[:, None],
                             out=np.zeros_like(coef), where=denominator[:, None] > 0)
        coef += previous
        if not np.isfinite(coef).all():
            raise FloatingPointError(f"Non-finite fit at round {rd + 1}")
        if (rd + 1) % 25 == 0 or rd == 0:
            loss = sum(w * loss_gradient(coef, s, psi)[0] for w, s in zip(weights, sites))
            history.append({"round": rd + 1, "nb_nll": loss,
                            "update_norm": float(np.linalg.norm(previous))})
    return coef, history, time.perf_counter() - started


def diagonal_score_p(coef, sites, k, psi=12.0):
    """Archived-style diagnostic; NOT a nuisance-adjusted standard error."""
    from scipy.special import ndtr
    information = np.zeros_like(coef)
    for site in sites:
        eta = site.x @ coef + site.offset[:, None]
        score = site.y - (site.y + psi) * expit(eta - np.log(psi))
        information += (site.x ** 2).T @ (score ** 2)
    z = coef[k:] * np.sqrt(information[k:])
    return np.clip(2 * ndtr(-np.abs(z)), 1e-15, 1 - 1e-15).reshape(-1, k, coef.shape[1])
