"""Data-local NB2 refit and full nuisance-adjusted covariance.

This is a finite-dimensional, all-variable refit, not a post-selection shortcut.
Sites exchange summaries at common parameters. The local Python emulator does
not implement transport security or distributed preprocessing. HC1 assumes
independent spots and does not address deconvolution or spatial dependence.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln, polygamma
from scipy.stats import chi2


@dataclass
class LocalGene:
    x: np.ndarray
    y: np.ndarray
    offset: np.ndarray
    weight: float = 1.0


def local_summary(parameter, site, estimate_dispersion=True, fixed_psi=12., second=False):
    d = site.x.shape[1]
    beta = parameter[:d]
    psi = np.exp(parameter[d]) if estimate_dispersion else fixed_psi
    eta = site.x @ beta + site.offset
    odds = eta - np.log(psi)
    p = expit(odds)
    q = expit(-odds)
    lp = (gammaln(site.y + psi) - gammaln(psi) - gammaln(site.y + 1)
          + site.y * odds - (site.y + psi) * np.logaddexp(0, odds))
    g_eta = (site.y + psi) * p - site.y
    scores = site.x * g_eta[:, None]
    if estimate_dispersion:
        s = (digamma(site.y + psi) - digamma(psi) + 1
             - np.logaddexp(0, odds) - (1 + site.y / psi) * q)
        g_logpsi = -psi * s
        scores = np.column_stack([scores, g_logpsi])
    loss = -site.weight * lp.sum()
    gradient = site.weight * scores.sum(0)
    if not second:
        return loss, gradient
    h_eta = (site.y + psi) * p * q
    hessian = np.zeros((len(parameter), len(parameter)))
    hessian[:d, :d] = site.x.T @ (h_eta[:, None] * site.x)
    if estimate_dispersion:
        cross = site.x.T @ (p * g_eta)
        hessian[:d, d] = cross
        hessian[d, :d] = cross
        # Differentiating w.r.t. log(psi) retains nuisance coupling.
        s_prime = (polygamma(1, site.y + psi) - polygamma(1, psi)
                   + p / psi + (site.y * q * q / psi - p * q) / psi)
        hessian[d, d] = np.sum(g_logpsi - psi * psi * s_prime)
    return loss, gradient, site.weight * hessian, site.weight ** 2 * (scores.T @ scores)


def aggregate(parameter, sites, estimate_dispersion=True, fixed_psi=12., second=False):
    local = [local_summary(parameter, site, estimate_dispersion, fixed_psi, second)
             for site in sites]
    return tuple(sum(part[i] for part in local) for i in range(len(local[0])))


def covariance(parameter, sites, estimate_dispersion=True, fixed_psi=12.):
    _, gradient, bread, meat = aggregate(parameter, sites, estimate_dispersion, fixed_psi, True)
    n = sum(len(site.y) for site in sites)
    scale = np.sqrt(np.maximum(np.diag(bread), 1e-30))
    standardized = bread / scale[:, None] / scale[None, :]
    eigenvalues = np.linalg.eigvalsh(standardized)
    if np.any(np.diag(bread) <= 0) or eigenvalues.min() < 1e-9:
        raise np.linalg.LinAlgError("Nonpositive or unidentifiable joint information")
    inverse = np.linalg.inv(standardized) / scale[:, None] / scale[None, :]
    robust = inverse @ meat @ inverse * n / (n - len(parameter))
    return inverse, robust, float(eigenvalues.max() / eigenvalues.min()), gradient


def block_pvalues(parameter, covariance_matrix, k):
    out = np.ones(k)
    for ct in range(k):
        indices = [k + ct, 2 * k + ct]
        beta = parameter[indices]
        v = covariance_matrix[np.ix_(indices, indices)]
        if np.linalg.eigvalsh(v).min() <= 0:
            raise np.linalg.LinAlgError("Nonpositive block covariance")
        statistic = float(beta @ np.linalg.solve(v, beta))
        out[ct] = chi2.sf(max(0., statistic), 2)
    return np.maximum(out, 1e-300)


def refit_gene(sites, k, estimate_dispersion=True, fixed_psi=12., initial=None):
    started = time.perf_counter()
    n = sum(len(site.y) for site in sites)
    d = sites[0].x.shape[1]
    length = d + int(estimate_dispersion)
    if d != 3 * k or n <= length + 2:
        raise ValueError("Incompatible or undersized joint-refit design")
    n_weighted = sum(site.weight * len(site.y) for site in sites)
    if initial is None:
        parameter = np.zeros(length)
        mean = sum(site.weight * site.y.sum() for site in sites) / n_weighted
        exposure = sum(site.weight * np.exp(site.offset).sum() for site in sites) / n_weighted
        parameter[:k] = np.log(mean / exposure + .1)
        if estimate_dispersion:
            parameter[-1] = np.log(fixed_psi)
    else:
        parameter = np.array(initial, dtype=float, copy=True)
    default = dict(parameter=parameter, p_model=np.ones(k), p_sandwich=np.ones(k),
                   converged=False, valid=False, psi=fixed_psi, iterations=0,
                   gradient_max=np.nan, condition_number=np.nan, n_spots=n,
                   covariance_status="not_fitted", psi_boundary=False)
    if sum(site.y.sum() for site in sites) == 0:
        return dict(default, reason="all_zero", seconds=time.perf_counter() - started)

    def objective(value):
        loss, gradient = aggregate(value, sites, estimate_dispersion, fixed_psi)
        return loss / n_weighted, gradient / n_weighted

    bounds = [(None, None)] * d + ([(-6., 8.)] if estimate_dispersion else [])
    result = minimize(objective, parameter, jac=True, method="L-BFGS-B", bounds=bounds,
                      options=dict(maxiter=500, ftol=1e-12, gtol=2e-7, maxls=30, maxcor=30))
    parameter = result.x
    loss, gradient = objective(parameter)
    boundary = estimate_dispersion and (parameter[-1] <= -5.999 or parameter[-1] >= 7.999)
    projected_gradient = gradient.copy()
    if boundary and parameter[-1] >= 7.999 and gradient[-1] < 0:
        projected_gradient[-1] = 0.
    converged = np.isfinite(parameter).all() and np.max(np.abs(projected_gradient)) < 2e-5
    out = dict(default, parameter=parameter, converged=bool(converged),
               psi=float(np.exp(parameter[-1])) if estimate_dispersion else fixed_psi,
               iterations=int(result.nit), gradient_max=float(np.max(np.abs(projected_gradient))),
               psi_boundary=bool(boundary), mean_nll=float(loss), optimizer_message=str(result.message))
    if not converged:
        out["reason"] = "nonstationary_fit"
    elif boundary:
        # Do not use an interior-Wald theorem for a dispersion-boundary fit.
        out["reason"] = "dispersion_boundary"
    else:
        try:
            model, robust, condition, _ = covariance(parameter, sites, estimate_dispersion, fixed_psi)
            out.update(p_model=block_pvalues(parameter, model, k),
                       p_sandwich=block_pvalues(parameter, robust, k), valid=True,
                       condition_number=condition, covariance_status="full_joint", reason="ok")
        except np.linalg.LinAlgError as error:
            out["reason"] = str(error)
    out["seconds"] = time.perf_counter() - started
    return out


def gene_sites(model_sites, gene, equal_sites=True):
    n = sum(len(site.y) for site in model_sites)
    m = len(model_sites)
    return [LocalGene(site.x, site.y[:, gene], site.offset,
                      n / (m * len(site.y)) if equal_sites else 1.) for site in model_sites]
