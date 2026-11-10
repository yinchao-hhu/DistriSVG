"""Weight-consistent information covariance for correctly specified NB refits."""
import numpy as np
from joint_refit import local_summary, covariance, block_pvalues


def weighted_covariance(parameter, sites, estimate_dispersion=True, fixed_psi=12.):
    inverse, _, condition, _ = covariance(parameter, sites, estimate_dispersion, fixed_psi)
    # E[-Hessian_i] = E[score_i score_i^T] under the sampling model, before weights.
    information_meat = sum(site.weight * local_summary(
        parameter, site, estimate_dispersion, fixed_psi, second=True)[2] for site in sites)
    result = inverse @ information_meat @ inverse
    result = (result + result.T) / 2
    if np.linalg.eigvalsh(result).min() <= 0:
        raise np.linalg.LinAlgError("Nonpositive weighted information covariance")
    return result, condition


def weighted_gene(parameter, sites, k, valid):
    if not valid:
        return np.ones(k), "invalid_original_refit"
    try:
        matrix, _ = weighted_covariance(parameter, sites)
        return block_pvalues(parameter, matrix, k), "ok"
    except np.linalg.LinAlgError as error:
        return np.ones(k), str(error)
