import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import nbinom

from joint_refit import LocalGene, aggregate, covariance, local_summary, refit_gene


def fixture(seed=825, n=360):
    rng = np.random.default_rng(seed)
    omega = rng.dirichlet([1, 1, 1], size=n)
    h = rng.normal(size=(n, 2))
    x = np.column_stack([omega, omega * h[:, :1], omega * h[:, 1:]])
    parameter = np.array([2., 2.2, 1.7, .3, 0, -.4, 0, .25, .2, np.log(4.)])
    offset = rng.normal(0, .2, n)
    mu = np.exp(x @ parameter[:-1] + offset)
    y = rng.negative_binomial(4., 4. / (4. + mu))
    return LocalGene(x, y, offset), parameter


class JointRefitTests(unittest.TestCase):
    def test_likelihood_matches_scipy(self):
        site, parameter = fixture()
        loss, _ = local_summary(parameter, site)
        mu = np.exp(site.x @ parameter[:-1] + site.offset)
        assert_allclose(loss, -nbinom.logpmf(site.y, 4., 4. / (4. + mu)).sum(), rtol=1e-12)

    def test_full_gradient_and_hessian(self):
        site, parameter = fixture()
        loss, gradient, hessian, _ = local_summary(parameter, site, second=True)
        epsilon = 1e-5
        eye = np.eye(len(parameter)) * epsilon
        numerical_g = np.array([(local_summary(parameter + step, site)[0]
                                 - local_summary(parameter - step, site)[0]) / (2 * epsilon)
                                for step in eye])
        numerical_h = np.column_stack([(local_summary(parameter + step, site)[1]
                                       - local_summary(parameter - step, site)[1]) / (2 * epsilon)
                                      for step in eye])
        assert_allclose(gradient, numerical_g, atol=2e-6, rtol=2e-6)
        assert_allclose(hessian, numerical_h, atol=2e-6, rtol=2e-6)

    def test_weighted_site_summary_identity(self):
        site, parameter = fixture()
        sites = [LocalGene(site.x[:120], site.y[:120], site.offset[:120]),
                 LocalGene(site.x[120:], site.y[120:], site.offset[120:])]
        for a, b in zip(aggregate(parameter, sites, second=True),
                        local_summary(parameter, site, second=True)):
            assert_allclose(a, b, rtol=1e-12, atol=1e-10)
        weighted = LocalGene(site.x, site.y, site.offset, 2.)
        plain = local_summary(parameter, site, second=True)
        other = local_summary(parameter, weighted, second=True)
        for index in range(3):
            assert_allclose(other[index], 2 * plain[index])
        assert_allclose(other[3], 4 * plain[3])

    def test_matches_statsmodels_independent_implementation(self):
        from statsmodels.discrete.discrete_model import NegativeBinomial
        site, parameter = fixture()
        result = refit_gene([site], 3)
        self.assertTrue(result["valid"], result.get("reason"))
        reference = NegativeBinomial(site.y, site.x, offset=site.offset).fit(
            start_params=np.r_[result["parameter"][:-1], 1 / result["psi"]],
            method="newton", maxiter=100, disp=False, tol=1e-8)
        self.assertTrue(reference.mle_retvals["converged"])
        assert_allclose(result["parameter"][:-1], reference.params[:-1], atol=2e-4)
        assert_allclose(result["psi"], 1 / reference.params[-1], rtol=1e-4)
        information, _, _, _ = covariance(result["parameter"], [site])
        assert_allclose(information[:-1, :-1], reference.cov_params()[:-1, :-1], rtol=5e-4, atol=2e-5)

    def test_initialization_not_selection(self):
        site, parameter = fixture()
        a = refit_gene([site], 3)
        b = refit_gene([site], 3, initial=parameter)
        assert_allclose(a["parameter"], b["parameter"], atol=3e-4)
        assert_allclose(a["p_sandwich"], b["p_sandwich"], atol=1e-4)

    def test_known_dispersion_and_local_pooled_equivalence(self):
        site, _ = fixture()
        sites = [LocalGene(site.x[:180], site.y[:180], site.offset[:180]),
                 LocalGene(site.x[180:], site.y[180:], site.offset[180:])]
        a = refit_gene([site], 3, estimate_dispersion=False, fixed_psi=4.)
        b = refit_gene(sites, 3, estimate_dispersion=False, fixed_psi=4.)
        self.assertTrue(a["valid"] and b["valid"])
        assert_allclose(a["parameter"], b["parameter"], atol=1e-6)
        assert_allclose(a["p_sandwich"], b["p_sandwich"], atol=1e-6)

    def test_zero_and_unidentifiable_are_flagged(self):
        site, _ = fixture()
        zero = LocalGene(site.x, np.zeros_like(site.y), site.offset)
        fit = refit_gene([zero], 3)
        self.assertFalse(fit["valid"])
        self.assertEqual(fit["reason"], "all_zero")
        assert_allclose(fit["p_sandwich"], 1.)
        site.x[:, 1] = site.x[:, 0]
        fit = refit_gene([site], 3, estimate_dispersion=False)
        self.assertFalse(fit["valid"])
        assert_allclose(fit["p_sandwich"], 1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
