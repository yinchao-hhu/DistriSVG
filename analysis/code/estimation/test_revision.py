import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import nbinom

from metrics import bh, pair_metrics, acat
from nb_core import Site, log_probability, loss_gradient, fit


class MetricsTests(unittest.TestCase):
    def test_confusion_counts(self):
        out = pair_metrics([1, 1, 0, 0], [0.001, 0.5, 0.001, 0.9])
        self.assertEqual([out[x] for x in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(out["f1"], 0.5)

    def test_no_discoveries(self):
        out = pair_metrics([0, 1], [0.8, 0.8])
        self.assertTrue(np.isnan(out["precision"]))
        self.assertEqual(out["fdp"], 0)

    def test_all_null(self):
        out = pair_metrics([0, 0], [0.8, 0.8])
        self.assertTrue(np.isnan(out["power"]))
        self.assertTrue(np.isnan(out["auroc"]))

    def test_ties_and_bh(self):
        out = pair_metrics([0, 1, 0, 1], [0.5] * 4)
        self.assertEqual(out["auroc"], 0.5)
        self.assertEqual(out["auprc"], 0.5)
        assert_allclose(bh([0.01, 0.02, 0.8]), [0.03, 0.03, 0.8])
        assert_allclose(acat([[0.01], [0.01]]), [0.01])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            pair_metrics([0, 1], [np.nan, 0.1])


class NBTests(unittest.TestCase):
    def test_log_probability_mean_variance(self):
        import torch
        for mu in (0.2, 2.0, 50.0):
            for psi in (1.0, 10.0):
                y = np.array([0., 1., 6., 100.])
                assert_allclose(log_probability(y, np.log(mu), psi), nbinom.logpmf(y, psi, psi / (psi + mu)), atol=1e-12)
                dist = torch.distributions.NegativeBinomial(
                    total_count=torch.tensor(psi, dtype=torch.float64),
                    logits=torch.tensor(np.log(mu) - np.log(psi), dtype=torch.float64))
                assert_allclose(float(dist.mean), mu)
                assert_allclose(float(dist.variance), mu + mu ** 2 / psi)
                assert_allclose(dist.log_prob(torch.tensor(y)).numpy(), log_probability(y, np.log(mu), psi), atol=1e-12)

    def test_gradient_with_offset(self):
        import torch
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=(17, 6)), rng.poisson(2, (17, 3))
        site = Site(x, y, rng.normal(size=17), np.ones(6, dtype=bool))
        coef = rng.normal(scale=0.1, size=(6, 3))
        _, grad = loss_gradient(coef, site, 12.)
        tc = torch.tensor(coef, requires_grad=True)
        logits = torch.tensor(x) @ tc + torch.tensor(site.offset)[:, None] - np.log(12.)
        dist = torch.distributions.NegativeBinomial(total_count=12., logits=logits)
        (-dist.log_prob(torch.tensor(y)).mean()).backward()
        assert_allclose(grad, tc.grad.numpy(), rtol=1e-11, atol=1e-12)
        for row, col in [(0, 0), (5, 2)]:
            plus, minus = coef.copy(), coef.copy()
            plus[row, col] += 1e-6
            minus[row, col] -= 1e-6
            diff = (loss_gradient(plus, site, 12.)[0] - loss_gradient(minus, site, 12.)[0]) / 2e-6
            assert_allclose(diff, grad[row, col], rtol=1e-5, atol=1e-8)

    def test_unavailable_block(self):
        site = Site(np.ones((10, 3)), np.ones((10, 2)), np.zeros(10), np.array([True, False, False]))
        coef, _, _ = fit([site], k=1, rounds=3)
        assert_allclose(coef[1:], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
