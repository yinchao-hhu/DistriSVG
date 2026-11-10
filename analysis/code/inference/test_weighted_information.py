import unittest
import numpy as np
from joint_refit import LocalGene, aggregate, covariance
from weighted_information import weighted_covariance


class WeightedInformationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        x = np.column_stack([np.ones(180), rng.normal(size=(180,2))])
        theta = np.array([1.,.1,-.2])
        mu = np.exp(x @ theta)
        y = rng.negative_binomial(12,12/(12+mu)).astype(float)
        self.parameter = theta
        self.sites = [LocalGene(x[:60],y[:60],np.zeros(60)),LocalGene(x[60:],y[60:],np.zeros(120))]

    def test_unit_weights_reduce_to_inverse_joint_information(self):
        inverse = covariance(self.parameter,self.sites,False)[0]
        weighted = weighted_covariance(self.parameter,self.sites,False)[0]
        np.testing.assert_allclose(weighted,inverse,rtol=1e-12,atol=1e-14)

    def test_common_rescaling_leaves_covariance_unchanged(self):
        self.sites[0].weight=1.5; self.sites[1].weight=.75
        first = weighted_covariance(self.parameter,self.sites,False)[0]
        for site in self.sites: site.weight *= 7
        second = weighted_covariance(self.parameter,self.sites,False)[0]
        np.testing.assert_allclose(first,second,rtol=1e-12,atol=1e-14)

    def test_manual_weighted_expected_information(self):
        self.sites[0].weight=1.5; self.sites[1].weight=.75
        bread = aggregate(self.parameter,self.sites,False,12,True)[2]
        meat = sum(site.weight**2 * (site.x.T @ (((site.y+12)*
            (np.exp(site.x@self.parameter)/(12+np.exp(site.x@self.parameter))) *
            (12/(12+np.exp(site.x@self.parameter))))[:,None]*site.x)) for site in self.sites)
        expected = np.linalg.solve(bread,meat) @ np.linalg.inv(bread)
        actual = weighted_covariance(self.parameter,self.sites,False)[0]
        np.testing.assert_allclose(expected,actual,rtol=1e-12,atol=1e-14)


if __name__ == "__main__":
    unittest.main()
