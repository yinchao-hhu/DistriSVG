import unittest
import numpy as np
from numpy.testing import assert_allclose
from run_pooled import pool, load_inputs, original
from joint_refit import LocalGene, aggregate, local_summary, refit_gene
from test_joint_refit import fixture


class PooledReplayTests(unittest.TestCase):
    def test_row_order_and_unit_weights(self):
        site,parameter=fixture()
        parts=[LocalGene(site.x[a:b],site.y[a:b],site.offset[a:b])
               for a,b in [(0,120),(120,240),(240,360)]]
        joined=pool(parts)
        assert_allclose(joined.x,site.x)
        assert_allclose(joined.y,site.y)
        for a,b in zip(aggregate(parameter,parts,second=True),local_summary(parameter,joined,second=True)):
            assert_allclose(a,b,rtol=1e-11,atol=1e-9)
        parts[0].weight=2
        with self.assertRaises(ValueError):pool(parts)

    def test_estimated_dispersion_replay(self):
        site,_=fixture()
        parts=[LocalGene(site.x[a:b],site.y[a:b],site.offset[a:b])
               for a,b in [(0,120),(120,240),(240,360)]]
        a=refit_gene(parts,3)
        b=refit_gene([pool(parts)],3)
        self.assertTrue(a['valid'] and b['valid'])
        assert_allclose(a['parameter'],b['parameter'],atol=3e-4)
        assert_allclose(a['p_model'],b['p_model'],atol=1e-4)

    def test_frozen_truth_and_dimensions(self):
        clients,types,active,_=load_inputs(0)
        self.assertEqual(len(types),5)
        self.assertEqual(active.sum(),20)
        self.assertEqual([x['y'].shape for x in clients],[(100,40)]*3)
        for family in original.old.FAMILIES:
            sites,_=original.old.make_sites(clients,family,'none')
            self.assertTrue(all(s.x.shape==(100,15) for s in sites))


if __name__=='__main__':unittest.main(verbosity=2)
