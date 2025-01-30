import unittest

import torch

import deep_tensor as dt 


torch.manual_seed(0)


class TestLinearDomain(unittest.TestCase):

    def test_linear_domain(self):

        bounds = torch.tensor([-2.0, 4.0])
        domain = dt.BoundedDomain(bounds=bounds)

        self.assertTrue((bounds-domain.bounds).abs().max() < 1e-8)
        self.assertAlmostEqual(domain.dxdl, 3.)
        self.assertAlmostEqual(domain.mean, 1.)
        self.assertAlmostEqual(domain.left, -2.)
        self.assertAlmostEqual(domain.right, 4.)

        xs = torch.tensor([-2., -1., 0., 1., 2., 3., 4.])
        ls, dldxs = domain.approx2local(xs)
        
        ls_true = torch.tensor([-1., -2./3., -1./3., 0., 1./3., 2./3., 1.])
        dldxs_true = torch.full(ls_true.shape, 1./3.)

        self.assertTrue((ls - ls_true).abs().max() < 1e-8)
        self.assertTrue((dldxs - dldxs_true).abs().max() < 1e-8)

        return