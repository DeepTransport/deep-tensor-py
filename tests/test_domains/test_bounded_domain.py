import unittest

import torch

import deep_tensor as dt 


torch.manual_seed(0)


class TestLinearDomain(unittest.TestCase):

    def setup_domain(self):
        bounds = torch.tensor([-2.0, 4.0])
        domain = dt.BoundedDomain(bounds=bounds)
        return domain

    def test_linear_domain(self):

        domain = self.setup_domain()

        bounds_true = torch.tensor([-2.0, 4.0])

        self.assertTrue((domain.bounds - bounds_true).abs().max() < 1e-8)
        self.assertAlmostEqual(domain.dxdl, 3.)
        self.assertAlmostEqual(domain.mean, 1.)
        self.assertAlmostEqual(domain.left, -2.)
        self.assertAlmostEqual(domain.right, 4.)
        return
    
    def test_approx2local(self):

        domain = self.setup_domain()

        xs = torch.tensor([-2., -1., 0., 1., 2., 3., 4.])
        ls, dldxs = domain.approx2local(xs)
        
        ls_true = torch.tensor([-1., -2./3., -1./3., 0., 1./3., 2./3., 1.])
        dldxs_true = torch.full(ls_true.shape, 1./3.)

        self.assertTrue((ls - ls_true).abs().max() < 1e-8)
        self.assertTrue((dldxs - dldxs_true).abs().max() < 1e-8)
        return 
    
    def test_local2approx(self):

        domain = self.setup_domain()

        ls = torch.tensor([-1., -0.5, 0., 0.5, 1.])
        xs, dxdls = domain.local2approx(ls)
        
        xs_true = torch.tensor([-2., -0.5, 1., 2.5, 4.])
        dxdls_true = torch.full(xs_true.shape, 3.)

        self.assertTrue((xs - xs_true).abs().max() < 1e-8)
        self.assertTrue((dxdls - dxdls_true).abs().max() < 1e-8)
        return 