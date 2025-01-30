import unittest

import torch

import deep_tensor as dt 


torch.manual_seed(0)


class TestApproxBases(unittest.TestCase):

    def setup_bases(self):

        polys = [
            dt.Lagrange1(num_elems=3),
            dt.Lagrange1(num_elems=2),
            dt.Lagrange1(num_elems=4)
        ]

        domains = [
            dt.BoundedDomain(bounds=torch.tensor([-2., 1.])),
            dt.BoundedDomain(bounds=torch.tensor([-1., 1.])),
            dt.BoundedDomain(bounds=torch.tensor([-2., 2.]))
        ]

        bases = dt.ApproxBases(polys, domains, dim=3)

        return bases

    def test_approx2local(self):
        
        bases = self.setup_bases()

        xs = torch.tensor([[-2., -1., -2.],
                           [-1.,  0.,  1.],
                           [ 1.,  1.,  2.]])
        
        indices = torch.arange(3)
        
        ls_true = torch.tensor([[-1., -1., -1.],
                                [-1./3., 0., 0.5],
                                [ 1., 1., 1.]])
        dxdls_true = torch.tensor([[2./3., 1., 0.5],
                                   [2./3., 1., 0.5],
                                   [2./3., 1., 0.5]])

        # Test with all indices
        ls, dldxs = bases.approx2local(xs, indices)
        self.assertTrue((ls-ls_true).abs().max() < 1e-8)
        self.assertTrue((dldxs-dxdls_true).abs().max() < 1e-8)

        # Test with a subset of the indices
        ls, dldxs = bases.approx2local(xs[:, 1:], indices[1:])
        self.assertTrue((ls-ls_true[:, 1:]).abs().max() < 1e-8)
        self.assertTrue((dldxs-dxdls_true[:, 1:]).abs().max() < 1e-8)

        # Test with a subset of the indices
        ls, dldxs = bases.approx2local(xs[:, 2:], indices[2:])
        self.assertTrue((ls-ls_true[:, 2:]).abs().max() < 1e-8)
        self.assertTrue((dldxs-dxdls_true[:, 2:]).abs().max() < 1e-8)

        # Test with a subset of the indices
        ls, dldxs = bases.approx2local(xs[:, [0, 2]], indices[[0, 2]])
        self.assertTrue((ls-ls_true[:, [0, 2]]).abs().max() < 1e-8)
        self.assertTrue((dldxs-dxdls_true[:, [0, 2]]).abs().max() < 1e-8)

        bases.approx2local_log_density(xs, indices)