import unittest

import torch

import deep_tensor as dt


torch.manual_seed(0)


class TestLegendre(unittest.TestCase):
    
    def test_properties(self):
        """Confirms that some simple properties of Legendre polynomials
        are initialised correctly.

        TODO: check the Golub-Welsch computation of the nodes and weights.

        """

        poly = dt.Legendre(order=3)

        a_true = torch.tensor([1., 3./2., 5./3., 7./4.])
        b_true = torch.tensor([0., 0., 0., 0.])
        c_true = torch.tensor([0., 1./2., 2./3., 3./4.])
        norm_true = torch.tensor([1., 3.**0.5, 5.**0.5, 7.**0.5])

        self.assertEqual(poly.order, 3)
        self.assertTrue((poly.a - a_true).abs().max() < 1e-8)
        self.assertTrue((poly.b - b_true).abs().max() < 1e-8)
        self.assertTrue((poly.c - c_true).abs().max() < 1e-8)
        self.assertTrue((poly.norm - norm_true).abs().max() < 1e-8)

    def test_eval_basis(self):

        poly = dt.Legendre(order=3)

        ls = torch.tensor([-1., -0.5, 0., 0.5, 1.])
        norm_true = torch.tensor([1., 3.**0.5, 5.**0.5, 7.**0.5])

        ps = poly.eval_basis(ls)
        ps_true = torch.tensor([[1., -1., 1., -1.],
                                [1., -1./2., -1./8., 7./16.],
                                [1., 0., -1./2., 0.],
                                [1., 1./2., -1./8., -7./16.],
                                [1., 1., 1., 1.]]) * norm_true

        dpdxs = poly.eval_basis_deriv(ls)
        dpdxs_true = torch.tensor([[0., 1., -3., 6.],
                                   [0., 1., -3./2., 9./24.],
                                   [0., 1., 0., -3./2.],
                                   [0., 1., 3./2., 9./24.],
                                   [0., 1., 3., 6.]]) * norm_true

        self.assertTrue((ps - ps_true).abs().max() < 1e-8)
        self.assertTrue((dpdxs - dpdxs_true).abs().max() < 1e-8)
        return


if __name__ == "__main__":
    unittest.main()