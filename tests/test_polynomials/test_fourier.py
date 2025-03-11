import unittest

import torch

import deep_tensor as dt


torch.manual_seed(0)


class TestFourier(unittest.TestCase):
    
    def test_properties(self):
        """Checks that the properties of a Fourier basis are 
        initialised correctly.
        """

        poly = dt.Fourier(order=2)

        nodes_true = torch.tensor([-2.0/3.0, -1.0/3.0, 0.0, 1.0/3.0, 2.0/3.0, 1.0])
        weights_true = torch.tensor([1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0])
        basis2node_true = torch.tensor([
            [
                1.0, 
                2 ** 0.5 * torch.tensor(-2.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(-4.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(-2.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(-4.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(-6.0*torch.pi/3.0).cos()
            ],
            [
                1.0, 
                2 ** 0.5 * torch.tensor(-1.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(-2.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(-1.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(-2.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(-3.0*torch.pi/3.0).cos()
            ],
            [
                1.0, 
                2 ** 0.5 * torch.tensor(0.0).sin(),
                2 ** 0.5 * torch.tensor(0.0).sin(),
                2 ** 0.5 * torch.tensor(0.0).cos(),
                2 ** 0.5 * torch.tensor(0.0).cos(),
                2 ** 0.5 * torch.tensor(0.0).cos()
            ],
            [
                1.0, 
                2 ** 0.5 * torch.tensor(1.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(2.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(1.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(2.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(3.0*torch.pi/3.0).cos()
            ],
            [
                1.0, 
                2 ** 0.5 * torch.tensor(2.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(4.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(2.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(4.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(6.0*torch.pi/3.0).cos()
            ],
            [
                1.0, 
                2 ** 0.5 * torch.tensor(1.0*torch.pi).sin(),
                2 ** 0.5 * torch.tensor(2.0*torch.pi).sin(),
                2 ** 0.5 * torch.tensor(1.0*torch.pi).cos(),
                2 ** 0.5 * torch.tensor(2.0*torch.pi).cos(),
                2 ** 0.5 * torch.tensor(3.0*torch.pi).cos()
            ],
        ])
        node2basis_true = 1.0/6.0 * torch.tensor([
            [
                1.0, 
                1.0, 
                1.0, 
                1.0, 
                1.0, 
                1.0
            ],
            [
                2 ** 0.5 * torch.tensor(-2.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(-1.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(0.0).sin(),
                2 ** 0.5 * torch.tensor(1.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(2.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(3.0*torch.pi/3.0).sin()
            ],
            [
                2 ** 0.5 * torch.tensor(-4.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(-2.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(0.0).sin(),
                2 ** 0.5 * torch.tensor(2.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(4.0*torch.pi/3.0).sin(),
                2 ** 0.5 * torch.tensor(6.0*torch.pi/3.0).sin()
            ],
            [
                2 ** 0.5 * torch.tensor(-2.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(-1.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(0.0).cos(),
                2 ** 0.5 * torch.tensor(1.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(2.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(3.0*torch.pi/3.0).cos()
            ],
            [
                2 ** 0.5 * torch.tensor(-4.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(-2.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(0.0).cos(),
                2 ** 0.5 * torch.tensor(2.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(4.0*torch.pi/3.0).cos(),
                2 ** 0.5 * torch.tensor(6.0*torch.pi/3.0).cos()
            ],
            [
                0.5 * 2 ** 0.5 * torch.tensor(-6.0*torch.pi/3.0).cos(),
                0.5 * 2 ** 0.5 * torch.tensor(-3.0*torch.pi/3.0).cos(),
                0.5 * 2 ** 0.5 * torch.tensor(0.0).cos(),
                0.5 * 2 ** 0.5 * torch.tensor(3.0*torch.pi/3.0).cos(),
                0.5 * 2 ** 0.5 * torch.tensor(6.0*torch.pi/3.0).cos(),
                0.5 * 2 ** 0.5 * torch.tensor(9.0*torch.pi/3.0).cos()
            ]
        ])
        omegas_true = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        mass_R_true = torch.eye(6)

        self.assertEqual(poly.order, 2)
        self.assertTrue((poly.nodes - nodes_true).abs().max() < 1e-8)
        self.assertTrue((poly.weights - weights_true).abs().max() < 1e-8)
        self.assertTrue((poly.basis2node - basis2node_true).abs().max() < 1e-8)
        self.assertTrue((poly.node2basis - node2basis_true).abs().max() < 1e-8)
        self.assertTrue((poly.omegas - omegas_true).abs().max() < 1e-8)
        self.assertTrue((poly.mass_R - mass_R_true).abs().max() < 1e-8)
        pass

    def test_eval_basis(self):

        # poly = dt.Legendre(order=3)

        # ls = torch.tensor([-1., -0.5, 0., 0.5, 1.])
        # norm_true = torch.tensor([1., 3.**0.5, 5.**0.5, 7.**0.5])

        # ps = poly.eval_basis(ls)
        # ps_true = torch.tensor([[1., -1., 1., -1.],
        #                         [1., -1./2., -1./8., 7./16.],
        #                         [1., 0., -1./2., 0.],
        #                         [1., 1./2., -1./8., -7./16.],
        #                         [1., 1., 1., 1.]]) * norm_true

        # dpdxs = poly.eval_basis_deriv(ls)
        # dpdxs_true = torch.tensor([[0., 1., -3., 6.],
        #                            [0., 1., -3./2., 9./24.],
        #                            [0., 1., 0., -3./2.],
        #                            [0., 1., 3./2., 9./24.],
        #                            [0., 1., 3., 6.]]) * norm_true

        # self.assertTrue((ps - ps_true).abs().max() < 1e-8)
        # self.assertTrue((dpdxs - dpdxs_true).abs().max() < 1e-8)
        # return
        pass


if __name__ == "__main__":
    unittest.main()