import unittest

import torch

import deep_tensor as dt
from deep_tensor.constants import EPS


torch.manual_seed(0)


class TestSpectralPolynomials(unittest.TestCase):

    def test_orthogonality(self):
        """Confirms that the product of the basis2node and node2basis 
        matrices is the identity matrix (see Cui and Dolgov 2022, 
        Appendix A).
        """

        polynomials: list[dt.Spectral] = [
            dt.Chebyshev1st(order=20),
            # dt.Chebyshev2nd(order=20),
            dt.Chebyshev2ndUnweighted(order=20),
            dt.Hermite(order=20),
            # dt.Laguerre(order=20),
            dt.Fourier(order=20),
            dt.Legendre(order=20)
        ]

        for poly in polynomials:
            with self.subTest(poly=poly):
                Id = poly.basis2node @ poly.node2basis 
                Id_true = torch.eye(poly.cardinality)
                self.assertTrue((Id_true - Id).abs().max() < 1e-8)

        return
    
    def test_legendre(self):
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

        ls = torch.tensor([-1., -0.5, 0., 0.5, 1.])

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
    
    def test_chebyshev_1st(self):
        """Confirms that some simple properties of Chebyshev 
        polynomials of the first degree are initialised correctly.
        """

        poly = dt.Chebyshev1st(order=5)

        nodes_true = torch.tensor([
            torch.tensor(11.0*torch.pi/12.0).cos(),
            torch.tensor(9.0*torch.pi/12.0).cos(),
            torch.tensor(7.0*torch.pi/12.0).cos(),
            torch.tensor(5.0*torch.pi/12.0).cos(),
            torch.tensor(3.0*torch.pi/12.0).cos(),
            torch.tensor(1.0*torch.pi/12.0).cos()
        ])
        weights_true = torch.full((6,), 1.0/6.0)
        basis2node_true = torch.tensor([
            [
                1.0, 
                torch.tensor(11.0*torch.pi/12.0).cos(), 
                torch.tensor(22.0*torch.pi/12.0).cos(), 
                torch.tensor(33.0*torch.pi/12.0).cos(), 
                torch.tensor(44.0*torch.pi/12.0).cos(), 
                torch.tensor(55.0*torch.pi/12.0).cos()
            ],
            [
                1.0, 
                torch.tensor(9.0*torch.pi/12.0).cos(), 
                torch.tensor(18.0*torch.pi/12.0).cos(), 
                torch.tensor(27.0*torch.pi/12.0).cos(), 
                torch.tensor(36.0*torch.pi/12.0).cos(), 
                torch.tensor(45.0*torch.pi/12.0).cos()
            ],
            [
                1.0, 
                torch.tensor(7.0*torch.pi/12.0).cos(), 
                torch.tensor(14.0*torch.pi/12.0).cos(), 
                torch.tensor(21.0*torch.pi/12.0).cos(), 
                torch.tensor(28.0*torch.pi/12.0).cos(), 
                torch.tensor(35.0*torch.pi/12.0).cos()
            ],
            [
                1.0, 
                torch.tensor(5.0*torch.pi/12.0).cos(), 
                torch.tensor(10.0*torch.pi/12.0).cos(), 
                torch.tensor(15.0*torch.pi/12.0).cos(), 
                torch.tensor(20.0*torch.pi/12.0).cos(), 
                torch.tensor(25.0*torch.pi/12.0).cos()
            ],
            [
                1.0, 
                torch.tensor(3.0*torch.pi/12.0).cos(), 
                torch.tensor(6.0*torch.pi/12.0).cos(), 
                torch.tensor(9.0*torch.pi/12.0).cos(), 
                torch.tensor(12.0*torch.pi/12.0).cos(), 
                torch.tensor(15.0*torch.pi/12.0).cos()
            ],
            [
                1.0, 
                torch.tensor(1.0*torch.pi/12.0).cos(), 
                torch.tensor(2.0*torch.pi/12.0).cos(), 
                torch.tensor(3.0*torch.pi/12.0).cos(), 
                torch.tensor(4.0*torch.pi/12.0).cos(), 
                torch.tensor(5.0*torch.pi/12.0).cos()
            ],
        ]) * poly.norm

        self.assertEqual(poly.order, 5)
        self.assertTrue((poly.nodes - nodes_true).abs().max() < 1e-8)
        self.assertTrue((poly.weights - weights_true).abs().max() < 1e-8)
        self.assertTrue((poly.basis2node - basis2node_true).abs().max() < 1e-8)

        ls = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        ws = poly.eval_measure(ls)

        ws_true = torch.tensor([
            torch.tensor(1.0/EPS).sqrt(),
            torch.tensor(4.0/3.0).sqrt(), 
            torch.tensor(1.0), 
            torch.tensor(4.0/3.0).sqrt(), 
            torch.tensor(1.0/EPS).sqrt(),
        ]) / torch.pi

        self.assertTrue((ws - ws_true).abs().max() < 1e-8)
        return


if __name__ == "__main__":
    unittest.main()