import unittest

import torch

import deep_tensor as dt
from deep_tensor.constants import EPS


torch.manual_seed(0)


class TestChebyshev1st(unittest.TestCase):
    
    def test_properties(self):
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
        norm_true = torch.tensor([
            1.0, 
            torch.tensor(2.0).sqrt(), 
            torch.tensor(2.0).sqrt(), 
            torch.tensor(2.0).sqrt(), 
            torch.tensor(2.0).sqrt(),
            torch.tensor(2.0).sqrt()
        ])

        self.assertEqual(poly.order, 5)
        self.assertTrue((poly.nodes - nodes_true).abs().max() < 1e-8)
        self.assertTrue((poly.weights - weights_true).abs().max() < 1e-8)
        self.assertTrue((poly.basis2node - basis2node_true).abs().max() < 1e-8)
        self.assertTrue((poly.norm - norm_true).abs().max() < 1e-8)
        return
    
    def test_eval_measure(self):
        
        poly = dt.Chebyshev1st(order=5)

        ls = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        ws = poly.eval_measure(ls)
        logws = poly.eval_log_measure(ls)

        ws_true = torch.tensor([
            torch.tensor(1.0/EPS).sqrt(),
            torch.tensor(4.0/3.0).sqrt(), 
            torch.tensor(1.0), 
            torch.tensor(4.0/3.0).sqrt(), 
            torch.tensor(1.0/EPS).sqrt(),
        ]) / torch.pi
        logws_true = torch.log(ws_true)

        self.assertTrue((ws - ws_true).abs().max() < 1e-8)
        self.assertTrue((logws - logws_true).abs().max() < 1e-8)
        return

    def test_eval_measure_deriv(self):

        poly = dt.Chebyshev1st(order=5)

        ls = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        dwdls = poly.eval_measure_deriv(ls)
        logdwdls = poly.eval_log_measure_deriv(ls)

        dwdls_true = torch.tensor([
            -1.0 * torch.tensor(EPS).pow(-3.0/2.0),
            -0.5 * torch.tensor(3.0/4.0).pow(-3.0/2.0), 
             0.0 * torch.tensor(1.0), 
             0.5 * torch.tensor(3.0/4.0).pow(-3.0/2.0), 
             1.0 * torch.tensor(EPS).pow(-3.0/2.0),
        ]) / torch.pi

        logdwdls_true = torch.tensor([-1.0/EPS, -4.0/6.0, 0.0, 4.0/6.0, 1.0/EPS])

        self.assertTrue((dwdls - dwdls_true).abs().max() < 1e-8)
        self.assertTrue((logdwdls - logdwdls_true).abs().max() < 1e-8)
        return
    
    def test_eval_basis_deriv(self):

        poly = dt.Chebyshev1st(order=3)
        ls = torch.tensor([-0.5, 0.0, 0.5])
        thetas = ls.acos()

        dpdls = poly.eval_basis_deriv(ls)

        dpdls_true = torch.tensor([
            [
                0.0,
                1.0 * torch.sin(1.0 * thetas[0]) / torch.sin(thetas[0]),
                2.0 * torch.sin(2.0 * thetas[0]) / torch.sin(thetas[0]),
                3.0 * torch.sin(3.0 * thetas[0]) / torch.sin(thetas[0])
            ],
            [
                0.0,
                1.0 * torch.sin(1.0 * thetas[1]) / torch.sin(thetas[1]),
                2.0 * torch.sin(2.0 * thetas[1]) / torch.sin(thetas[1]),
                3.0 * torch.sin(3.0 * thetas[1]) / torch.sin(thetas[1])
            ],
            [
                0.0,
                1.0 * torch.sin(1.0 * thetas[2]) / torch.sin(thetas[2]),
                2.0 * torch.sin(2.0 * thetas[2]) / torch.sin(thetas[2]),
                3.0 * torch.sin(3.0 * thetas[2]) / torch.sin(thetas[2])
            ]
        ]) * poly.norm

        self.assertTrue((dpdls - dpdls_true).abs().max() < 1e-8)
        return


if __name__ == "__main__":
    unittest.main()