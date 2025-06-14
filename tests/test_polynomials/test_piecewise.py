import unittest

import torch

import deep_tensor as dt


torch.manual_seed(0)


class TestPiecewisePolynomials(unittest.TestCase):

    def test_lagrange_1d(self):
        """Verifies that some simple operations with Lagrange1 
        polynomials work as intended.
        """

        poly = dt.Lagrange1(num_elems=4)

        ls = torch.linspace(-1.0, 1.0, 9)
        coeffs = torch.tensor([[2.0], [3.0], [2.0], [3.0], [2.0]])

        basis_vals = poly.eval_basis(ls)
        weights = poly.eval_measure(ls)
        radon_vals = poly.eval_radon(coeffs, ls)
        func_vals = poly.eval(coeffs, ls)

        basis_vals_true = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])

        weights_true = torch.full((9,), 0.5)

        radon_vals_true = torch.tensor([
            [2.0], [2.5], [3.0], 
            [2.5], [2.0], [2.5], 
            [3.0], [2.5], [2.0]
        ])

        func_vals_true = torch.tensor([
            [1.00], [1.25], [1.50],
            [1.25], [1.00], [1.25],
            [1.50], [1.25], [1.00]
        ])

        self.assertTrue((basis_vals_true - basis_vals).abs().max() < 1e-8)
        self.assertTrue((weights_true - weights).abs().max() < 1e-8)
        self.assertTrue((radon_vals_true - radon_vals).abs().max() < 1e-8)
        self.assertTrue((func_vals_true - func_vals).abs().max() < 1e-8)

        return
    
    def test_lagrange_1d_deriv(self):
        """Verifies that the derivative operations of a Lagrange1 
        polynomial work as intended.
        """

        poly = dt.Lagrange1(num_elems=4)

        ls = torch.linspace(-0.75, 0.75, 4)
        coeffs = torch.tensor([[2.0], [3.0], [2.0], [3.0], [2.0]])

        deriv_vals = poly.eval_basis_deriv(ls)
        weights_deriv = poly.eval_measure_deriv(ls)
        radon_deriv_vals = poly.eval_radon_deriv(coeffs, ls)
        func_deriv_vals = poly.eval_deriv(coeffs, ls)

        deriv_vals_true = torch.tensor([
            [-2.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, -2.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, -2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, -2.0, 2.0]
        ])

        weights_deriv_true = torch.zeros((4,))
        radon_deriv_vals_true = torch.tensor([[2.0], [-2.0], [2.0], [-2.0]])
        func_deriv_vals_true = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])

        self.assertTrue((deriv_vals_true - deriv_vals).abs().max() < 1e-8)
        self.assertTrue((weights_deriv_true - weights_deriv).abs().max() < 1e-8)
        self.assertTrue((radon_deriv_vals_true - radon_deriv_vals).abs().max() < 1e-8)
        self.assertTrue((func_deriv_vals_true - func_deriv_vals).abs().max() < 1e-8)

        return