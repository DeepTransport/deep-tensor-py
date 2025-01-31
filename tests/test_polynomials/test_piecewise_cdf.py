import unittest

import torch

import deep_tensor as dt


torch.manual_seed(0)


class TestPiecewiseCDF(unittest.TestCase):

    def setup_cdf(self):
        poly = dt.Lagrange1(num_elems=2)
        cdf = dt.Lagrange1CDF(poly=poly)
        return cdf

    def test_lagrange_1d_cdf(self):
        """Verifies that the attributes of a Lagrange1CDF object are 
        correctly initialised.
        """
        
        cdf = self.setup_cdf()

        nodes_true = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])

        V_inv_true = torch.tensor([[ 1.0,  0.0,  0.0],
                                   [-3.0,  4.0, -1.0],
                                   [ 2.0, -4.0,  2.0]])

        node2elem_true = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], 
                                       [0.0, 1.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 1.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 1.0]])

        self.assertTrue((cdf.nodes - nodes_true).abs().max() < 1e-8)
        self.assertTrue((V_inv_true - cdf.V_inv).abs().max() < 1e-8)
        self.assertTrue((node2elem_true - cdf.node2elem).abs().max() < 1e-8)
        return
    
    def test_pdf2cdf(self):
        """Checks that the values of the CDF for a Lagrange1 polynomial 
        are computed correctly.
        """

        cdf = self.setup_cdf()

        pls = torch.tensor([1.0, 2.0, 3.0, 2.5, 2.0]).square()

        poly_coef_true = torch.tensor([[1.0,  9.0], 
                                       [4.0, -6.0], 
                                       [4.0,  1.0]])

        cdf_poly_grid_true = torch.tensor([[0.0],
                                           [13.0/3.0], 
                                           [32.0/3.0]])
        
        poly_norm_true = torch.tensor([32.0/3.0])

        cdf_data = cdf.pdf2cdf(pls)

        self.assertTrue(cdf_data.num_samples == 1)
        self.assertTrue((poly_coef_true - cdf_data.poly_coef).abs().max() < 1e-8)
        self.assertTrue((cdf_poly_grid_true - cdf_data.cdf_poly_grid).abs().max() < 1e-8)
        self.assertTrue((poly_norm_true - cdf_data.poly_norm).abs().max() < 1e-8)
        return
    
    def test_eval_cdf(self):
        """Checks that the CDF of a Lagrange1 polynomial is evaluated 
        correctly.
        """

        cdf = self.setup_cdf()

        pls = torch.tensor([1.0, 2.0, 3.0, 2.5, 2.0]).square()
        ls = torch.tensor([-0.5, 0.5])

        zs_true = torch.tensor([7.0/64.0, 195.0/256.0])

        zs = cdf.eval_cdf(pls, ls)

        self.assertTrue((zs - zs_true).abs().max() < 1e-8)
        return

if __name__ == "__main__":
    unittest.main()