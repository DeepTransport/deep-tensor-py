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

        nodes_true = torch.tensor([-1., -0.5, 0., 0.5, 1.])

        V_inv_true = torch.tensor([[ 1.,  0.,  0.],
                                   [-3.,  4., -1.],
                                   [ 2., -4.,  2.]])

        node2elem_true = torch.tensor([[1., 0., 0., 0., 0.], 
                                       [0., 1., 0., 0., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 0., 1., 0., 0.],
                                       [0., 0., 0., 1., 0.],
                                       [0., 0., 0., 0., 1.]])

        self.assertTrue((cdf.nodes - nodes_true).abs().max() < 1e-8)
        self.assertTrue((V_inv_true - cdf.V_inv).abs().max() < 1e-8)
        self.assertTrue((node2elem_true - cdf.node2elem).abs().max() < 1e-8)
        return
    
    def test_pdf2cdf(self):
        """Checks that the values of the CDF for a Lagrange1 polynomial 
        are computed correctly.
        """

        cdf = self.setup_cdf()

        ps = torch.tensor([1., 2., 3., 2.5, 2.]).square()

        poly_coef_true = torch.tensor([[[1., 4., 4.]], 
                                        [[9., -6., 1.]]])

        cdf_poly_grid_true = torch.tensor([[0.],
                                           [13./3.], 
                                           [32./3.]])
        
        poly_norm_true = torch.tensor([32./3.])
        cdf_data = cdf.pdf2cdf(ps)

        self.assertTrue(cdf_data.n_cdfs == 1)
        self.assertTrue((poly_coef_true - cdf_data.poly_coef).abs().max() < 1e-8)
        self.assertTrue((cdf_poly_grid_true - cdf_data.cdf_poly_grid).abs().max() < 1e-8)
        self.assertTrue((poly_norm_true - cdf_data.poly_norm).abs().max() < 1e-8)
        return
    
    def test_eval_cdf(self):
        """Checks that the CDF of a Lagrange1 polynomial is evaluated 
        correctly.
        """

        cdf = self.setup_cdf()

        ls = torch.tensor([-0.5, 0.5])
        zs_true = torch.tensor([7./64., 195./256.])

        # Test case where there is a single PDF for all samples
        pls = torch.tensor([1., 2., 3., 2.5, 2.]).square()
        zs = cdf.eval_cdf(pls, ls)
        self.assertTrue((zs - zs_true).abs().max() < 1e-8)

        # Test case where there is an individual PDF for each sample
        pls = torch.hstack((pls[:, None], pls[:, None]))
        zs = cdf.eval_cdf(pls, ls)
        self.assertTrue((zs - zs_true).abs().max() < 1e-8)
        return

    def test_invert_cdf(self):
        """Checks that the inverse CDF method of a Lagrange1 polynomial 
        is evaluated correctly.
        """

        cdf = self.setup_cdf()

        zs = torch.tensor([7./64., 195./256.])
        ls_true = torch.tensor([-0.5, 0.5])

        # Test case where there is a single PDF for all samples
        pls = torch.tensor([1., 2., 3., 2.5, 2.]).square()
        ls = cdf.invert_cdf(pls, zs)
        self.assertTrue((ls - ls_true).abs().max() < 1e-8)

        # Test case where there is an individual PDF for each sample
        pls = torch.hstack((pls[:, None], pls[:, None]))
        ls = cdf.invert_cdf(pls, zs)
        self.assertTrue((ls - ls_true).abs().max() < 1e-8)
        return


if __name__ == "__main__":
    unittest.main()