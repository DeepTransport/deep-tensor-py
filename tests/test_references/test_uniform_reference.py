import unittest

import torch

import deep_tensor as dt


torch.manual_seed(0)


EPS = 1e-6


class TestUniformReference(unittest.TestCase):

    def test_pdf_and_cdf(self):
        """Tests that the PDF and CDF of the uniform reference 
        distribution are evaluated correctly.
        """

        ref = dt.UniformReference()

        rs = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        zs, dzdrs = ref.eval_cdf(rs)
        zs_true = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        dzdrs_true = torch.ones(5)

        pdfs, grad_pdfs = ref.eval_pdf(rs)
        pdfs_true = torch.ones(5)
        grad_pdfs_true = torch.zeros(5)

        self.assertTrue((zs - zs_true).abs().max() < EPS)
        self.assertTrue((dzdrs - dzdrs_true).abs().max() < EPS)
        self.assertTrue((pdfs - pdfs_true).abs().max() < EPS)
        self.assertTrue((grad_pdfs - grad_pdfs_true).abs().max() < EPS)
        return

    def test_inverse_cdf(self):
        """Tests that the invert_cdf method works correctly.
        """

        ref = dt.UniformReference()

        zs = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        rs = ref.invert_cdf(zs)
        
        rs_true = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        self.assertTrue((rs - rs_true).abs().max() < EPS)
        return
    
    def test_eval_potential(self):
        """Tests that the eval_potential method works correctly.
        """

        ref = dt.UniformReference()

        rs = torch.rand(3, 4)
        log_pdfs, log_grad_pdfs = ref.eval_potential(rs)

        log_pdfs_true = torch.zeros(3)
        log_grad_pdfs_true = torch.zeros(3, 4)

        self.assertTrue((log_pdfs - log_pdfs_true).abs().max() < EPS)
        self.assertTrue((log_grad_pdfs - log_grad_pdfs_true).abs().max() < EPS)
        return
    
    def test_sample(self):
        """Tests that the sampling methods work as intended."""

        ref = dt.UniformReference()

        rs = ref.random(n=4, d=3)
        self.assertEqual(rs.shape, torch.Size([4, 3]))
        self.assertTrue(rs.min() >= 0.0)
        self.assertTrue(rs.max() <= 1.0)

        rs = ref.sobol(n=4, d=3)
        self.assertEqual(rs.shape, torch.Size([4, 3]))
        self.assertTrue(rs.min() >= 0.0)
        self.assertTrue(rs.max() <= 1.0)
        return


if __name__ == "__main__":
    unittest.main()