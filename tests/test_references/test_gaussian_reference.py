import math
import unittest 

import torch

import deep_tensor as dt



class TestGaussianReference(unittest.TestCase):

    def test_pdf_and_cdf(self):

        ref = dt.GaussianReference()

        # Compute area under density
        cdf_left = 0.5 * (1.0 - math.erf(2.0 * math.sqrt(2.0)))
        norm = 1.0 - 2.0 * cdf_left

        rs = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])

        pdfs, grad_pdfs = ref.eval_pdf(rs)  # TODO: check the gradient
        cdfs, grad_cdfs = ref.eval_cdf(rs)

        # Compute exact PDF
        exps = torch.tensor([math.exp(8), math.exp(2), 1.0, math.exp(2), math.exp(8)])
        pdfs_true = (1.0 / (exps * math.sqrt(2.0 * math.pi))) / norm

        # Compute exact CDF
        cdfs_true = torch.tensor([0.5 * (1 - math.erf(2.0 * math.sqrt(2))), 
                                  0.5 * (1 - math.erf(math.sqrt(2))), 
                                  0.5, 
                                  0.5 * (math.erf(math.sqrt(2)) + 1), 
                                  0.5 * (math.erf(2*math.sqrt(2)) + 1)])
        cdfs_true = (cdfs_true - cdf_left) / norm
        
        self.assertTrue((pdfs-pdfs_true).abs().max() < 1e-6)
        self.assertTrue((grad_cdfs-pdfs_true).abs().max() < 1e-6)

        self.assertTrue((cdfs-cdfs_true).abs().max() < 1e-6)
        return 
    
    def test_sample(self):
        """Tests that the sampling methods work as intended."""

        ref = dt.GaussianReference()

        rs = ref.random(n=4, d=3)
        self.assertEqual(rs.shape, torch.Size([4, 3]))
        self.assertTrue(rs.min() >= -4.0)
        self.assertTrue(rs.max() <= 4.0)

        rs = ref.sobol(n=4, d=3)
        self.assertEqual(rs.shape, torch.Size([4, 3]))
        self.assertTrue(rs.min() >= -4.0)
        self.assertTrue(rs.max() <= 4.0)
        return