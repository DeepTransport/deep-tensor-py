import unittest

import torch

import deep_tensor as dt


torch.manual_seed(0)


class TestUniformReference(unittest.TestCase):

    def build_reference(self) -> dt.UniformReference:
        domain = torch.tensor([1., 3.])
        ref = dt.UniformReference(domain)
        return ref

    def test_pdf_and_cdf(self):
        """Tests that the PDF and CDF of the uniform reference 
        distribution are evaluated correctly.
        """

        ref = self.build_reference()

        rs = torch.tensor([1., 1.5, 2., 2.5, 3.])

        zs, dzdrs = ref.eval_cdf(rs)
        zs_true = torch.tensor([0., 0.25, 0.5, 0.75, 1.0])
        dzdrs_true = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])

        pdfs, grad_pdfs = ref.eval_pdf(rs)
        pdfs_true = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        grad_pdfs_true = torch.zeros(5)

        self.assertTrue((zs - zs_true).abs().max() < 1e-8)
        self.assertTrue((dzdrs - dzdrs_true).abs().max() < 1e-8)
        self.assertTrue((pdfs - pdfs_true).abs().max() < 1e-8)
        self.assertTrue((grad_pdfs - grad_pdfs_true).abs().max() < 1e-8)
        return

    def test_inverse_cdf(self):
        """Tests that the invert_cdf method works correctly.
        """

        ref = self.build_reference()

        zs = torch.tensor([0., 0.25, 0.5, 0.75, 1.])
        rs = ref.invert_cdf(zs)
        rs_true = torch.tensor([1., 1.5, 2., 2.5, 3.])

        self.assertTrue((rs - rs_true).abs().max() < 1e-8)
        return
    
    def test_log_joint_pdf(self):
        """Tests the the log_joint_pdf method works correctly.
        """

        ref = self.build_reference()

        rs = 1. + 2. * torch.rand(3, 4)

        log_pdfs, log_grad_pdfs = ref.log_joint_pdf(rs)

        log_pdfs_true = torch.full((3,), torch.tensor(1./16.).log())
        log_grad_pdfs_true = torch.zeros(3)

        self.assertTrue((log_pdfs - log_pdfs_true).abs().max() < 1e-8)
        self.assertTrue((log_grad_pdfs - log_grad_pdfs_true).abs().max() < 1e-8)
        return
    
    def test_sample(self):
        """Tests that the sampling methods work as intended.
        """

        ref = self.build_reference()

        rs = ref.random(d=3, n=4)
        self.assertEqual(rs.shape, torch.Size([4, 3]))
        self.assertTrue(rs.min() >= 1.)
        self.assertTrue(rs.max() <= 3.)

        rs = ref.sobol(d=3, n=4)
        self.assertEqual(rs.shape, torch.Size([4, 3]))
        self.assertTrue(rs.min() >= 1.)
        self.assertTrue(rs.max() <= 3.)
        return


if __name__ == "__main__":
    unittest.main()