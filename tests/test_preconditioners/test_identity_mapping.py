import unittest

import torch

import deep_tensor as dt


EPS = 1e-2


class TestIdentityMapping(unittest.TestCase):

    @staticmethod
    def generate_mapping():

        bounds = [-4.0, 4.0]
        domain = dt.BoundedDomain(bounds)
        reference = dt.GaussianReference(domain)

        preconditioner = dt.IdentityMapping(dim=3, reference=reference)
        return preconditioner

    def test_Q(self):
        """Tests the Q() and neglogdet_Q() methods."""

        preconditioner = self.generate_mapping()

        us = torch.tensor([[-2.0, 0.0, 2.0],
                           [1.0, 2.0, 3.0]])
        
        xs, neglogdets = preconditioner.Q(us)

        xs_true = us.clone()

        xs_first, neglogdets_first = preconditioner.Q(us[:, :2], subset="first")
        xs_last, neglogdets_last = preconditioner.Q(us[:, -2:], subset="last")
        neglogdets_true = torch.zeros((2,))
        
        self.assertTrue((xs-xs_true).abs().max() < EPS)
        self.assertTrue((xs_first-xs_true[:, :2]).abs().max() < EPS)
        self.assertTrue((xs_last-xs_true[:, -2:]).abs().max() < EPS)

        self.assertTrue((neglogdets-neglogdets_true).abs().max() < EPS)
        self.assertTrue((neglogdets_first-neglogdets_true).abs().max() < EPS)
        self.assertTrue((neglogdets_last-neglogdets).abs().max() < EPS)
        return
    
    def test_Q_inv(self):
        """Tests the Q_inv() and neglogdet_Q_inv() methods."""

        preconditioner = self.generate_mapping()

        xs = torch.tensor([[-2.0, 0.0, 2.0],
                           [1.0, 2.0, 3.0]])
    
        us, neglogdets = preconditioner.Q_inv(xs)

        us_first, neglogdets_first = preconditioner.Q_inv(xs[:, :2], subset="first")
        us_last, neglogdets_last = preconditioner.Q_inv(xs[:, -2:], subset="last")

        us_true = xs.clone()
        neglogdets_true = torch.zeros((2,))

        self.assertTrue((us-us_true).abs().max() < EPS)
        self.assertTrue((us_first-us_true[:, :2]).abs().max() < EPS)
        self.assertTrue((us_last-us_true[:, -2:]).abs().max() < EPS)

        self.assertTrue((neglogdets-neglogdets_true).abs().max() < EPS)
        self.assertTrue((neglogdets_first-neglogdets_true).abs().max() < EPS)
        self.assertTrue((neglogdets_last-neglogdets).abs().max() < EPS)
        return