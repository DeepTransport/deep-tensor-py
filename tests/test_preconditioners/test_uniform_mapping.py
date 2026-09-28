import unittest

import torch

import deep_tensor as dt


EPS = 1e-2


class TestUniformMapping(unittest.TestCase):

    @staticmethod
    def generate_mapping():

        bounds = [-4.0, 4.0]
        domain = dt.BoundedDomain(bounds)
        reference = dt.GaussianReference(domain)

        bounds = torch.tensor([[-2.0, 2.0], 
                               [0.0, 3.0], 
                               [-1.0, 2.0]])
        preconditioner = dt.UniformMapping(bounds, reference)
        return preconditioner

    def test_Q(self):
        """Tests the Q() and neglogdet_Q() methods."""

        preconditioner = self.generate_mapping()

        us = torch.tensor([[-4.0, 0.0, 4.0], 
                           [4.0, 0.0, -4.0]])
        
        xs = preconditioner.Q(us)[0]

        xs_true = torch.tensor([[-2.0, 1.5, 2.0], 
                                [2.0, 1.5, -1.0]])

        xs_first = preconditioner.Q(us[:, :2], subset="first")[0]
        xs_last = preconditioner.Q(us[:, -2:], subset="last")[0]
        
        self.assertTrue((xs-xs_true).abs().max() < EPS)
        self.assertTrue((xs_first-xs_true[:, :2]).abs().max() < EPS)
        self.assertTrue((xs_last-xs_true[:, -2:]).abs().max() < EPS)
        return
    
    def test_Q_inv(self):
        """Tests the Q_inv() and neglogdet_Q_inv() methods."""

        preconditioner = self.generate_mapping()

        xs = torch.tensor([[-2.0, 1.5, 2.0], 
                           [2.0, 1.5, -1.0]])
    
        us = preconditioner.Q_inv(xs)[0]

        us_true = torch.tensor([[-4.0, 0.0, 4.0], 
                                [4.0, 0.0, -4.0]])

        us_first = preconditioner.Q_inv(xs[:, :2], subset="first")[0]
        us_last = preconditioner.Q_inv(xs[:, -2:], subset="last")[0]

        self.assertTrue((us-us_true).abs().max() < EPS)
        self.assertTrue((us_first-us_true[:, :2]).abs().max() < EPS)
        self.assertTrue((us_last-us_true[:, -2:]).abs().max() < EPS)
        return