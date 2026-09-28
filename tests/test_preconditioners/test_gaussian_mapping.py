import unittest

import math
import torch

import deep_tensor as dt


EPS = 1e-8


class TestGaussianMapping(unittest.TestCase):

    def test_Q_identity(self):
        """Tests the Q() method with an identity transformation."""

        dim = 3
        mean = torch.zeros((dim,))
        cov = torch.eye(dim)
        us = torch.tensor([[-2.0, 0.0, 2.0],
                           [1.0, 2.0, 3.0]])

        preconditioner = dt.GaussianMapping(mean, cov)
        xs, neglogdets = preconditioner.Q(us)
        
        xs_true = us.clone()
        neglogdets_true = torch.zeros_like(us[:, 0])
        
        self.assertTrue((xs-xs_true).abs().max() < EPS)
        self.assertTrue((neglogdets-neglogdets_true).abs().max() < EPS)
        return
    
    def test_Q_nonidentity(self):
        """Tests the Q() method with a non-identity transformation."""

        mean = torch.tensor([1.0, 2.0])
        cov = torch.tensor([[1.0, 0.5], 
                            [0.5, 1.0]])
        us = torch.tensor([[-2.0, 0.0],
                           [1.0, 2.0]])
        us_first = torch.tensor([[1.0],
                                 [3.0]])

        preconditioner = dt.GaussianMapping(mean, cov)
        xs, neglogdets = preconditioner.Q(us)
        xs_first, neglogdets_first = preconditioner.Q(us_first)
        
        xs_true = torch.tensor([[-1.0, 1.0],
                                [2.0, 2.5 + math.sqrt(3.0)]])
        xs_first_true = torch.tensor([[2.0],
                                      [4.0]])
        neglogdets_true = torch.full((2,), -math.log(0.5*math.sqrt(3.0)))
        neglogdets_first_true = torch.zeros((2,))
        
        self.assertTrue((xs-xs_true).abs().max() < EPS)
        self.assertTrue((xs_first-xs_first_true).abs().max() < EPS)
        self.assertTrue((neglogdets-neglogdets_true).abs().max() < EPS)
        self.assertTrue((neglogdets_first-neglogdets_first_true).abs().max() < EPS)
        return
    
    def test_Q_inv_identity(self):
        """Tests the Q_inv() method with an identity transformation."""

        dim = 3
        mean = torch.zeros((dim,))
        cov = torch.eye(dim)
        xs = torch.tensor([[-2.0, 0.0, 2.0],
                           [1.0, 2.0, 3.0]])

        preconditioner = dt.GaussianMapping(mean, cov)
        us, neglogdets = preconditioner.Q_inv(xs)

        us_true = xs.clone()
        neglogdets_true = torch.zeros_like(xs[:, 0])
        
        self.assertTrue((us-us_true).abs().max() < EPS)
        self.assertTrue((neglogdets-neglogdets_true).abs().max() < EPS)
        return
    
    def test_Q_inv_nonidentity(self):
        """Tests the Q_inv() method with a non-identity transformation."""

        mean = torch.tensor([1.0, 2.0])
        cov = torch.tensor([[1.0, 0.5], 
                            [0.5, 1.0]])
        xs = torch.tensor([[-2.0, 0.0],
                           [1.0, 2.0]])
        xs_first = torch.tensor([[1.0],
                                 [3.0]])

        preconditioner = dt.GaussianMapping(mean, cov)
        us, neglogdets = preconditioner.Q_inv(xs)
        us_first, neglogdets_first = preconditioner.Q_inv(xs_first, subset="first")
        
        us_true = torch.tensor([[-3.0, -1.0 / math.sqrt(3.0)],
                                [0.0, 0.0]])
        us_first_true = torch.tensor([[0.0],
                                      [2.0]])
        neglogdets_true = torch.full((2,), -math.log(2.0 / math.sqrt(3.0)))
        neglogdets_first_true = torch.zeros((2,))
        
        self.assertTrue((us-us_true).abs().max() < EPS)
        self.assertTrue((us_first-us_first_true).abs().max() < EPS)
        self.assertTrue((neglogdets-neglogdets_true).abs().max() < EPS)
        self.assertTrue((neglogdets_first-neglogdets_first_true).abs().max() < EPS)
        return