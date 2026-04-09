import unittest

import math
import torch
from torch import Tensor

import deep_tensor as dt


EPS = 1e-2


class TestAffineMapping(unittest.TestCase):

    @staticmethod
    def generate_mapping(A: Tensor, b: Tensor | None = None):
        preconditioner = dt.AffineMapping(A, b)
        return preconditioner

    def test_Q_identity(self):
        """Tests the Q() method with an identity transformation."""

        dim = 3
        A = torch.eye(dim)
        b = torch.zeros((dim,))
        us = torch.tensor([[-2.0, 0.0, 2.0],
                           [1.0, 2.0, 3.0]])

        preconditioner = self.generate_mapping(A, b)        
        xs, neglogdets = preconditioner.Q(us)
        
        xs_true = us.clone()
        neglogdets_true = torch.zeros_like(us[:, 0])
        
        self.assertTrue((xs-xs_true).abs().max() < EPS)
        self.assertTrue((neglogdets-neglogdets_true).abs().max() < EPS)
        return
    
    def test_Q_nonidentity(self):
        """Tests the Q() method with a non-identity transformation."""

        dim = 3
        A = 2.0 * torch.eye(dim)
        b = torch.tensor([1.0, 2.0, 3.0])
        us = torch.tensor([[-2.0, 0.0, 2.0],
                           [1.0, 2.0, 3.0]])

        preconditioner = self.generate_mapping(A, b)        
        xs, neglogdets = preconditioner.Q(us)
        
        xs_true = torch.tensor([[-3.0, 2.0, 7.0],
                                [3.0, 6.0, 9.0]])
        neglogdets_true = torch.full((2,), -math.log(8.0))
        
        self.assertTrue((xs-xs_true).abs().max() < EPS)
        self.assertTrue((neglogdets-neglogdets_true).abs().max() < EPS)
        return
    
    def test_Q_inv_identity(self):
        """Tests the Q_inv() method with an identity transformation."""

        dim = 3
        A = torch.eye(dim)
        b = torch.zeros((dim,))
        xs = torch.tensor([[-2.0, 0.0, 2.0],
                           [1.0, 2.0, 3.0]])

        preconditioner = self.generate_mapping(A, b)
        us, neglogdets = preconditioner.Q_inv(xs)

        us_true = xs.clone()
        neglogdets_true = torch.zeros_like(xs[:, 0])
        
        self.assertTrue((us-us_true).abs().max() < EPS)
        self.assertTrue((neglogdets-neglogdets_true).abs().max() < EPS)
        return
    
    def test_Q_inv_nonidentity(self):
        """Tests the Q_inv() method with a non-identity transformation."""

        dim = 3
        A = 2.0 * torch.eye(dim)
        b = torch.tensor([1.0, 2.0, 3.0])
        us = torch.tensor([[-2.0, 0.0, 2.0],
                           [1.0, 2.0, 3.0]])

        preconditioner = self.generate_mapping(A, b)        
        xs, neglogdets = preconditioner.Q_inv(us)
        
        xs_true = torch.tensor([[-1.5, -1.0, -0.5],
                                [0.0, 0.0, 0.0]])
        neglogdets_true = torch.full((2,), math.log(8.0))
        
        self.assertTrue((xs-xs_true).abs().max() < EPS)
        self.assertTrue((neglogdets-neglogdets_true).abs().max() < EPS)
        return