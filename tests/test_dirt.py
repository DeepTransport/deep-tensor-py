import unittest

import torch
from torch.linalg import norm

from double_banana import build_double_banana_dirt


torch.manual_seed(0)


class TestDIRT(unittest.TestCase):
    """Verifies that the deep Rosenblatt transport and inverse 
    Rosenblatt transports of a DIRT object are actually inverses of one
    another using the double banana model.
    """

    def test_double_banana_dirt(self):
        
        dirt = build_double_banana_dirt()

        rs = dirt.reference.random(d=2, n=1000)

        xs = dirt.eval_irt(rs)[0]
        r0 = dirt.eval_rt(xs)[0]

        error = norm(rs-r0, ord="fro")
        self.assertTrue(error < 1e-8)

        return