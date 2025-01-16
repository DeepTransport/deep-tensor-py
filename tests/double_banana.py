import torch

import deep_tensor as dt

from examples.double_banana.double_banana import DoubleBanana


def build_double_banana_dirt() -> dt.TTDIRT:

    dim = 2
    sigma = 0.3
    data = torch.tensor([3.0, 5.0])
    model = DoubleBanana(sigma, data)

    bounds = torch.tensor([-4.0, 4.0])
    domain = dt.BoundedDomain(bounds)

    poly = dt.Lagrange1(num_elems=50)
    bases = dt.ApproxBases(poly, domain, dim)

    bridge = dt.Tempering1()
    dirt = dt.TTDIRT(model.potential_dirt, bases, bridge=bridge)
    
    return dirt