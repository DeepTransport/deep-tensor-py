from typing import List, Tuple

import torch 
from torch import Tensor

from .marginal_estimator import MarginalEstimator


class EmpiricalMarginals(object):
    """A set of marginal densities estimated from data.

    TODO: eventually might need to pass in a set of indices, so as to 
    be able to evaluate the final n cdfs.
    """

    def __init__(self, marginals: List[MarginalEstimator]):
        self.marginals = marginals 
        self.dim = len(marginals)
        return
    
    def eval_cdf(self, xs: Tensor) -> Tuple[Tensor, Tensor]:
        d_xs = xs.shape[1]
        zs = torch.zeros_like(xs)
        dzdxs = torch.zeros_like(xs)
        for i in range(d_xs):
            zs[:, i], dzdxs[:, i] = self.marginals[i].eval_cdf(xs[:, i])
        return zs, dzdxs
    
    def invert_cdf(self, zs: Tensor) -> Tensor:
        d_zs = zs.shape[1]
        xs = torch.zeros_like(zs)
        for i in range(d_zs):
            xs[:, i] = self.marginals[i].invert_cdf(zs[:, i])
        return xs

    def eval_potential(self, xs: Tensor) -> Tensor:
        d_xs = xs.shape[1]
        neglogfxs = torch.zeros((xs.shape[0],))
        for i in range(d_xs):
            neglogfxs += self.marginals[i].eval_potential(xs[:, i])
        return neglogfxs