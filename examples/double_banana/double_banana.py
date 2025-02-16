import torch
from torch import Tensor


class DoubleBanana():

    def __init__(self, sigma: float, data: Tensor):
        self.sigma = sigma
        self.data = data
        self.num_data = data.numel()

    def potential_dirt(self, u):
        """Evaluates the DoubleBanana pdf on a grid of values, u."""

        F = torch.log((1-u[:, 0])**2 + 100*(u[:, 1]-u[:, 0]**2)**2)
        F = torch.tile(F, (2, 1)).T

        potential_likelihood = torch.sum((F-self.data)**2, 1) / (2*self.sigma**2)
        potential_prior = 0.5 * torch.sum(u**2, 1)

        return potential_likelihood, potential_prior
    

class ConditionalBanana():

    def __init__(self, sigma: Tensor|int):
        self.sigma = sigma 

    def eval_potential_joint(self, z: Tensor):
        y = z[0, :]
        u = z[1:3, :]
