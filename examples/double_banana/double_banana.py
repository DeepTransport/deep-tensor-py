import torch
from torch import Tensor


class DoubleBanana():

    def __init__(self, sigma: float, data: Tensor) -> None:

        # TODO: could make these inputs eventually
        self.dx = 1.5
        self.Q = self.dx * torch.eye(2)
        self.Q_inv = (1.0 / self.dx) * torch.eye(2)
        self.L_inv = (1.0 / self.dx) * torch.eye(2)
        self.b = torch.tensor([0.0, 1.0])

        self.sigma = sigma
        self.data = data
        self.n_data = data.numel()
        return

    def negloglik(self, xs: Tensor) -> Tensor:
        """Evaluates the potential of the likelihood function for the 
        double banana.
        """
        F = torch.log((1-xs[:, 0])**2 + 100*(xs[:, 1]-xs[:, 0]**2)**2)
        F = torch.tile(F, (2, 1)).T
        neglogliks = torch.sum((F-self.data)**2, 1) / (2*self.sigma**2)
        return neglogliks
    
    def neglogpri(self, xs: Tensor) -> Tensor:
        """Evaluates the prior for the double banana.
        """
        neglogpris = 0.5 * ((xs - self.b) @ self.L_inv.T).square().sum(dim=1)
        return neglogpris
    

class ConditionalBanana():

    def __init__(self, sigma: Tensor|int):
        self.sigma = sigma 

    def eval_potential_joint(self, z: Tensor):
        y = z[0, :]
        u = z[1:3, :]
