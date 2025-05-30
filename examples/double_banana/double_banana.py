import torch
from torch import Tensor


class DoubleBanana():

    def __init__(self, sigma: float, data: Tensor) -> None:

        self.sigma = sigma
        self.data = data
        self.n_data = data.numel()
        return

    def param_to_obs(self, xs: Tensor) -> Tensor:
        """Evaluates the parameter-to-observable mapping.
        """
        F = torch.log((1-xs[:, 0]).square() + 100*(xs[:, 1]-xs[:, 0].square()).square())
        return F[:, None]

    def _negloglik(self, xs: Tensor, y_obs: Tensor) -> Tensor:
        """Evaluates the potential of the likelihood function for the 
        double banana.
        """
        F = self.param_to_obs(xs).repeat(1, 2)
        neglogliks = (F - y_obs).square().sum(dim=1) / (2*self.sigma**2)
        return neglogliks

    def negloglik(self, xs: Tensor) -> Tensor:
        """Evaluates the potential of the likelihood function for the 
        double banana.
        """
        return self._negloglik(xs, self.data)
    
    def neglogpri(self, xs: Tensor) -> Tensor:
        """Evaluates the prior for the double banana.
        """
        neglogpris = 0.5 * xs.square().sum(dim=1)
        return neglogpris
    
    def negloglik_joint(self, yxs: Tensor) -> Tensor:
        ys, xs = yxs[:, :1], yxs[:, -2:]
        return self._negloglik(xs, ys)

    def neglogpri_joint(self, yxs: Tensor) -> Tensor:
        xs = yxs[:, -2:]
        return self.neglogpri(xs)
    
    def potential_joint(self, yxs: Tensor) -> Tensor:
        return self.negloglik_joint(yxs) + self.neglogpri_joint(yxs)