import abc 

import torch
from torch import Tensor

from .bridge import Bridge


class SingleBeta(Bridge, abc.ABC):
    """Likelihood tempering."""

    def __init__(
        self, 
        betas: Tensor,
        ess_tol: Tensor|float = 0.5, 
        ess_tol_init: Tensor|float = 0.5,
        beta_factor: Tensor|float = 1.05,
        min_beta: Tensor|float = 1e-4
    ):

        self.betas = betas.sort()[0]
        self.ess_tol = torch.tensor(ess_tol)
        self.ess_tol_init = torch.tensor(ess_tol_init)
        self.beta_factor = torch.tensor(beta_factor)
        self.min_beta = torch.tensor(min_beta)
        self.init_beta = torch.tensor(min_beta)
        return