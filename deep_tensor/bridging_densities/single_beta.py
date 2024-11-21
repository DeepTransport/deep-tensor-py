import abc 

import torch

from .bridge import Bridge


class SingleBeta(Bridge, abc.ABC):
    """Likelihood tempering."""

    def __init__(
        self, 
        betas: torch.Tensor,
        ess_tol: float=0.5, 
        ess_tol_init: float=0.5, # TODO: make these consts somewhere?
        beta_factor: float=1.05,
        min_beta: float=1e-4
    ):

        self.betas = betas

        self.ess_tol = torch.tensor(ess_tol)
        self.ess_tol_init = torch.tensor(ess_tol_init)

        self.beta_factor = torch.tensor(beta_factor)
        self.min_beta = torch.tensor(min_beta)
        self.init_beta = torch.tensor(min_beta)

        return

