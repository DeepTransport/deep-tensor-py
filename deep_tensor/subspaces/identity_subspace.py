from typing import Callable, Tuple

import torch 
from torch import Tensor

from .subspace import Subspace


class IdentitySubspace(Subspace):
    r"""Identity subspace (*i.e.,* no dimension reduction).
    
    Parameters
    ----------
    dim:
        The dimension of the target density.
    device:
        The device to carry out computations on.
    
    """

    def __init__(
        self, 
        dim: int, 
        device: torch.device = torch.get_default_device()
    ):
        self.device = device
        self.basis_red = torch.eye(dim, device=self.device)
        self.basis_comp = torch.zeros((dim, 0), device=self.device)
        self.num_eval = 0
        self.num_eval_grad = 0
        return
    
    @property 
    def is_fixed(self) -> bool:
        return True
    
    def eval_coef2red(self, vs: Tensor) -> Tensor:
        # Note: we need to override the default implementation here 
        # (and in eval_red2coef, eval_coef2comp, eval_comp2coef, 
        # project_red, project_comp) because the implementation does 
        # not work when evaluating marginal densities. The 
        # IdentitySubspace is the only subspace that allows for the 
        # evaluation of marginal densities.
        return vs
    
    def eval_red2coef(self, xs: Tensor) -> Tensor:
        return xs
    
    def eval_coef2comp(self, ws: Tensor) -> Tensor:
        # Note: when evaluating marginal functions, we do not know the 
        # dimension here. Giving the output a single dimension allows 
        # it to broadcast with the component in the reduced space 
        # regardless of the dimension.
        num_ws = ws.shape[0]
        xs_comp = torch.zeros((num_ws, 1), device=self.device)
        return xs_comp
    
    def eval_comp2coef(self, xs: Tensor) -> Tensor:
        num_xs = xs.shape[0]
        ws = torch.zeros((num_xs, 0), device=self.device)
        return ws
    
    def project_red(self, xs: Tensor) -> Tensor:
        return xs
    
    def project_comp(self, xs: Tensor) -> Tensor:
        return torch.zeros_like(xs)
    
    def eval_neglogprofile(
        self, 
        eval_neglogtarget: Callable[[Tensor], Tensor], 
        xs: Tensor
    ) -> Tensor:
        return eval_neglogtarget(xs)
    
    def update(
        self, 
        grad_neglogbridge: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]],
        grad_neglogratio: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]]
    ) -> None: 
        return

    def clone(self) -> IdentitySubspace:
        return IdentitySubspace(self.dim)