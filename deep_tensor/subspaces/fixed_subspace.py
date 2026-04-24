import math
from typing import Callable, Tuple

import torch
from torch import Tensor

from .subspace import Subspace


class FixedSubspace(Subspace):
    r"""A fixed subspace.
    
    Parameters
    ----------
    basis:
        The basis for the subspace.
    num_comp:
        The number of samples from the complement subspace to use when 
        evaluating the profile function.
    fixed_comp:
        Whether to fix the samples from the complement subspace.
    device:
        The device to carry out computations on.

    TODO: need to pass in complement space samples etc..
    TODO: need to check the evaluations... should this be recorded in 
    the DIRT/SIRT object? I don't know what is being recorded in the 
    reduced subspaces.

    """

    def __init__(
        self, 
        basis: Tensor,
        num_comp: int = 0,
        fixed_comp: bool = True, 
        device: torch.device = torch.get_default_device()
    ):
        self.basis_red = basis
        self.basis_comp = self._compute_basis_comp(self.basis_red)
        self.num_comp = num_comp
        self.fixed_comp = fixed_comp
        self.device = device
        self.num_eval = 0
        self.num_eval_grad = 0
        if self.fixed_comp and self.num_comp > 0:
            self._compute_samples_comp(self.num_comp)
        return
    
    @property
    def is_fixed(self) -> bool:
        return True

    def eval_neglogprofile(
        self,
        eval_neglogratio: Callable[[Tensor], Tensor],
        vs_red: Tensor
    ) -> Tensor:
        
        xs_red = self.eval_coef2red(vs_red)

        if self.num_comp == 0:
            return eval_neglogratio(xs_red)
    
        if not self.fixed_comp:
            self._compute_samples_comp(self.num_comp)
        
        num_red = xs_red.shape[0]
        num_comp = self.xs_comp.shape[0]
        xs = xs_red[:, None, :] + self.xs_comp[None, :, :]
        xs = xs.reshape(-1, self.dim)
        neglogfxs = eval_neglogratio(xs)
        neglogfxs = neglogfxs.reshape(num_red, num_comp)
        neglogfxs_mean = (
            - torch.logsumexp(-neglogfxs, dim=1)
            + math.log(num_comp)
        )
        return neglogfxs_mean 
    
    def update(
        self, 
        grad_neglogbridge: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]], 
        grad_neglogratio: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        return
    
    def clone(self) -> FixedSubspace:
        subspace = FixedSubspace(
            self.basis_red, 
            self.num_comp, 
            self.fixed_comp, 
            self.device
        )
        return subspace
