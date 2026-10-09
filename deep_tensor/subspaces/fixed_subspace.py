from __future__ import annotations

import math
from typing import Callable, Tuple

import torch
from torch import Tensor

from .subspace import Subspace
from ..references import GaussianReference


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

    """

    def __init__(
        self, 
        basis: Tensor,
        num_comp: int = 0,
        fixed_comp: bool = True, 
        device: torch.device = torch.get_default_device()
    ):
        self._basis_red = basis
        self._basis_comp = self._compute_basis_comp(self._basis_red)
        self._num_comp = num_comp
        self._fixed_comp = fixed_comp
        self._device = device
        self._num_eval = 0
        self._num_eval_grad = 0
        if self._fixed_comp and self._num_comp > 0:
            self._compute_samples_comp(self._num_comp)
        return
    
    @property
    def _is_fixed(self) -> bool:
        return True

    def _eval_neglogprofile(
        self,
        eval_neglogratio: Callable[[Tensor], Tensor],
        vs_red: Tensor
    ) -> Tensor:
        
        xs_red = self._eval_coef2red(vs_red)

        if self._num_comp == 0:
            return eval_neglogratio(xs_red)
        
        num_red = xs_red.shape[0]
        if self._fixed_comp:
            xs_comp = self._xs_comp[None, :, :]
        else: 
            xs_comp = self._generate_xs_comp(self._num_comp * num_red)
            xs_comp = xs_comp.reshape(num_red, self._num_comp, self._dim)
        
        xs = xs_red[:, None, :] + xs_comp
        xs = xs.reshape(-1, self._dim)
        neglogfxs = eval_neglogratio(xs)
        neglogfxs = neglogfxs.reshape(num_red, self._num_comp)
        neglogfxs_mean = (
            - torch.logsumexp(-neglogfxs, dim=1)
            + math.log(self._num_comp)
        )
        return neglogfxs_mean 
    
    def _update(
        self, 
        grad_neglogratio: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]],
        reference: GaussianReference
    ) -> None:
        return
    
    def _clone(self) -> FixedSubspace:
        subspace = FixedSubspace(
            self._basis_red, 
            self._num_comp, 
            self._fixed_comp, 
            self._device
        )
        return subspace
