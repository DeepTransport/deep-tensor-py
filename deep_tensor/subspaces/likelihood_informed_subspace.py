from __future__ import annotations

import logging
import math
from typing import Callable, Tuple

import torch 
from torch import Tensor

from .subspace import Subspace
from ..debiasing.importance_sampling import estimate_ess_ratio
from ..references import GaussianReference
from ..tools.printing import lis_info


logger = logging.getLogger(__name__)

UPDATE_METHODS_LIS = ("rebuild", "augment")


class LikelihoodInformedSubspace(Subspace):
    r"""A likelihood-informed subspace.

    Parameters
    ----------
    dim:
        The dimension of the (full) target random variable.
    num_comp:
        The number of samples from the complement subspace to use when 
        evaluating the profile function.
    fixed_comp:
        Whether to fix the samples from the complement subspace.
    update_method:
        How to update the subspace. This can be `'rebuild'` (construct 
        a new subspace from scratch at each DIRT layer), or `'augment'` 
        (retain the previously-constructed subspace at each DIRT layer, 
        and potentially add new components).
    num_samples_gram:
        The number of samples to use to construct a Monte Carlo 
        estimate of the Gram matrix.
    eps:
        The tolerance, $\epsilon$, used to select the dimension of the 
        subspace. The dimension of the subspace is the smallest $r$ 
        such that 
        $$
            \frac{1}{2}\left(\sum_{k=r+1}^{d}\lambda_{k}\right)^{1/2} 
                \leq \epsilon,
        $$
        where $\{\lambda_{k}\}_{k=1}^{n}$ denote the eigenvalues of the 
        current Gram matrix ordered from largest to smallest.
    initial_basis:
        A set of basis vectors to initialise the subspace with. Note 
        that this is only supported if `update_method='augment'`.
    verbose:
        Whether to print diagnostic information (ESS of the samples 
        used as part of the importance sampling estimate of the Gram 
        matrix, and the subspace dimension) when the subspace is 
        updated.
    device:
        The device to carry out computations on.

    """

    def __init__(
        self, 
        dim: int, 
        num_comp: int = 0,
        fixed_comp: bool = True,
        update_method: str = "augment",
        num_samples_gram: int = 100,
        eps: float = 0.01,
        initial_basis: Tensor | None = None,
        verbose: bool = True,
        device: torch.device = torch.get_default_device()
    ):
        
        if update_method not in UPDATE_METHODS_LIS:
            msg = (
                "Unknown update method. Accepted methods are `"
                f"{"`, `".join(UPDATE_METHODS_LIS)}`."
            )
            raise Exception(msg)

        self._num_comp = num_comp 
        self._fixed_comp = fixed_comp
        self._update_method = update_method
        self._num_samples_gram = num_samples_gram
        self._eps = eps
        self._num_eval = 0
        self._num_eval_grad = 0
        self._initial_basis = initial_basis
        self._verbose = verbose
        self._device = device
        if self._initial_basis is None:
            self._basis_red = torch.zeros((dim, 0), device=self._device)
            self._basis_comp = torch.eye(dim, device=self._device)
        if self._initial_basis is not None:
            self._basis_red = self._initial_basis.clone()
            self._basis_comp = self._compute_basis_comp(self._basis_red)
        if self._fixed_comp and self._num_comp > 0:
            self._compute_samples_comp(self._num_comp)
        self._P_red = self._basis_red @ self._basis_red.T
        self._P_comp = self._basis_comp @ self._basis_comp.T
        return
    
    @property
    def _is_fixed(self) -> bool:
        return False

    def _check_weights(self, weights: Tensor) -> Tensor:
        """Checks a set of importance weights."""
        if ~weights.isnan().any():
            return weights
        msg = "Some weights take NaN values."
        logger.warning(msg)
        return weights.nan_to_num()
    
    def _compute_dim(self, eigvals: Tensor) -> int:
        """Computes the dimension of the updated LIS based on the 
        eigenvalues of the Gram matrix.
        """
        energies = torch.cumsum(eigvals.abs(), dim=0)
        dim_comp = torch.sum(0.5 * torch.sqrt(energies) < self._eps)
        dim_red = self._dim - dim_comp
        return int(dim_red)
    
    def _build_gram(self, grad_neglogliks: Tensor, weights: Tensor) -> Tensor:
        """Computes an importance sampling estimate of the Gram matrix."""
        grad_neglogliks = torch.nan_to_num(grad_neglogliks)
        ws = weights[None, None, :]
        gs_0 = grad_neglogliks.T[:, None, :]
        gs_1 = grad_neglogliks.T[None, :, :]
        gram = (ws * gs_0 * gs_1).sum(dim=2)
        return gram

    def _update_basis_augment(self, eigvals: Tensor, eigvecs: Tensor) -> None:
        """Augments the existing basis with a new set of (orthogonal) 
        vectors.
        """
        dim_aug = self._compute_dim(eigvals)
        if dim_aug < 2 - self._dim_red:
            msg = "Dimension of computed subspace is less than 2. Increasing..."
            logger.info(msg)
            dim_aug = 2 - self._dim_red
        basis_aug = eigvecs.flip(dims=(1,))[:, :dim_aug]
        self._basis_red = torch.hstack((self._basis_red, basis_aug))
        return 

    def _update_basis_rebuild(self, eigvals: Tensor, eigvecs: Tensor) -> None:
        """Computes a new basis from scratch."""
        dim_red = self._compute_dim(eigvals)
        if dim_red < 2:
            msg = "Dimension of computed subspace is less than 2. Increasing..."
            logger.info(msg)
            dim_red = 2
        self._basis_red = eigvecs.flip(dims=(1,))[:, :dim_red]
        return
    
    def _print_diagnostics(self, ess: Tensor) -> None:
        diagnostics = [
            f"Dim: {self._dim_red}", 
            f"ESS: {round(float(ess))}"
        ]
        lis_info(" | ".join(diagnostics).ljust(40))
        return
    
    def _update(
        self, 
        grad_neglogratio: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]],
        reference: GaussianReference
    ) -> None:

        if self._update_method not in UPDATE_METHODS_LIS:
            msg = (
                "Unknown update method provided. "
                + "Accepted values are " 
                + ", ".join(UPDATE_METHODS_LIS) + "."
            )
            raise Exception(msg)

        lis_info("Computing estimate of Gram matrix...", end="\r")

        rs = reference.random(self._num_samples_gram, self._dim, device=self._device)
        neglogref_rs, neglogratios, grad_neglogratios = grad_neglogratio(rs)
        self._num_eval += self._num_samples_gram
        self._num_eval_grad += self._num_samples_gram

        log_weights = neglogref_rs - neglogratios
        log_weights -= log_weights.max()
        weights = log_weights.exp() / log_weights.exp().sum()
        weights = self._check_weights(weights)

        grad_neglogref_rs = reference.eval_potential(rs)[1]
        grad_neglogliks = grad_neglogratios - grad_neglogref_rs

        gram = self._build_gram(grad_neglogliks, weights)
        if self._update_method == "augment":
            gram = self._P_comp @ gram @ self._P_comp 
        eigvals, eigvecs = torch.linalg.eigh(gram)

        if self._update_method == "augment":
            self._update_basis_augment(eigvals, eigvecs)
        elif self._update_method == "rebuild":
            self._update_basis_rebuild(eigvals, eigvecs)
        self._basis_comp = self._compute_basis_comp(self._basis_red)    
        self._P_red = self._basis_red @ self._basis_red.T
        self._P_comp = self._basis_comp @ self._basis_comp.T
        
        if self._fixed_comp and self._num_comp > 0:
            self._compute_samples_comp(self._num_comp)

        if self._verbose:
            ess = estimate_ess_ratio(log_weights) * self._num_samples_gram
            self._print_diagnostics(ess)
        return 
    
    def _eval_neglogprofile(
        self, 
        eval_neglogratio: Callable[[Tensor], Tensor], 
        vs_red: Tensor
    ) -> Tensor:

        xs_red = self._eval_coef2red(vs_red)

        if self._num_comp == 0:
            return eval_neglogratio(xs_red)
        
        num_red = xs_red.shape[0]
        num_comp = self._xs_comp.shape[0]
        if self._fixed_comp:
            xs_comp = self._xs_comp[None, :, :]
        else: 
            xs_comp = self._generate_xs_comp(self._num_comp * num_red)
            xs_comp = xs_comp.reshape(num_red, self._num_comp, self._dim)
        xs = xs_red[:, None, :] + xs_comp
        xs = xs.reshape(-1, self._dim_red + self._dim_comp)
        neglogfxs = eval_neglogratio(xs)
        neglogfxs = neglogfxs.reshape(num_red, num_comp)
        neglogfxs_mean = (
            - torch.logsumexp(-neglogfxs, dim=1)
            + math.log(num_comp)
        )
        return neglogfxs_mean 

    def _clone(self) -> LikelihoodInformedSubspace:
        subspace = LikelihoodInformedSubspace(
            dim=self._dim, 
            num_comp=self._num_comp,
            fixed_comp=self._fixed_comp,
            update_method=self._update_method,
            num_samples_gram=self._num_samples_gram, 
            eps=self._eps, 
            initial_basis=self._basis_red,
            verbose=self._verbose,
            device=self._device
        )
        return subspace