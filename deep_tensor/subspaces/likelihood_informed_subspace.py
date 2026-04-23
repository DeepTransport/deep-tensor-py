import logging
import math
from typing import Callable, Tuple
import warnings

import torch 
from torch import Tensor

from .subspace import Subspace
from ..debiasing.importance_sampling import estimate_ess_ratio
from ..tools.printing import lis_info


logger = logging.getLogger(__name__)

UPDATE_METHODS_LIS = ("fixed", "rebuild", "augment")


class LikelihoodInformedSubspace(Subspace):
    r"""A likelihood-informed subspace.

    Parameters
    ----------
    num_comp:
        The number of samples from the complement subspace to use when 
        evaluating the profile function.
    fixed_comp:
        Whether to fix the samples from the complement subspace.
    update_method:
        How to update the subspace. This can be `'fixed'` (no updating, 
        need to specify `initial_basis`), `'rebuild'` (construct a new 
        subspace from scratch at each DIRT layer), or `'augment'` 
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
        A set of basis vectors to initialise the subspace with.
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
        device: torch.device = torch.get_default_device()
    ):
        
        if update_method not in UPDATE_METHODS_LIS:
            msg = (
                "Unknown update method. Accepted methods are `"
                f"{"`, `".join(UPDATE_METHODS_LIS)}`."
            )
            raise Exception(msg)
        if update_method == "fixed" and initial_basis is None:
            msg = (
                "If update_method==`fixed`, an initial basis must be "
                "supplied."
            )
            raise Exception(msg)
        if update_method == "rebuild" and initial_basis is not None:
            msg = (
                "If update_method==`rebuild`, the initial basis is not "
                "used. To start from an initial subspace, use "
                "update_method==`augment`."
            )
            warnings.warn(msg)

        self.num_comp = num_comp 
        self.fixed_comp = fixed_comp
        self.update_method = update_method
        self.num_samples_gram = num_samples_gram
        self.eps = eps
        self.num_eval = 0
        self.num_eval_grad = 0
        self.initial_basis = initial_basis
        self.device = device
        if self.initial_basis is None:
            self.basis_red = torch.zeros((dim, 0), device=self.device)
            self.basis_comp = torch.eye(dim, device=self.device)
        if self.initial_basis is not None:
            self.basis_red = self.initial_basis.clone()
            self.basis_comp = self._compute_basis_comp(self.basis_red)
            if self.fixed_comp and self.num_comp > 0:
                self._recompute_samples_comp()
        self.P_red = self.basis_red @ self.basis_red.T
        self.P_comp = self.basis_comp @ self.basis_comp.T

        return
    
    @property
    def is_fixed(self) -> bool:
        return self.update_method == "fixed"

    def _check_weights(self, weights: Tensor) -> None:
        """Checks a set of importance weights."""
        # TODO: should also check the gradients for nans, before and 
        # after taking the reference off..
        if weights.isnan().any():
            msg = "Some weights take NaN values."
            logger.warning(msg)
        return
    
    def _compute_dim(self, eigvals: Tensor) -> int:
        """Computes the dimension of the updated LIS based on the 
        eigenvalues of the Gram matrix.
        """
        energies = torch.cumsum(eigvals.abs(), dim=0)
        dim_comp = torch.sum(0.5 * torch.sqrt(energies) < self.eps)
        dim_red = self.dim - dim_comp
        return int(dim_red)
    
    def _build_H(self, grads: Tensor, weights: Tensor) -> Tensor:
        """Computes an importance sampling estimate of the Gram matrix."""
        grads = torch.nan_to_num(grads)
        H = torch.zeros((self.dim, self.dim))
        for grad, weight in zip(grads, weights):
            H += weight * grad[:, None] @ grad[None, :]
        return H
    
    def _recompute_samples_comp(self) -> None:
        """Re-computes the (fixed) set of samples in the complement 
        subspace.
        """
        shape_vs_comp = (self.num_comp, self.dim_comp)
        self.vs_comp = torch.randn(shape_vs_comp, device=self.device)
        self.xs_comp = self.eval_coef2comp(self.vs_comp)
        return
    
    def _print_diagnostics(self, ess: Tensor) -> None:
        diagnostics = [
            f"Dim: {self.dim_red}", 
            f"ESS: {round(float(ess))}"
        ]
        lis_info(" | ".join(diagnostics).ljust(40))
        return

    def _update_augment(
        self, 
        grad_neglogbridge: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        
        # Generate a set of samples distributed according to biasing 
        # density. TODO: should these be generated according to reference?
        shape_rs = (self.num_samples_gram, self.dim)
        rs = torch.randn(shape_rs, device=self.device)
        neglogfus, neglogbridges, grad_neglogbridges = grad_neglogbridge(rs)

        self.num_eval += rs.shape[0]
        self.num_eval_grad += rs.shape[0]

        log_weights = neglogfus - neglogbridges
        log_weights -= log_weights.max()
        weights = log_weights.exp() / log_weights.exp().sum()
        self._check_weights(weights)
        ess = estimate_ess_ratio(log_weights) * weights.numel()

        # Subtract the contribution of the standard Gaussian to the 
        # gradient of the bridging density
        grad_neglogref_us = rs.clone()
        grad_neglogliks = grad_neglogbridges - grad_neglogref_us

        H = self._build_H(grad_neglogliks, weights)
        H_comp = self.P_comp @ H @ self.P_comp
        eigvals, eigvecs = torch.linalg.eigh(H_comp)
        dim_red = self._compute_dim(eigvals)

        # Update basis and projection operators
        basis_up = eigvecs.flip(dims=(1,))[:, :dim_red]
        self.basis_red = torch.hstack((self.basis_red, basis_up))
        self.basis_comp = self._compute_basis_comp(self.basis_red)
        self.P_red = self.basis_red @ self.basis_red.T
        self.P_comp = self.basis_comp @ self.basis_comp.T

        self._print_diagnostics(ess)
        return
    
    def _update_rebuild(
        self, 
        grad_neglogratio: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]]
    ) -> None:

        # Generate a set of samples distributed according to biasing 
        # density. TODO: should these be generated according to reference?
        # TODO: it might be a good idea to pass the reference into these 
        # functions..
        rs = torch.randn((self.num_samples_gram, self.dim), device=self.device)
        neglogref_rs, neglogratios, grad_neglogratios = grad_neglogratio(rs)

        self.num_eval += rs.shape[0]
        self.num_eval_grad += rs.shape[0]

        log_weights = neglogref_rs - neglogratios
        log_weights -= log_weights.max()
        weights = log_weights.exp() / log_weights.exp().sum()
        self._check_weights(weights)
        ess = estimate_ess_ratio(log_weights) * weights.numel()

        # Subtract the contribution of the standard Gaussian to the 
        # gradient of the ratio
        # self.reference.eval_potential()
        grad_neglogref_rs = rs.clone()
        grad_neglogliks = grad_neglogratios - grad_neglogref_rs

        H = self._build_H(grad_neglogliks, weights)
        eigvals, eigvecs = torch.linalg.eigh(H)
        dim_red = self._compute_dim(eigvals)
        dim_red = max(dim_red, 2)

        # Update basis and projection operators
        self.basis_red = eigvecs.flip(dims=(1,))[:, :dim_red]
        self.basis_comp = self._compute_basis_comp(self.basis_red)
        self.P_red = self.basis_red @ self.basis_red.T
        self.P_comp = self.basis_comp @ self.basis_comp.T

        self._print_diagnostics(ess)
        return
    
    def update(
        self, 
        grad_neglogbridge: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]],
        grad_neglogratio: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]]
    ) -> None:

        if self.update_method == "fixed":
            return
        
        lis_info("Computing estimate of Gram matrix...", end="\r")

        if self.update_method == "augment":
            self._update_augment(grad_neglogbridge)
        elif self.update_method == "rebuild":
            self._update_rebuild(grad_neglogratio)
        else:
            msg = "Unknown update method provided."
            raise Exception(msg)

        # Estimate some errors
        # self.error_acc = torch.trace(self.P_comp @ H @ self.P_comp)
        # eigvals, _ = torch.linalg.eigh(H)
        # self.error_new = torch.sum(eigvals[:self.dim_comp])
        if self.fixed_comp and self.num_comp > 0:
            self._recompute_samples_comp()
        return 
    
    def eval_neglogprofile(
        self, 
        eval_neglogratio: Callable[[Tensor], Tensor], 
        vs_red: Tensor
    ) -> Tensor:

        xs_red = self.eval_coef2red(vs_red)

        if self.num_comp == 0:
            return eval_neglogratio(xs_red)
        
        if self.vs_comp is None:
            # Generate a new set of samples in the complement subspace
            self._recompute_samples_comp()
        
        num_red = xs_red.shape[0]
        num_comp = self.xs_comp.shape[0]
        xs = xs_red[:, None, :] + self.xs_comp[None, :, :]
        xs = xs.reshape(-1, self.dim_red + self.dim_comp)
        neglogfxs = eval_neglogratio(xs)
        neglogfxs = neglogfxs.reshape(num_red, num_comp)
        neglogfxs_mean = (
            - torch.logsumexp(-neglogfxs, dim=1)
            + math.log(num_comp)
        )
        return neglogfxs_mean 

    def clone(self) -> LikelihoodInformedSubspace:
        subspace = LikelihoodInformedSubspace(
            dim=self.dim, 
            num_comp=self.num_comp,
            fixed_comp=self.fixed_comp,
            update_method=self.update_method,
            num_samples_gram=self.num_samples_gram, 
            eps=self.eps, 
            initial_basis=self.basis_red
        )
        return subspace