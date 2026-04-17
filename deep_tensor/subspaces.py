import abc
import logging
import math
from typing import Callable, Tuple
import warnings

import torch 
from torch import Tensor

from .tools.printing import lis_info
from .debiasing.importance_sampling import estimate_ess_ratio


logger = logging.getLogger(__name__)


TARGET_FUNCS_LIS = ("bridge", "ratio")
UPDATE_METHODS_LIS = ("augment", "rebuild", "fixed")


class Subspace(abc.ABC):

    @property 
    @abc.abstractmethod 
    def is_fixed(self) -> bool:
        pass
    
    @property 
    def basis_red(self) -> Tensor:
        return self._basis_red 
    
    @basis_red.setter
    def basis_red(self, val: Tensor) -> None:
        self._basis_red = val 
        return
    
    @property 
    def basis_comp(self) -> Tensor:
        return self._basis_comp
    
    @basis_comp.setter
    def basis_comp(self, val: Tensor) -> None:
        self._basis_comp = val 
        return
    
    @property 
    def num_eval(self) -> int:
        return self._num_eval
    
    @num_eval.setter
    def num_eval(self, val: int) -> None:
        self._num_eval = val 
        return
    
    @property 
    def num_eval_grad(self) -> int:
        return self._num_eval_grad
    
    @num_eval_grad.setter
    def num_eval_grad(self, val: int) -> None:
        self._num_eval_grad = val 
        return
    
    @property 
    def dim(self) -> int:
        return self.dim_red + self.dim_comp

    @property
    def dim_red(self) -> int:
        return self.basis_red.shape[1]

    @property 
    def dim_comp(self) -> int:
        return self.basis_comp.shape[1]
    
    @property 
    def P_red(self) -> Tensor:
        return self._P_red
    
    @P_red.setter
    def P_red(self, val: Tensor) -> None:
        self._P_red = val 
        return
    
    @property 
    def P_comp(self) -> Tensor:
        return self._P_comp
    
    @P_comp.setter
    def P_comp(self, val: Tensor) -> None:
        self._P_comp = val 
        return

    def eval_coef2red(self, vs: Tensor) -> Tensor:
        """Computes the reduced subspace vectors associated with a 
        set of coefficients.
        """
        vs = torch.atleast_2d(vs)
        return vs @ self.basis_red.T 
    
    def eval_red2coef(self, xs: Tensor) -> Tensor:
        """Computes the reduced subspace coefficients associated with a 
        set of vectors.
        """
        xs = torch.atleast_2d(xs)
        return xs @ self.basis_red
    
    def eval_coef2comp(self, ws: Tensor) -> Tensor:
        """Computes the complement subspace vectors associated with a 
        set of coefficients.
        """
        ws = torch.atleast_2d(ws)
        return ws @ self.basis_comp.T
    
    def eval_comp2coef(self, xs: Tensor) -> Tensor:
        """Computes the complement subspace coefficients associated 
        with a set of vectors.
        """
        xs = torch.atleast_2d(xs)
        return xs @ self.basis_comp
    
    def project_red(self, xs: Tensor) -> Tensor:
        """Projects a set of vectors onto the LDT subspace."""
        xs = torch.atleast_2d(xs)
        return xs @ self.P_red
    
    def project_comp(self, xs: Tensor) -> Tensor:
        """Projects a set of vectors onto the complement subspace."""
        xs = torch.atleast_2d(xs)
        return xs @ self.P_comp
    
    def _compute_basis_comp(self, basis_red: Tensor) -> Tensor:
        """Given a basis for the reduced subspace, computes a basis for 
        the complement subspace.
        """
        P_comp = torch.eye(basis_red.shape[0]) - basis_red @ basis_red.T
        _, eigvecs = torch.linalg.eigh(P_comp)
        basis_comp = eigvecs[:, self.dim_red:]
        return basis_comp

    @abc.abstractmethod 
    def eval_neglogprofile(
        self, 
        eval_neglogtarget: Callable[[Tensor], Tensor],
        vs_red: Tensor
    ) -> Tensor:
        r"""Evalutes the negative logarithm of the profile function at a 
        set of points in the reduced subspace.
        
        Parameters
        ----------
        target_func:
            A function that accepts an $n \times d$ matrix containing a 
            set of samples in the reference domain, and returns an 
            $n$-dimensional vector containing the negative logarithm of 
            the target function (composed with the current IRT mapping) 
            evaluated at each sample.
        vs_red:
            An $n \times d_{r}$ matrix containing the coefficients 
            associated with a set of samples in the reduced subspace.

        Returns
        -------
        neglogprofiles:
            An $n$-dimensional vector containing the negative 
            logarithm of the profile function evaluated at each of the 
            samples in `vs_red`.

        """
        pass

    @abc.abstractmethod 
    def update(
        self,
        grad_neglogbridge: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]],
        grad_neglogratio: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        r"""Updates the basis associated with the current reduced subspace.
        
        Parameters
        ----------
        grad_neglogbridge:
            A function that accepts an $n \times d$ matrix containing a 
            set of samples in the approximation domain, and return an
            $n$-dimensional vector containing the negative logarithm of 
            the target function evaluated at each sample, and an 
            $n \times d$ matrix containing the gradient of the negative 
            logarithm of the target function evaluated at each sample.
        grad_neglogratio:
            TODO: write this.
        
        """
        pass

    @abc.abstractmethod 
    def clone(self) -> Subspace:
        """Returns a copy of the subspace."""
        pass


class IdentitySubspace(Subspace):
    r"""Identity subspace (*i.e.,* the DIRT will be constructed on the full space).
    
    Parameters
    ----------
    dim:
        The dimension of the target density.
    
    """

    def __init__(self, dim: int):
        self.basis_red = torch.eye(dim)
        self.basis_comp = torch.zeros((dim, 0))
        self.num_eval = 0
        self.num_eval_grad = 0
        return
    
    @property 
    def is_fixed(self) -> bool:
        return True
    
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


class LikelihoodInformedSubspace(Subspace):
    """A likelihood-informed subspace.

    Parameters
    ----------
    target_func:
        Whether to estimate the Gram matrix associated with the 
        bridging density or ratio function at each iteration.
    num_comp:
        The number of samples from the complement subspace to use when 
        evaluating the profile function.
    fixed_comp:
        Whether to fix the samples from the complement subspace.
    update_method:
        How to update the subspace ("augment", "rebuild", "static").
    num_samples_gram:
        The number of samples to use to construct a Monte Carlo 
        estimate of the Gram matrix.
    eps:
        TODO: write down the inequality with eps used to select 
        dimension of subspace.
    initial_basis:
        A set of basis vectors to initialise the subspace with.

    """

    def __init__(
        self, 
        dim: int, 
        target_func: str = "bridge",
        num_comp: int = 0,
        fixed_comp: bool = True,
        update_method: str = "augment",
        num_samples_gram: int = 100,
        eps: float = 0.01,
        initial_basis: Tensor | None = None
    ):

        target_func = target_func.lower()
        update_method = update_method.lower()
        if target_func not in TARGET_FUNCS_LIS:
            msg = (
                "Unknown target function. Accepted target functions are `"
                f"{"`, `".join(TARGET_FUNCS_LIS)}`."
            )
            raise Exception(msg)
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

        self.target_func = target_func 
        self.num_comp = num_comp 
        self.fixed_comp = fixed_comp
        self.update_method = update_method
        self.num_samples_gram = num_samples_gram
        self.eps = eps
        self.num_eval = 0
        self.num_eval_grad = 0
        self.initial_basis = initial_basis

        if self.initial_basis is None:
            # TODO: fix device here..
            self.basis_red = torch.zeros((dim, 0))
            self.basis_comp = torch.eye(dim)
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
        # TODO: fix device.
        self.vs_comp = torch.randn((self.num_comp, self.dim_comp))
        self.xs_comp = self.eval_coef2comp(self.vs_comp)
        return
    
    def _print_diagnostics(self, ess: Tensor) -> None:
        diagnostics = [f"Dim: {self.dim_red}", f"ESS: {round(float(ess))}"]
        lis_info(" | ".join(diagnostics).ljust(40))
        return

    def _update_augment(
        self, 
        grad_neglogbridge: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]]
    ) -> None:
        
        # Generate a set of samples distributed according to biasing 
        # density. TODO: should these be generated according to reference?
        # TODO: fix device.
        rs = torch.randn((self.num_samples_gram, self.dim))
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
        
        # TODO: if an initial subspace gets passed in, should it be 
        # retained every time the subspace gets rebuilt? alternative 
        # is to not allow passing in an initial subspace if the method 
        # is `rebuild`.

        # Generate a set of samples distributed according to biasing 
        # density. TODO: should these be generated according to reference?
        # TODO: fix device.
        rs = torch.randn((self.num_samples_gram, self.dim))
        neglogfus, neglogratios, grad_neglogratios = grad_neglogratio(rs)

        self.num_eval += rs.shape[0]
        self.num_eval_grad += rs.shape[0]

        neglogref_rs = 0.5 * rs.square().sum(dim=1)

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
            target_func=self.target_func, 
            num_comp=self.num_comp,
            fixed_comp=self.fixed_comp,
            update_method=self.update_method,
            num_samples_gram=self.num_samples_gram, 
            eps=self.eps, 
            initial_basis=self.basis_red
        )
        return subspace