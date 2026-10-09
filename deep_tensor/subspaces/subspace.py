from __future__ import annotations

import abc 
from typing import Callable, Tuple

import torch
from torch import Tensor 

from ..references import Reference


class Subspace(abc.ABC):
    _basis_red: Tensor
    _basis_comp: Tensor
    _P_red: Tensor
    _P_comp: Tensor
    _num_comp: int
    _num_eval: int
    _num_eval_grad: int
    _device: torch.device

    @property 
    @abc.abstractmethod 
    def _is_fixed(self) -> bool:
        pass
    
    @property 
    def _dim(self) -> int:
        return self._dim_red + self._dim_comp

    @property
    def _dim_red(self) -> int:
        return self._basis_red.shape[1]

    @property 
    def _dim_comp(self) -> int:
        return self._basis_comp.shape[1]
    
    def _compute_basis_comp(self, basis_red: Tensor) -> Tensor:
        """Given a basis for the reduced subspace, computes a basis for 
        the complement subspace.
        """
        P_comp = torch.eye(basis_red.shape[0]) - basis_red @ basis_red.T
        _, eigvecs = torch.linalg.eigh(P_comp)
        basis_comp = eigvecs[:, self._dim_red:]
        return basis_comp
    
    def _compute_samples_comp(self, num_comp: int) -> None:
        """Computes a (fixed) set of samples in the complement subspace."""
        shape_vs_comp = (num_comp, self._dim_comp)
        self._vs_comp = torch.randn(shape_vs_comp, device=self._device)
        self._xs_comp = self._eval_coef2comp(self._vs_comp)
        return
    
    def _generate_xs_comp(self, num_samples: int) -> Tensor:
        """Generates a set of samples in the complement subspace with 
        the appropriate dimension.
        """
        shape_comp = (num_samples, self._dim_comp)
        vs_comp = torch.randn(shape_comp, device=self._device)
        xs_comp = self._eval_coef2comp(vs_comp)
        return xs_comp

    def _eval_coef2red(self, vs: Tensor) -> Tensor:
        """Computes the reduced subspace vectors associated with a 
        set of coefficients.
        """
        vs = torch.atleast_2d(vs)
        return vs @ self._basis_red.T 
    
    def _eval_red2coef(self, xs: Tensor) -> Tensor:
        """Computes the reduced subspace coefficients associated with a 
        set of vectors.
        """
        xs = torch.atleast_2d(xs)
        return xs @ self._basis_red
    
    def _eval_coef2comp(self, ws: Tensor) -> Tensor:
        """Computes the complement subspace vectors associated with a 
        set of coefficients.
        """
        ws = torch.atleast_2d(ws)
        return ws @ self._basis_comp.T
    
    def _eval_comp2coef(self, xs: Tensor) -> Tensor:
        """Computes the complement subspace coefficients associated 
        with a set of vectors.
        """
        xs = torch.atleast_2d(xs)
        return xs @ self._basis_comp
    
    def _project_red(self, xs: Tensor) -> Tensor:
        """Projects a set of vectors onto the LDT subspace."""
        xs = torch.atleast_2d(xs)
        return xs @ self._P_red
    
    def _project_comp(self, xs: Tensor) -> Tensor:
        """Projects a set of vectors onto the complement subspace."""
        xs = torch.atleast_2d(xs)
        return xs @ self._P_comp

    @abc.abstractmethod 
    def _eval_neglogprofile(
        self, 
        eval_neglogtarget: Callable[[Tensor], Tensor],
        vs_red: Tensor
    ) -> Tensor:
        r"""Evaluates the negative logarithm of the profile function at 
        a set of points in the reduced subspace.
        
        Parameters
        ----------
        eval_neglogtarget:
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
    def _update(
        self,
        grad_neglogratio: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]],
        reference: Reference
    ) -> None:
        """Updates the basis associated with the current reduced 
        subspace.
        """
        pass

    @abc.abstractmethod 
    def _clone(self) -> Subspace:
        """Returns a copy of the subspace."""
        pass