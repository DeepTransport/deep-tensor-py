from __future__ import annotations

import abc 
from typing import Callable, Tuple

import torch
from torch import Tensor 


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
    def num_comp(self) -> int:
        return self._num_comp
    
    @num_comp.setter
    def num_comp(self, val: int) -> None:
        self._num_comp = val 
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
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @device.setter 
    def device(self, val: torch.device) -> None:
        self._device = val 
        return
    
    def _compute_basis_comp(self, basis_red: Tensor) -> Tensor:
        """Given a basis for the reduced subspace, computes a basis for 
        the complement subspace.
        """
        P_comp = torch.eye(basis_red.shape[0]) - basis_red @ basis_red.T
        _, eigvecs = torch.linalg.eigh(P_comp)
        basis_comp = eigvecs[:, self.dim_red:]
        return basis_comp
    
    def _compute_samples_comp(self, num_comp: int) -> None:
        """Computes a (fixed) set of samples in the complement subspace."""
        shape_vs_comp = (num_comp, self.dim_comp)
        self.vs_comp = torch.randn(shape_vs_comp, device=self.device)
        self.xs_comp = self.eval_coef2comp(self.vs_comp)
        return
    
    def _generate_xs_comp(self, num_samples: int) -> Tensor:
        """Generates a set of samples in the complement subspace with 
        the appropriate dimension.
        """
        shape_comp = (num_samples, self.dim_comp)
        vs_comp = torch.randn(shape_comp, device=self.device)
        xs_comp = self.eval_coef2comp(vs_comp)
        return xs_comp

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
        """Updates the basis associated with the current reduced 
        subspace.
        """
        pass

    @abc.abstractmethod 
    def clone(self) -> Subspace:
        """Returns a copy of the subspace."""
        pass