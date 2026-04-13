import abc
from typing import List, Tuple

import torch
from torch import Tensor

from ..preconditioners import Preconditioner
from ..target_functions import TargetFunc


class Bridge(abc.ABC):
    
    @property
    @abc.abstractmethod
    def is_last(self) -> bool:
        pass

    @property 
    def num_layers(self) -> int:
        return self._num_layers
    
    @num_layers.setter
    def num_layers(self, value: int) -> None:
        self._num_layers = value
        return
    
    @property
    def is_adaptive(self) -> bool:
        return self._is_adaptive
    
    @is_adaptive.setter 
    def is_adaptive(self, value: bool) -> None:
        self._is_adaptive = value 
        return

    @abc.abstractmethod
    def ratio_func(
        self, 
        method: str,
        rs: Tensor,
        us: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:
        pass

    @abc.abstractmethod
    def update(self, us: Tensor, neglogfus_dirt: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates the current bridging density, the next ratio 
        function and the ratio between the current bridging density and 
        the next bridging density at each of a set of samples.

        Parameters
        ----------
        us:
            An n * d matrix containing the samples from `rs` after 
            applying the current DIRT mapping to them.
        neglogfus_dirt:
            An n-dimensional vector containing evaluations of the 
            current DIRT density at each sample in us.  
        
        Returns
        -------
        log_weights:
            An n-dimensional vector containing the ratio between 
            the current and new bridging densities evaluated at each 
            sample.
        neglogbridges:
            An n-dimensional vector containing the current bridging 
            density evaluated at each sample.
        
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the parameters of the bridging density to those at 
        initialisation.
        """
        pass

    @abc.abstractmethod
    def eval_gradneglog(self, us: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates the current bridging density and its gradient at a 
        set of samples.
        """
        pass

    def initialise(
        self, 
        preconditioner: Preconditioner, 
        target_func: TargetFunc
    ) -> None:
        self.reset()
        self.preconditioner = preconditioner
        self.reference = self.preconditioner.reference
        self.target_func = target_func
        return
    
    def _check_grad(self) -> None:
        """Throws an error if no gradients are supplied for the target 
        function.
        """
        if not self.target_func._has_grad:
            msg = "Gradients of the target function have not been supplied."
            raise Exception(msg)
        return
    
    def _grad_x2u(self, dfdxs: Tensor, dxdus: Tensor) -> Tensor:
        """Converts a set of gradients in terms of x to gradients in 
        terms of u.

        Parameters
        ----------
        dfdxs:
            An n * d matrix containing evaluations of the gradient of 
            function f with respect to x.
        dxdus:
            A d * n * d tensor where each slice in the second dimension 
            is an evaluation of the gradient of x with respect to u.

        Returns
        -------
        dfdus:
            An n * d matrix containing the corresponding evaluations of 
            the gradient of function f with respect to u.
        
        """
        dfdus = torch.einsum("...i, i...j", dfdxs, dxdus)
        return dfdus
    
    def _eval_pullback(self, us: Tensor) -> Tensor:
        """Evaluates the pullback of the target density under the 
        preconditioning mapping.
        """
        xs, neglogdets = self.preconditioner.Q(us)
        neglogfxs = self.target_func(xs)
        neglogfus = neglogfxs + neglogdets
        return neglogfus
    
    def _eval_pullback_grad(self, us: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates the pullback of the target density under the 
        preconditioning mapping, and its gradient.
        """
        xs, neglogdets, dxdus = self.preconditioner.grad_Q(us)
        neglogfxs, grad_neglogfxs = self.target_func.grad_func(xs) # NOTE: this may not work with a RareEventFunc currently.. maybe it should
        neglogfus = neglogfxs + neglogdets
        grad_neglogfus = self._grad_x2u(grad_neglogfxs, dxdus)
        return neglogfus, grad_neglogfus
    
    def _reorder(
        self, 
        xs: Tensor, 
        neglogratios: Tensor,
        log_weights: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Returns a reordered set of indices based on the importance
        weights between the current bridging density and the density 
        of the approximation to the previous target density evaluated
        at a set of samples from the approximation to the previous 
        target density.

        Parameters
        ----------
        xs:
            An n * d matrix containing a set of samples distributed 
            according to the approximation to the previous target 
            density.
        neglogratios:
            An n-dimensional vector containing the negative logarithm 
            of the ratio function evaluated at each sample in xs.
        log_weights:
            An n-dimensional vector containing the logarithm of the 
            ratio between the current bridging density and the density 
            of the approximation to the previous target density 
            evaluated at each sample in xs.

        Returns
        -------
        xs:
            An n * d matrix containing the reordered samples.
        neglogratios:
            An n-dimensional vector containing the negative logarithm 
            of the ratio function evaluated at each sample in xs.

        """
        reordered_inds = torch.argsort(log_weights).flip(dims=(0,))
        xs = xs[reordered_inds]
        neglogratios = neglogratios[reordered_inds]
        return xs, neglogratios

    def _get_diagnostics(
        self, 
        log_weights: Tensor | None,
        neglogfus: Tensor | None,
        neglogfus_dirt: Tensor | None
    ) -> List[str]:
        """Returns some information about the current bridging density.

        Parameters
        ----------
        log_weights:
            An n-dimensional vector containing the logarithm of the 
            ratio between the next bridging density and the current 
            bridging density evaluated at a set of samples from the 
            DIRT approximation to the current bridging density.
        neglogfus:
            An n-dimensional vector containing the negative logarithm 
            of the current bridging density evaluated at a set of 
            samples from the DIRT approximation.
        neglogfus_dirt:
            An-dimensional vector containin the negative logarithm of 
            the DIRT approximation to the current bridging density 
            evaluated at the same set of samples as above. 

        """
        return []