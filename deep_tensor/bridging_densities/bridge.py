import abc
from typing import List, Tuple

import torch
from torch import Tensor

from ..preconditioners import Preconditioner
from ..target_functions import TargetFunc


class Bridge(abc.ABC):
    """Base class for all bridges. Generates a set of bridging 
    densities to construct a DIRT approximation to.
    """

    _num_layers: int
    _is_adaptive: bool

    @property
    @abc.abstractmethod
    def _is_last(self) -> bool:
        pass

    @abc.abstractmethod
    def _eval_neglogratio(
        self, 
        method: str,
        rs: Tensor,
        us: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:
        """TODO: write docstring for this.."""
        pass

    @abc.abstractmethod 
    def _grad_neglogratio(
        self,
        method: str,
        rs: Tensor,
        us: Tensor,
        neglogfus_dirt: Tensor,
        grad_neglogfus_dirt: Tensor | None, 
        dudrs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Evaluates the gradient of the negative logarithm of the 
        current ratio function.
        
        Parameters
        ----------
        method:
            The ratio function to compute ('aratio' or 'eratio').
        rs:
            An n * d matrix containing a set of samples in the 
            reference domain.
        us:
            An n * d matrix containing the corresponding samples after 
            applying the IRT (without the preconditioner).
        neglogfus_dirt:
            An n-dimensional vector containing the pushforward of the 
            reference density under the IRT mapping evaluated at each 
            element in 'us'.
        grad_neglogfus_dirt:
            An n * d matrix where each row contains the gradient of the 
            negative logarithm of the pushforward of the reference 
            density under the IRT mapping, evaluated at each element in 
            'us'.
        dudrs:
            An d * n * d tensor, where `dudrs[:, i, :]` contains the 
            Jacobian of the DIRT mapping evaluated for `us[i, :]`.
        
        Returns
        -------
        neglogratios:
            An n-dimensional vector containing the negative logarithm 
            of the current ratio function evaluated at each element in 
            `us`.
        grad_neglogratios:
            An n * d matrix containing the gradient of the composition 
            of the current IRT and ratio function evaluated at each 
            element in `rs`.
            
        """
        pass

    @abc.abstractmethod
    def _grad_neglogbridge(
        self, 
        us: Tensor,
        dudrs: Tensor 
    ) -> Tuple[Tensor, Tensor]:
        """Evaluates the current bridging density and its gradient at a 
        set of samples.

        Parameters
        ----------
        us:
            An n * d matrix containing a set of samples from the 
            reference domain after applying the IRT (without the 
            preconditioning mapping).
        dudrs:
            An d * n * d matrix. `dudrs[:, i, :]` contains the gradient 
            of the IRT mapping evaluated at `us[i, :]`.

        Returns
        -------
        neglogbridges:
            An n-dimensional vector containing the negative logarithm 
            of the current bridging density (pulled back under the 
            preconditioner) evaluated at each sample in `us`.
        grad_neglogbridges:
            An n * d matrix, where the rows contain evaluations of the 
            gradient of the negative logarithm of the composition of 
            the IRT and the current bridging density evaluated at each 
            sample in `us`.
        
        """
        pass

    @abc.abstractmethod
    def _update(self, us: Tensor, neglogfus_dirt: Tensor) -> Tuple[Tensor, Tensor]:
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
            An n-dimensional vector containing the logarithm of the 
            ratio between the current and new bridging densities 
            evaluated at each sample.
        neglogbridges:
            An n-dimensional vector containing the negative logarithm 
            of the current bridging density evaluated at each sample.
        
        """
        pass

    @abc.abstractmethod
    def _reset(self) -> None:
        """Resets the parameters of the bridging density to those at 
        initialisation.
        """
        pass

    def _initialise(
        self, 
        preconditioner: Preconditioner, 
        target_func: TargetFunc
    ) -> None:
        self._reset()
        self._preconditioner = preconditioner
        self._reference = self._preconditioner.reference
        self._target_func = target_func
        return
    
    def _check_grad(self) -> None:
        """Throws an error if no gradients are supplied for the target 
        function.
        """
        if not self._target_func._has_grad:
            msg = "Gradients of the target function have not been supplied."
            raise Exception(msg)
        return
    
    def _grad_chain(self, dfdxs: Tensor, dxdus: Tensor) -> Tensor:
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
        dfdus = torch.einsum("j...i, ...j", dxdus, dfdxs)
        return dfdus
    
    def _eval_pullback(self, us: Tensor) -> Tensor:
        """Evaluates the pullback of the target density under the 
        preconditioning mapping.
        """
        xs, neglogdets = self._preconditioner.Q(us)
        neglogfxs = self._target_func(xs)
        neglogfus = neglogfxs + neglogdets
        return neglogfus
    
    def _grad_pullback(self, us: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates the pullback of the target density under the 
        preconditioning mapping, and its gradient.
        """
        xs, neglogdets, dxdus = self._preconditioner.grad_Q(us)
        neglogfxs, grad_neglogfxs = self._target_func._grad_func(xs)
        neglogfus = neglogfxs + neglogdets
        grad_neglogfus = self._grad_chain(grad_neglogfxs, dxdus)
        return neglogfus, grad_neglogfus
    
    def _reorder(
        self, 
        xs: Tensor, 
        neglogratios: Tensor,
        log_weights: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Reorders a set of samples based on the importance weights 
        between the current bridging density and the density of the 
        approximation to the previous target density evaluated at a set 
        of samples from the approximation to the previous target 
        density.

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
            An n-dimensional vector containing the negative logarithm 
            of the DIRT approximation to the current bridging density 
            evaluated at the same set of samples as above. 

        """
        return []