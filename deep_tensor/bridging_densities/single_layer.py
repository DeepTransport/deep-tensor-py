from torch import Tensor

from .bridge import Bridge
from ..references import Reference


class SingleLayer(Bridge):
    r"""Constructs the DIRT using a single layer.
    
    In this setting, the DIRT algorithm reduces to the SIRT algorithm 
    (see Cui and Dolgov, 2022).

    References
    ----------
    Cui, T and Dolgov, S (2022). *[Deep composition of tensor-trains 
    using squared inverse Rosenblatt transports](https://doi.org/10.1007/s10208-021-09537-5).* 
    Foundations of Computational Mathematics, **22**, 1863--1922.

    """

    def __init__(self):
        self.n_layers = 0
        return

    @property 
    def is_last(self) -> bool:
        return True
    
    @property
    def is_adaptive(self) -> bool:
        return False

    @property 
    def n_layers(self) -> int:
        return self._n_layers
    
    @n_layers.setter
    def n_layers(self, value: int) -> None:
        self._n_layers = value
        return
    
    def _get_ratio_func(
        self, 
        reference: Reference, 
        method: str,
        rs: Tensor,
        neglogliks: Tensor, 
        neglogpris: Tensor, 
        neglogfxs: Tensor
    ) -> Tensor:
        neglogratios = neglogliks + neglogpris
        return neglogratios
    
    def _compute_log_weights(
        self, 
        neglogliks: Tensor,
        neglogpris: Tensor,
        neglogfxs: Tensor
    ) -> Tensor:
        """Returns the logarithm of the ratio between the current 
        bridging density and the density of the approximation to the 
        previous bridging density evaluated at each of a set of samples
        distributed according to the previous bridging density.

        Parameters
        ----------
        neglogliks:
            An n-dimensional vector containing the negative 
            log-likelihood function evaluated at each sample.
        neglogpris:
            An n-dimensional vector containing the negative log-prior 
            density evaluated at each sample.
        neglogfxs:
            An n-dimensional vector containing the negative logarithm 
            of the approximation to the previous bridging density 
            evaluated at each sample.

        Returns
        -------
        log_weights:
            The logarithm of the ratio between the current bridging 
            density and the density of the approximation to the 
            previous bridging density evaluated at each sample.
        
        """
        log_weights = -neglogliks - neglogpris + neglogfxs
        return log_weights