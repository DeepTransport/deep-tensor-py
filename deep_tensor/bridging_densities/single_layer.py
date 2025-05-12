from torch import Tensor

from .bridge import Bridge


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
        method: str,
        neglogrefs_rs: Tensor,
        neglogrefs: Tensor, 
        neglogfxs: Tensor, 
        neglogfxs_dirt: Tensor
    ) -> Tensor:
        neglogratios = neglogfxs.clone()
        return neglogratios
    
    def _compute_log_weights(
        self, 
        neglogrefs: Tensor,
        neglogfxs: Tensor,
        neglogfxs_dirt: Tensor
    ) -> Tensor:
        
        log_weights = -neglogfxs + neglogfxs_dirt
        return log_weights