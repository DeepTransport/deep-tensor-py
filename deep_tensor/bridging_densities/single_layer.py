from typing import Dict

from torch import Tensor

from .bridge import Bridge


class AbstractSingleLayer(Bridge):

    @property 
    def is_last(self) -> bool:
        return True
    
    @property
    def params_dict(self) -> Dict:
        return {"n_layers": self.n_layers}


class SingleLayer(AbstractSingleLayer):
    r"""Constructs the DIRT using a single layer.
    
    In this setting, the DIRT algorithm reduces to the SIRT algorithm 
    (see Cui and Dolgov, 2022).

    References
    ----------
    Cui, T and Dolgov, S (2022). *[Deep composition of tensor-trains 
    using squared inverse Rosenblatt transports](https://doi.org/10.1007/s10208-021-09537-5).* 
    Foundations of Computational Mathematics **22**, 1863--1922.

    """

    def __init__(self):
        self.n_layers = 0
        self.is_adaptive = False
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


class SavedSingleLayer(AbstractSingleLayer):

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        return
    
    def _compute_log_weights(self, neglogliks, neglogpris, neglogfxs):
        raise NotImplementedError()
    
    def _get_ratio_func(self, reference, method, rs, neglogliks, neglogpris, neglogfxs):
        raise NotImplementedError()