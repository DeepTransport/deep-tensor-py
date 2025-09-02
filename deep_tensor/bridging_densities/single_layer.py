from typing import Tuple

from torch import Tensor

from .bridge import Bridge


class SingleLayer(Bridge):
    r"""Constructs the DIRT using a single layer.
    
    In this setting, the DIRT algorithm reduces to the SIRT algorithm; 
    see @Cui2022.

    """

    def __init__(self):
        self.n_layers = 0
        self.is_adaptive = False
        return
    
    @property 
    def is_last(self) -> bool:
        return True

    def update(
        self, 
        us: Tensor, 
        neglogfus_dirt: Tensor
    ) -> Tuple[Tensor, Tensor]:
        
        xs, neglogdets = self.apply_preconditioner(us)
        neglogfxs = self.target_func(xs)
        neglogfus = neglogfxs + neglogdets

        log_weights = -neglogfus + neglogfus_dirt
        return log_weights, neglogfus
        
    def ratio_func(
        self,
        method: str,
        rs: Tensor, 
        us: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:
        xs, neglogdets = self.apply_preconditioner(us)
        neglogfxs = self.target_func(xs)
        neglogfus = neglogfxs + neglogdets
        return neglogfus