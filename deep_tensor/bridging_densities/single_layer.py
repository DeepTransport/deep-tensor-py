from typing import Tuple

from torch import Tensor

from .bridge import Bridge


class SingleLayer(Bridge):
    r"""Constructs the DIRT using a single layer.
    
    In this setting, the DIRT algorithm reduces to the SIRT algorithm; 
    see @Cui2022.

    """

    def __init__(self):
        self.num_layers = 0
        self.is_adaptive = False
        return
    
    @property 
    def is_last(self) -> bool:
        return True
    
    def reset(self) -> None:
        self.num_layers = 0
        return

    def update(
        self, 
        us: Tensor, 
        neglogfus_dirt: Tensor
    ) -> Tuple[Tensor, Tensor]:
        neglogfus = self._eval_pullback(us)
        log_weights = -neglogfus + neglogfus_dirt
        return log_weights, neglogfus
        
    def _eval_neglogratio(
        self,
        method: str,
        rs: Tensor, 
        us: Tensor,
        neglogfus_dirt: Tensor
    ) -> Tensor:
        return self._eval_pullback(us)
    
    def _grad_neglogratio(
        self,
        method: str,
        rs: Tensor, 
        us: Tensor,
        neglogfus_dirt: Tensor,
        grad_neglogfus_dirt: Tensor,
        dudrs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return self._grad_pullback(us)
    
    def _grad_neglogbridge(
        self, 
        us: Tensor,
        dudrs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return self._grad_pullback(us)