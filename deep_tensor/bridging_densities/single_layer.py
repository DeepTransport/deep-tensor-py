from typing import Dict, Tuple

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
    
    In this setting, the DIRT algorithm reduces to the SIRT algorithm; 
    see @Cui2022.

    """

    def __init__(self):
        self.n_layers = 0
        self.is_adaptive = False
        return

    def update(self, 
        method: str, 
        rs: Tensor, 
        us: Tensor, 
        neglogfus_dirt: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        
        xs, neglogdets = self.apply_preconditioner(us)
        neglogfxs = self.target_func(xs)
        neglogfus = neglogfxs + neglogdets

        log_weights = -neglogfus + neglogfus_dirt
        return log_weights, neglogfus, neglogfus
        
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


class SavedSingleLayer(AbstractSingleLayer):

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        return
    
    def _compute_log_weights(self, neglogliks, neglogpris, neglogfxs):
        raise NotImplementedError()
    
    def _get_ratio_func(self, reference, method, rs, neglogliks, neglogpris, neglogfxs):
        raise NotImplementedError()