import torch 

from .recurr import Recurr


class Jacobi11(Recurr):

    def __init__(self, order: int):

        k = torch.arange(order+1)
        
        a = (2*k+3) * (k+2) / (k+1) / (k+3)
        b = torch.zeros_like(k)
        c = (k+2)/(k+3)

        norm = ((2.0*k+3.0) * (k+2.0) / (8.0 * (k+1.0)) * (4/3)).sqrt()
        Recurr.__init__(self, order, a, b, c, norm)
        
        self._domain = torch.tensor([-1.0, 1.0])
        self._constant_weight = False

        return

    @property 
    def domain(self) -> torch.Tensor:
        return self._domain 
    
    @property
    def constant_weight(self) -> bool:
        return self._constant_weight
    
    def sample_measure(self, n: int) -> torch.Tensor:
        return 