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
    def nodes(self) -> torch.Tensor:
        return self._nodes
    
    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    @property 
    def domain(self) -> torch.Tensor:
        return self._domain 
    
    @property
    def constant_weight(self) -> bool:
        return self._constant_weight
    
    def sample_measure(self, n: int) -> torch.Tensor:
        beta = torch.distributions.beta.Beta(2.0, 2.0)
        xs = beta.sample(n)
        xs = (2.0 * xs) - 1
        return xs
    
    def sample_measure_skip(self, n: int) -> torch.Tensor:
        x0 = 0.5 * (self.nodes.min() - 1.0)
        x1 = 0.5 * (self.nodes.max() + 1.0)
        xs = torch.rand(n) * (x1-x0) + x0
        return xs
    
    def eval_measure(self, xs: torch.Tensor) -> torch.Tensor:
        ws = 0.75 * (1.0 - xs**2)
        return ws
    
    def eval_log_measure(self, xs: torch.Tensor) -> torch.Tensor:
        ws = torch.log(1.0 - xs**2) + torch.log(0.75)
        return ws
    
    def eval_measure_deriv(self, xs: torch.Tensor) -> torch.Tensor:
        ws = -0.75 * xs
        return ws
    
    def eval_log_measure_deriv(self, xs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not implemented.")