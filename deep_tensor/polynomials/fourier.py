import torch

from .spectral import Spectral


class Fourier(Spectral):

    def __init__(self, order: int):

        self._domain = torch.tensor([-1.0, 1.0])
        self._constant_weight = True
        
        self.order = order
        self.m = self.order + 1

        num_nodes = 2 + self.order * 2
        n = torch.arange(num_nodes)

        self.nodes = torch.sort((2.0/num_nodes) * (n+1) - 1)
        self.weights = torch.ones_like(self.nodes) / num_nodes

        self.c = (torch.arange(self.order)+1) * torch.pi

        self.post_construction()
        self.node2basis[-1] *= 0.5  # TODO: figure out what's going on here / whether this has any effect on the TT building

        return

    @property
    def domain(self) -> torch.Tensor:
        return self._domain
    
    @property
    def constant_weight(self) -> bool:
        return self._constant_weight

    def sample_measure(self, n: int) -> torch.Tensor:
        return torch.rand(n) * 2 - 1
    
    def sample_measure_skip(self, n: int) -> torch.Tensor:
        return self.sample_measure(n)
    
    def eval_measure(self, x: torch.Tensor):
        return 0.5 * torch.ones_like(x)
    
    def eval_log_measure(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full(x.shape, torch.log(0.5))
    
    def eval_measure_deriv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
    
    def eval_log_measure_deriv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
    
    def eval_basis(self, xs: torch.Tensor) -> torch.Tensor:
        raise Exception("TODO: write me")
        return 
    
    def eval(self, coeffs, xs):
        raise Exception("TODO: write me")
        return super().eval(coeffs, xs)

"""
        
        function f = eval_basis(obj, x)
            %
            tmp = x(:).*obj.c;     
            %f = [ones(size(x(:))), sin(tmp)*sqrt(2), cos(tmp)*sqrt(2)];
            f = [ones(size(x(:))), sin(tmp)*sqrt(2), cos(tmp)*sqrt(2), ...
                cos( x(:)*(obj.m*pi) )*sqrt(2)];
        end
        
        function f = eval_basis_deri(obj, x)
            %
            tmp = x(:).*obj.c;
            %f = [zeros(length(x),1), cos(tmp).*(obj.c*sqrt(2)), -sin(tmp).*(obj.c*sqrt(2))];
            f = [zeros(length(x),1), cos(tmp).*(obj.c*sqrt(2)), -sin(tmp).*(obj.c*sqrt(2)), ...
                -sin(x(:)*(obj.m*pi))*(sqrt(2)*obj.m*pi)];
        end
    end
end"""