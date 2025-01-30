import torch

from .linear_domain import LinearDomain


class BoundedDomain(LinearDomain):

    def __init__(
        self, 
        bounds: torch.Tensor|None=None
    ):
        
        if bounds is None:
            bounds = torch.tensor([-1.0, 1.0])

        if bounds[0] >= bounds[1]:
            msg = "Left-hand bound must be less than right-hand bound."
            raise Exception(msg)

        self._bounds = bounds
        self._mean = self.bounds.mean()
        self._dxdl = 0.5 * (self.bounds[1] - self.bounds[0])
        
        self._left = self.bounds[0]
        self._right = self.bounds[1]

        return
    
    @property
    def bounds(self):
        return self._bounds
    
    @property 
    def left(self):
        return self._left
    
    @property
    def right(self):
        return self._right

    @property
    def mean(self):
        return self._mean
    
    @property
    def dxdl(self):
        return self._dxdl