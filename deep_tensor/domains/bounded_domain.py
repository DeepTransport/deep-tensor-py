import torch

from .linear_domain import LinearDomain


class BoundedDomain(LinearDomain):

    def __init__(
        self, 
        bounds: torch.Tensor=torch.tensor([-1.0, 1.0])
    ):

        if bounds[0] >= bounds[1]:
            msg = ("Left-hand boundary must be less " 
                   + "than right-hand boundary.")
            raise Exception(msg)

        self._bounds = bounds
        self._mean = torch.mean(self._bounds)
        self._dxdz = 0.5 * (self._bounds[1]-self._bounds[0])
        
        self.left = self.bounds[0]
        self.right = self.bounds[1]

        return
    
    @property
    def bounds(self):
        return self._bounds

    @property
    def mean(self):
        return self._mean
    
    @property
    def dxdz(self):
        return self._dxdz