import torch 
from torch import Tensor
from torchdiffeq import odeint


class ContaminantSolver():

    def __init__(self, xs: Tensor, ys: Tensor, x0: Tensor | None = None):
        self.xs = xs
        self.ys = ys
        self.nx = self.xs.numel()
        self.ny = self.ys.numel()
        if x0 is None:
            x0 = torch.tensor([0.0, 0.5])
        self.x0 = x0 
        self.t = torch.tensor([0.0, torch.inf])
        return
    
    def interpolate_flux(self, xy: Tensor) -> Tensor:
    
        x, y = xy 

        i = min(torch.sum(self.xs <= x), self.nx) - 1
        j = min(torch.sum(self.ys <= y), self.ny) - 1

        x0, x1 = self.xs[i], self.xs[i+1]
        y0, y1 = self.ys[j], self.ys[j+1]

        f_xy0 = (self.flux[i, j] * (x1-x) + self.flux[i+1, j] * (x-x0)) / (x1-x0)
        f_xy1 = (self.flux[i, j+1] * (x1-x) + self.flux[i+1, j+1] * (x-x0)) / (x1-x0)

        f_xy = (f_xy0 * (y-y0) + f_xy1 * (y1-y)) / (y1-y0)

        return f_xy

    @staticmethod
    def event_fn(t: Tensor, x: Tensor) -> Tensor: 
        return x[0] < 1.0
    
    def rhs_transport(self, t: Tensor, x: Tensor) -> Tensor:

        if (x < 0.0).any() or (x > 1.0).any():
            return torch.tensor([0.0, 0.0])

        kdudx = self.interpolate_flux(x)
        return -kdudx

    def solve(self, flux: Tensor) -> Tensor:
        self.flux = flux.reshape(self.nx, self.ny, 2)
        t_break = odeint(self.rhs_transport, self.x0, self.t, 
                         event_fn=self.event_fn)[0]
        return t_break

    def solve_ts(self, flux: Tensor, ts: Tensor) -> Tensor:
        self.flux = flux.reshape(self.nx, self.ny, 2)
        xs = odeint(self.rhs_transport, self.x0, ts)
        return xs  # type: ignore