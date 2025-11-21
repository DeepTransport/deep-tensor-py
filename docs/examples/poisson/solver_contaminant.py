import math 

import torch 
from torch import Tensor


class ContaminantSolver():
    """Assumes that the spacing in the x and y directions is constant,
    and that ."""

    def __init__(self, xs: Tensor, ys: Tensor, x0: Tensor | None = None):
        self.xs = xs
        self.ys = ys
        self.nx = self.xs.numel()
        self.ny = self.ys.numel()
        self.dx = self.xs[1] - self.xs[0]
        self.dy = self.ys[1] - self.ys[0]
        if x0 is None:
            x0 = torch.tensor([0.0, 0.5])
        self.x0 = x0 
        self.t = torch.tensor([0.0, torch.inf])
        return
    
    def interpolate_flux(self, xy: Tensor) -> Tensor:
    
        x, y = xy 
        j = min(math.floor(x / self.dx), self.nx-2)
        i = min(math.floor(y / self.dy), self.ny-2)

        x0, x1 = self.xs[j], self.xs[j+1]
        y0, y1 = self.ys[i], self.ys[i+1]

        k_00 = self.k[i, j]
        k_01 = self.k[i+1, j]
        k_10 = self.k[i, j+1]
        k_11 = self.k[i+1, j+1]

        # Interpolate k
        k_0y = (k_00 * (y1-y) + k_01 * (y-y0)) / self.dy
        k_1y = (k_10 * (y1-y) + k_11 * (y-y0)) / self.dy
        k_xy = (k_0y * (x-x0) + k_1y * (x1-x)) / self.dx

        # Interpolate u
        u_00 = self.u[i, j]
        u_01 = self.u[i+1, j]
        u_10 = self.u[i, j+1]
        u_11 = self.u[i+1, j+1]

        u_0y = (u_01 * (y1-y) + u_00 * (y-y0)) / self.dy 
        u_1y = (u_11 * (y1-y) + u_10 * (y-y0)) / self.dy 
        u_x0 = (u_00 * (x1-x) + u_10 * (x-x0)) / self.dx
        u_x1 = (u_01 * (x1-x) + u_11 * (x-x0)) / self.dx 

        dudx = (u_1y - u_0y) / self.dx
        dudy = (u_x1 - u_x0) / self.dy 

        flux = -k_xy * torch.hstack([dudx, dudy])
        return flux
    
    def rhs_transport(self, t: Tensor, x: Tensor) -> Tensor:
        if (x < 0.0).any() or (x > 1.0).any():
            return torch.tensor([0.0, 0.0])
        return self.interpolate_flux(x)

    def solve(self, k: Tensor, u: Tensor) -> Tensor:
        self.k, self.u = k, u
        x = self.x0.clone()
        t = 0.0
        while True:
            flux = self.interpolate_flux(x)
            dt = 1.0 / (2 * (self.nx-1) * flux.square().sum().sqrt())
            x += dt * flux
            t += dt 
            if x[0] > 1.0:
                return t

    def solve_ts(self, k: Tensor, u: Tensor, ts: Tensor) -> Tensor:
        self.k, self.u = k, u
        xs = [self.x0]
        t = 0.0
        while True:
            flux = self.interpolate_flux(xs[-1])
            dt = 1.0 / (2 * (self.nx-1) * flux.square().sum().sqrt())
            x = xs[-1] + dt * flux
            xs.append(x)
            t += dt 
            if xs[-1][0] > 1.0:
                return torch.vstack(xs)