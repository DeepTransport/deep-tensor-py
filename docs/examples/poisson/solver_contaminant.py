import torch 
from torch import Tensor


def compute_reordering_matrix(xs: Tensor) -> Tensor:
    """Returns a sparse matrix that re-orders a set of coordinates such 
    they are ordered from smallest to largest in the first coordinate, 
    and (in cases where the first coordinate is the same) smallest to 
    largest in the second coordinate.
    """

    num_xs = xs.shape[0]

    cs = xs.clone()
    cs[:, 1] *= 1.0e+4  # TODO: change this based on coordinate spacing
    inds = torch.vstack([
        torch.arange(num_xs),
        torch.sort(cs.sum(dim=1)).indices
    ])
    vals = torch.ones((num_xs,))
    P = torch.sparse_coo_tensor(inds, vals, size=(num_xs, num_xs))
    
    return P


class ContaminantSolver():
    """Assumes that the spacing in the x and y directions is constant."""

    def __init__(
        self, 
        xs: Tensor, 
        ys: Tensor, 
        coords: Tensor,
        x0: Tensor | None = None,
    ):
        self.xs = xs
        self.ys = ys
        self.P = compute_reordering_matrix(coords)
        self.nx = self.xs.numel()
        self.ny = self.ys.numel()
        self.dx = self.xs[1] - self.xs[0]
        self.dy = self.ys[1] - self.ys[0]
        if x0 is None:
            x0 = torch.tensor([[0.0, 0.5]])
        self.x0 = x0 
        self.t = torch.tensor([0.0, torch.inf])
        return
    
    def interpolate_flux(self, xys: Tensor) -> Tensor:
    
        num_xs = xys.shape[0]

        xs, ys = xys.T 
        jj = torch.minimum(torch.floor(xs / self.dx).int(), torch.tensor(self.nx-2))
        ii = torch.minimum(torch.floor(ys / self.dy).int(), torch.tensor(self.ny-2))

        x0, x1 = self.xs[jj], self.xs[jj+1]
        y0, y1 = self.ys[ii], self.ys[ii+1]

        row_inds = torch.arange(num_xs)

        k_00 = self.ks[row_inds, self.nx*ii+jj]
        k_01 = self.ks[row_inds, self.nx*(ii+1)+jj]
        k_10 = self.ks[row_inds, self.nx*ii+jj+1]
        k_11 = self.ks[row_inds, self.nx*(ii+1)+jj+1]

        u_00 = self.us[row_inds, self.nx*ii+jj]
        u_01 = self.us[row_inds, self.nx*(ii+1)+jj]
        u_10 = self.us[row_inds, self.nx*ii+jj+1]
        u_11 = self.us[row_inds, self.nx*(ii+1)+jj+1]

        k_0y = (k_00 * (y1-ys) + k_01 * (ys-y0)) / self.dy
        k_1y = (k_10 * (y1-ys) + k_11 * (ys-y0)) / self.dy
        k_xy = (k_0y * (xs-x0) + k_1y * (x1-xs)) / self.dx

        u_0y = (u_01 * (y1-ys) + u_00 * (ys-y0)) / self.dy 
        u_1y = (u_11 * (y1-ys) + u_10 * (ys-y0)) / self.dy 
        u_x0 = (u_00 * (x1-xs) + u_10 * (xs-x0)) / self.dx
        u_x1 = (u_01 * (x1-xs) + u_11 * (xs-x0)) / self.dx 

        dudx = (u_1y - u_0y) / self.dx
        dudy = (u_x1 - u_x0) / self.dy 

        fluxes = -k_xy * torch.vstack([dudx, dudy])
        return fluxes.T
    
    def rhs_transport(self, t: Tensor, x: Tensor) -> Tensor:
        if (x < 0.0).any() or (x > 1.0).any():
            return torch.tensor([0.0, 0.0])
        return self.interpolate_flux(x)

    def solve(self, log_ks: Tensor, us: Tensor) -> Tensor:
        num_xs = log_ks.shape[0]
        self.ks = log_ks.exp() @ self.P.T
        self.us = us @ self.P.T
        xs = self.x0.repeat(num_xs, 1)
        ts = torch.zeros((num_xs, 1))
        incomplete = torch.full((num_xs,), True)
        while True:
            fluxes = self.interpolate_flux(xs[incomplete])
            dts = 1.0 / (2 * (self.nx-1) * fluxes.square().sum(dim=1, keepdim=True).sqrt())
            xs[incomplete] += dts * fluxes
            ts[incomplete] += dts
            incomplete = xs[:, 0] < 1.0
            if not incomplete.any():
                return ts.flatten()

    def solve_path(self, log_ks: Tensor, us: Tensor) -> Tensor:
        log_ks = torch.atleast_2d(log_ks)
        us = torch.atleast_2d(us)
        self.ks = log_ks.exp() @ self.P.T
        self.us = us @ self.P.T
        xs = [self.x0]
        t = 0.0
        while True:
            flux = self.interpolate_flux(xs[-1])
            dt = 1.0 / (2 * (self.nx-1) * flux.square().sum().sqrt())
            x = xs[-1] + dt * flux
            xs.append(x)
            t += dt 
            if xs[-1][0][0] > 1.0:
                return torch.vstack(xs)