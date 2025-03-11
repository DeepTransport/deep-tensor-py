import torch

from deep_tensor import Direction


class OU():

    def __init__(self, d: float, a: float):

        self.a = torch.tensor(a)
        self.d = torch.tensor(d)

        self.A = torch.diag(torch.full((self.d-1,), -torch.sqrt(1.0 - self.a**2)), -1) + torch.eye(self.d)
        self.D = torch.diag(torch.concatenate((torch.tensor([1]), torch.full((d-1,), a))))

        self.B = torch.linalg.solve(self.D, self.A)
        self.Q = self.B.T @ self.B
        self.C = torch.linalg.inv(self.Q)

        self.norm = torch.sqrt((2.0*torch.pi)**self.d / torch.linalg.det(self.Q))

        return

    def eval_potential(self, x: torch.Tensor) -> torch.Tensor:
        f = 0.5 * torch.sum((self.B @ x.T) ** 2, dim=0) + torch.log(self.norm)
        return f
    
    def eval_potential_marginal(self, indices, x: torch.Tensor) -> torch.Tensor:
        C = self.C[indices[:, None], indices[None, :]]
        f = 0.5 * torch.sum(torch.linalg.solve(C, x.T) * x.T, dim=0)
        z = 0.5 * torch.linalg.det(C).log() + 0.5 * indices.numel() * torch.tensor(2.0 * torch.pi).log()
        f = z + f 
        return f
    
    def eval_potential_cond(
        self, 
        xs_cond_l: torch.Tensor, 
        xs_cond_r: torch.Tensor, 
        subset: str
    ) -> torch.Tensor:
        
        xs = torch.hstack((xs_cond_l, xs_cond_r))
        fs = 0.5 * (self.B @ xs.T).square().sum(dim=0) + self.norm.log()

        dim_xl = xs_cond_l.shape[1]
        dim_xr = xs_cond_r.shape[1]

        if subset == "first":
            indices = torch.arange(dim_xl)
            fms = self.eval_potential_marginal(indices, xs_cond_l)
        else:
            indices = dim_xl + torch.arange(dim_xr)
            fms = self.eval_potential_marginal(indices, xs_cond_r)

        fs -= fms
        return fs