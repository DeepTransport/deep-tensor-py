import torch
from torch.linalg import solve

import deep_tensor as dt

from examples.ou_process.ou import OU


def build_ou_sirt(
    poly: dt.Basis1D, 
    method: str, 
    dim: int
) -> dt.TTSIRT:

    model = OU(d=dim, a=0.5)

    def potential_func(x: torch.Tensor):
        return model.eval_potential(x)
    
    n_samp = 1_000
    xs_samp = solve(model.B, torch.randn((dim, n_samp))).T
    input_data = dt.InputData(xs_samp)

    domain = dt.BoundedDomain(bounds=torch.tensor([-5.0, 5.0]))
    bases = dt.ApproxBases(polys=poly, domains=domain, dim=dim) 
    options = dt.TTOptions(tt_method=method, max_rank=20, max_als=1) 
    
    sirt = dt.TTSIRT(
        potential_func, 
        bases, 
        options=options, 
        input_data=input_data
    )
    
    return sirt