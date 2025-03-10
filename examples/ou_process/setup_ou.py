"""Builds a set of TTSIRT approximations to the posterior distribution 
associated with an OU process.

"""

from copy import deepcopy

from matplotlib import pyplot as plt
import torch
from torch import Tensor

import deep_tensor as dt

from examples.ou_process.ou import OU

plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(0)


dim = 5
a = 0.5

model = OU(dim, a)

def potential_func(x: Tensor):
    return model.eval_potential(x)

debug_size = 10_000
debug_x = torch.linalg.solve(model.B, torch.randn((dim, debug_size))).T

sample_x = torch.linalg.solve(model.B, torch.randn((dim, 1_000))).T
input_data = dt.InputData(sample_x, debug_x)

domain = dt.BoundedDomain(bounds=torch.tensor([-5.0, 5.0]))

polys_dict = {
    "legendre": dt.Legendre(order=40),
    "fourier": dt.Fourier(order=20),
    "chebyshev1st": dt.Chebyshev1st(order=40),
    "chebyshev2nd": dt.Chebyshev2nd(order=40),
    "lagrange1": dt.Lagrange1(num_elems=40),
    "lagrangep": dt.LagrangeP(order=5, num_elems=8)
}

bases_dict = {
    poly: dt.ApproxBases(
        polys=polys_dict[poly], 
        domains=domain, 
        dim=dim
    ) for poly in polys_dict
}

tt_methods_list = ["amen", "random", "amen_round"]

options_dict = {
    "amen": dt.TTOptions(
        tt_method="amen",
        cross_tol=1e-4,
        local_tol=0.0,
        max_rank=19,
        max_cross=5,
        init_rank=2,
        kick_rank=2
    ),
    "random": dt.TTOptions(
        tt_method="random",
        cross_tol=1e-4,
        local_tol=1e-10,
        max_rank=19,
        max_cross=1
    )
}

sirts = {}

for poly in bases_dict:
    
    sirts[poly] = {}
    
    for method in tt_methods_list:
        
        input_data.count = 0
        
        if method != "amen_round":
            
            sirts[poly][method] = dt.TTSIRT(
                potential_func, 
                bases_dict[poly], 
                options=options_dict[method], 
                input_data=input_data
            )

        else:
            
            sirt: dt.TTSIRT = deepcopy(sirts[poly]["amen"])
            sirt.round(1e-2)
            
            sirts[poly][method] = dt.TTSIRT(
                potential_func, 
                prev_approx=sirt.approx,
                input_data=input_data
            )