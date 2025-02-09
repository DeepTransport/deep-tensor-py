"""Builds a set of TTSIRT approximations to the posterior distribution 
associated with an OU process.

"""


from matplotlib import pyplot as plt
import torch

import deep_tensor as dt

from examples.ou_process.ou import OU

plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(0)


dim = 20
a = 0.50

model = OU(dim, a)

def potential_func(x: torch.Tensor):
    return model.eval_potential(x)

debug_size = 10_000
debug_x = torch.linalg.solve(model.B, torch.randn((dim, debug_size))).T

sample_x = torch.linalg.solve(model.B, torch.randn((dim, 1_000))).T
input_data = dt.InputData(sample_x, debug_x)

domain = dt.BoundedDomain(bounds=torch.tensor([-5.0, 5.0]))

polys_dict = {
    "lagrange1": dt.Lagrange1(num_elems=40),
    "legendre": dt.Legendre(order=40),
    "fourier": dt.Fourier(order=20),
    "lagrangep": dt.LagrangeP(order=5, num_elems=8)
}

bases_dict = {
    poly: dt.ApproxBases(
        polys=polys_dict[poly], 
        domains=domain, 
        dim=dim
    ) for poly in polys_dict
}

tt_methods_list = ["amen", "random", "fixed_rank"]

options_dict = {
    method: dt.TTOptions(
        tt_method=method,
        max_rank=19, 
        max_als=1
    ) for method in tt_methods_list
}

sirts = {}

for poly in bases_dict:
    sirts[poly] = {}
    for method in options_dict:
        sirts[poly][method] = dt.TTSIRT(
            potential_func, 
            bases_dict[poly], 
            options=options_dict[method], 
            input_data=input_data
        )