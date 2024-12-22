"""TODO: write docstring.
TODO: add timing.

Sets up the OU process"""

from matplotlib import pyplot as plt
import torch

import deep_tensor as dt

from examples.ou_process.ou import OU

plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(64)


dim = 20
a = 0.5

model = OU(dim, a)

def potential_func(x: torch.Tensor):
    return model.eval_potential(x)

debug_size = 10_000
debug_x = torch.linalg.solve(model.B, torch.randn((dim, debug_size))).T

sample_x = torch.linalg.solve(model.B, torch.randn((dim, 1_000))).T
input_data = dt.InputData(sample_x, debug_x)

domain = dt.BoundedDomain(bounds=torch.tensor([-5.0, 5.0]))

polys_list = [
    dt.Legendre(order=40),
    dt.Fourier(order=20),
    dt.Lagrange1(num_elems=40)
]

# bases{4} = ApproxBases(Lagrangep(5,8), dom, d);

bases_list = [
    dt.ApproxBases(polys=polys, domains=domain, dim=dim)
    for polys in polys_list
]

tt_methods_list = ["random", "fixed_rank"]

options_list = [
    dt.TTOptions(
        tt_method=method,
        als_tol=1e-4, 
        max_rank=20, 
        max_als=1
    ) for method in tt_methods_list
]

sirts = {}

for i, bases in enumerate(bases_list):
    sirts[i] = {}
    for j, options in enumerate(options_list):
        sirts[i][j] = dt.TTSIRT(
            potential_func, 
            bases, 
            options=options, 
            input_data=input_data
        )