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
# input_data = dt.InputData(sample_x, debug_x)
input_data = dt.InputData(xs_debug=debug_x)
# input_data = dt.InputData()

polys_dict = {
    "lagrange_bound": dt.Lagrange1(num_elems=50),
    "legendre_log": dt.Legendre(order=40),
    # "lagrangep_alg": dt.LagrangeP(5, 8),
    "legendre_alg": dt.Legendre(order=40),
}

domains_dict = {
    "lagrange_bound": dt.BoundedDomain(bounds=torch.tensor([-5.0, 5.0])),
    "legendre_log": dt.LogarithmicMapping(4.0),
    # "lagrangep_alg": dt.AlgebraicMapping(4.0),
    "legendre_alg": dt.AlgebraicMapping(4.0),
}

bases_dict = {
    poly: dt.ApproxBases(
        polys=polys_dict[poly], 
        domains=domains_dict[poly], 
        dim=dim
    ) for poly in polys_dict
}

tt_methods_list = ["fixed_rank"]

# zs = 2 * torch.rand(10, dim) - 1
# us, dudzs = domains_dict["legendre_log"].local2approx(zs)
# zs_0, dzdus_0 = domains_dict["legendre_log"].approx2local(us)
# print((zs-zs_0).abs().max())
# print(dudzs/dzdus_0)
# print(dzdus_0)

options_dict = {
    method: dt.TTOptions(
        tt_method=method,
        max_rank=19, 
        max_als=2
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