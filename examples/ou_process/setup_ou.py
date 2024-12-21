from matplotlib import pyplot as plt 
import torch

from deep_tensor import (
    ApproxBases,
    BoundedDomain, 
    InputData,
    Lagrange1,
    Legendre, 
    TTOptions, 
    TTSIRT
)

from examples.ou_process.ou import OU

torch.manual_seed(3)


# Set up the OU process
d = 20
a = 0.5

model = OU(d, a)

def potential_func(x: torch.Tensor):
    return model.eval_potential(x)

debug_size = 10_000
debug_x = torch.linalg.solve(model.B, torch.randn((d, debug_size))).T

sample_x = torch.linalg.solve(model.B, torch.randn((d, 1_000))).T
input_data = InputData(sample_x, debug_x)

domain = BoundedDomain(bounds=torch.tensor([-5.0, 5.0]))

bases = ApproxBases(
    polys=Lagrange1(num_elems=20), 
    domains=domain, 
    dim=d
)

# bases = ApproxBases(
#     polys=Legendre(order=40),
#     domains=domain,
#     dim=d
# )

# bases{1} = ApproxBases(Legendre(40), dom, d);
# bases{2} = ApproxBases(Fourier(20), dom, d);
# bases{3} = ApproxBases(Lagrange1(40), dom, d);
# bases{4} = ApproxBases(Lagrangep(5,8), dom, d);
# bases{5} = ApproxBases(Hermite(10), UnboundedDomain(), d);

options = TTOptions(
    tt_method="random",
    als_tol=1e-8, 
    local_tol=1e-6,
    max_rank=20, 
    max_als=4
)

sirt = TTSIRT(
    potential_func, 
    bases, 
    options=options, 
    input_data=input_data
)