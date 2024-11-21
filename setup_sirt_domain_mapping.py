import torch

from ou import OU

from deep_tensor import (
    ApproxBases,
    BoundedDomain, 
    InputData,
    Lagrange1,
    Legendre, 
    TTOptions, 
    TTSIRT
)

from deep_tensor.utils import info

torch.manual_seed(0)


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

# bases = ApproxBases(
#     polys=[Legendre(order=40)], 
#     in_domains=[domain], 
#     dim=d
# )

bases = ApproxBases(
    polys=[Lagrange1(num_elems=40)],
    in_domains=[domain],
    dim=d
)

# bases{1} = ApproxBases(Legendre(40), dom, d);
# bases{2} = ApproxBases(Fourier(20), dom, d);
# bases{3} = ApproxBases(Lagrange1(40), dom, d);
# bases{4} = ApproxBases(Lagrangep(5,8), dom, d);
# %bases{5} = ApproxBases(Hermite(10), UnboundedDomain(), d);

options = TTOptions(
    tt_method="fixed_rank", 
    als_tol=1e-8, 
    local_tol=1e-8,
    max_rank=20, 
    max_als=1
)

irt = TTSIRT(
    potential_func, 
    bases, 
    options=options, 
    input_data=input_data
)

from deep_tensor.directions import Direction

irt.marginalise(direction=Direction.BACKWARD)

zs = torch.rand((10_000, d))
xs, fs = irt.eval_irt_nograd(zs)  # I assume that the marginal/PDF/CDF stuff is wrong somewhere. Probably the CDF stuff..? The PDF stuff is tested with the debug samples.
z0 = irt.eval_rt(xs)

transform_error = torch.linalg.norm(zs-z0, ord="fro")
potential_error = torch.linalg.norm(potential_func(xs) - fs)
pdf_error = torch.linalg.norm(torch.exp(-potential_func(xs)) - torch.exp(-fs))

info(f"Transform error: {transform_error}")
info(f"Potential error: {potential_error}")
info(f"PDF error: {pdf_error}")

#disp(['cov eror: ' num2str(norm(data.C - cov(r'))/norm(data.C))])
from matplotlib import pyplot as plt 

plt.scatter(potential_func(xs), fs, s=10)
plt.xlabel("Potential")
plt.ylabel("FTT")
plt.title("Actual potential function vs FTT")
plt.show()

# subplot(2,2,1); plot(abs(func(r) - f), '.'); title('actual potential function vs fft')
# subplot(2,2,2); plot(func(r), f, '.');
# subplot(2,2,3); plot(data.C - cov(r')); title('actual covariance vs sample covariance')