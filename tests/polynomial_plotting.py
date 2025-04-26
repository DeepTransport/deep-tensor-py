from matplotlib import pyplot as plt
import torch
from torch import Tensor

import deep_tensor as dt


plt.style.use("examples/plotstyle.mplstyle")

plot_params_interp = {
    "c": "tab:blue", 
    "zorder": 1,
    "label": "Interp"
}

plot_params_exact = {
    "ls": "--", 
    "c": "tab:orange", 
    "zorder": 2,
    "label": "Exact"
}

tau = 1e-8

# def pdf(ls: Tensor):
#     return torch.exp(-5.0 * torch.abs(ls - 0.2))

std = 0.1

def pdf(ls: Tensor):
    return torch.exp(-(ls.square() / (2 * std**2)))

z_min = 0.5 * (1.0 + torch.erf(-1.0 / (std * torch.tensor(2.0).sqrt())))
z_max = 0.5 * (1.0 + torch.erf(1.0 / (std * torch.tensor(2.0).sqrt())))

def cdf(ls: Tensor):
    return (0.5 * (1.0 + torch.erf(ls / (std * torch.tensor(2.0).sqrt()))) - z_min) / (z_max-z_min)

order = 50
num_elems = 50
order_p, num_elems_p = 5, 10

polys: dict[str, dt.Spectral] = {
    "Chebyshev1st": dt.Chebyshev1st(order),
    "Chebyshev2nd": dt.Chebyshev2nd(order),
    "Fourier": dt.Fourier(order),
    "Lagrange1": dt.Lagrange1(num_elems),
    "LagrangeP": dt.LagrangeP(order_p, num_elems_p),
    "Legendre": dt.Legendre(order)
}

for poly_name, poly in polys.items():

    poly_cdf = dt.construct_cdf(poly)
    
    ps_nodes = pdf(poly.nodes)
    ws_nodes = poly.eval_measure(poly.nodes)

    coefs = torch.sqrt(ps_nodes / ws_nodes)[:, None]

    if isinstance(poly, dt.Spectral):
        coefs = poly.node2basis @ coefs

    pp = poly.eval_radon(coefs, poly_cdf.nodes).square()

    ls = torch.linspace(*poly.domain, 10_000)
    f_sqrt_interp = poly.eval_radon(coefs, ls).flatten()
    f_interp = f_sqrt_interp.square()
    F_interp = poly_cdf.eval_cdf(pp, ls)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ps_true = pdf(ls)
    ws_true = poly.eval_measure(ls)

    f_sqrt_exact = torch.sqrt(ps_true / ws_true)
    f_exact = ps_true / ws_true
    F_exact = cdf(ls)

    # print(torch.linalg.norm(f_exact-f_interp))
    # print(torch.linalg.norm(F_exact-F_interp))
    print((f_exact-f_interp).abs().max())
    print((F_exact-F_interp).abs().max())

    axes[0].plot(ls, f_sqrt_interp, **plot_params_interp)
    axes[0].plot(ls, f_sqrt_exact, **plot_params_exact)

    axes[1].plot(ls, f_interp, **plot_params_interp)
    axes[1].plot(ls, f_exact, **plot_params_exact)

    axes[2].plot(ls, F_interp, **plot_params_interp)
    axes[2].plot(ls, F_exact, **plot_params_exact)

    axes[0].set_xlabel(r"$\ell$")
    axes[0].set_ylabel(r"$\sqrt{\hat{f}(\ell)/\omega(\ell)}$")
    axes[1].set_xlabel(r"$\ell$")
    axes[1].set_ylabel(r"$\hat{f}(\ell)/\omega(\ell)$")
    axes[2].set_xlabel(r"$\ell$")
    axes[2].set_ylabel(r"$F(\ell)$")

    axes[1].set_title(poly_name)
    axes[0].legend()

    plt.show()