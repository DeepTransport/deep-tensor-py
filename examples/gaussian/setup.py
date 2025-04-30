"""Builds a DIRT approximation to an ND Gaussian with an arbitrary mean 
and covariance matrix.
"""

from matplotlib import pyplot as plt
import torch
from torch import Tensor

import deep_tensor as dt

from random_gaussian import RandomGaussian


plt.style.use("examples/plotstyle.mplstyle")
torch.manual_seed(10)

g = RandomGaussian(dim=3)

bounds = torch.tensor([-4.0, 4.0])
domain = dt.BoundedDomain(bounds=bounds)
reference = dt.GaussianReference(domain=domain)

prior = dt.PriorTransformation(
    reference, 
    g.Q, 
    g.Q_inv, 
    g.neglogabsdet_Q_inv, 
    g.dim
)

poly = dt.Fourier(order=20)
dirt_options = dt.DIRTOptions()
tt_options = dt.TTOptions(tt_method="amen", max_rank=20, max_cross=3, cdf_tol=1e-3)
# bridge = dt.SingleLayer()
bridge = dt.Tempering()

dirt = dt.TTDIRT(
    g.negloglik, 
    prior,
    poly, 
    bridge=bridge,
    sirt_options=tt_options,
    dirt_options=dirt_options
)

# import cProfile
# cProfile.run("""
# dirt = dt.TTDIRT(
#     g.negloglik, 
#     prior,
#     poly, 
#     bridge=bridge,
#     sirt_options=tt_options,
#     dirt_options=dirt_options
# )
# """, filename="test")

# import pstats
# from pstats import SortKey
# p = pstats.Stats('test')
# p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(100)



def plot_potentials(
    potentials_true: Tensor,
    potentials_dirt: Tensor
):
    
    min_potential = torch.min(potentials_dirt.min(), potentials_true.min())
    max_potential = torch.max(potentials_dirt.max(), potentials_true.max())
    
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([min_potential, max_potential], [min_potential, max_potential], ls="--", c="k", zorder=1)
    ax.scatter(potentials_true, potentials_dirt, s=5, zorder=2)
    
    ax.set_xlabel(r"$-\log(f(x))$ (True)")
    ax.set_ylabel(r"$-\log(\hat{f}(x))$ (DIRT)")
    ax.set_title("Potential Comparison")

    plt.show()