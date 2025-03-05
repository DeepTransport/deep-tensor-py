"""Evaluates the quality of the FTT surrogate to some of the marginal 
distributions of the OU process.

"""

import time

from torch.linalg import norm

from examples.ou_process.setup_ou_domain_mappings import *


subsets = ["first", "last"]

headers = [
    "Polynomial",
    "TT Method",
    "Subset",
    "Transform Error",
    "Density Error",
    "Approx. Error",
    "Time (s)"
]
headers = [f"{h:16}" for h in headers]

print("")
print(" | ".join(headers))
print("-+-".join(["-" * 16] * len(headers)))

zs = torch.rand((10_000, dim))

for poly in polys_dict:
    for method in options_dict:

        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        for ax in axes.flat:
            ax.set_box_aspect(1)

        for i, subset in enumerate(subsets):

            sirt: dt.TTSIRT = sirts[poly][method]
            t0 = time.time()
            if subset == "first":
                indices = torch.arange(3)
            else:
                indices = torch.arange(dim-1, 1, -1)

            t0 = time.time()
            xs, potential_xs = sirt.eval_irt(zs[:, indices], subset)
            fxs = sirt.eval_pdf(xs, subset)
            z0 = sirt.eval_rt(xs, subset)
            t1 = time.time()

            potential_true = model.eval_potential_marginal(indices, xs)
            transform_error = norm(zs[:, indices] - z0, ord="fro")
            density_error = norm(torch.exp(-potential_xs) - fxs)
            approx_error = norm(torch.exp(-potential_xs) - torch.exp(-potential_true))
            info = [
                f"{poly:16}",
                f"{method:16}",
                f"{subset:16}",
                f"{transform_error:=16.5e}",
                f"{density_error:=16.5e}",
                f"{approx_error:=16.5e}",
                f"{t1-t0:=16.5f}"
            ]
            print(" | ".join(info))

            axes[i][0].scatter(torch.arange(10_000), torch.abs(potential_true - potential_xs), s=4)
            axes[i][0].set_ylim(bottom=0.0)
            axes[i][0].set_ylabel("Error")
            axes[i][0].set_title("Error in potential function")

            axes[i][1].scatter(potential_true, potential_xs, s=4)
            axes[i][1].set_xlabel("True potential")
            axes[i][1].set_ylabel("FTT")
            axes[i][1].set_title("True potential vs FTT")
            
        plt.savefig(f"examples/ou_process/figures/02_marginal_{poly}_{method}.pdf")