"""Evaluates the quality of the FTT surrogate to the joint distribution 
for the OU process.

"""

import time

from torch.linalg import norm

from examples.ou_process.setup_ou import * 


headers = [
    "Polynomial",
    "TT Method",
    "Transform Error",
    "Potential Error",
    "PDF Error",
    "Covariance Error",
    "Time (s)"
]
headers = [f"{h:16}" for h in headers]

print("")
print(" | ".join(headers))
print("-+-".join(["-" * 16] * len(headers)))

zs = torch.rand((10_000, dim))

for poly in polys_dict:
    for method in tt_methods_list:

        sirt: dt.TTSIRT = sirts[poly][method]

        t0 = time.time()
        xs, potential_xs = sirt.eval_irt(zs)
        z0 = sirt.eval_rt(xs)
        t1 = time.time()

        transform_error = norm(zs-z0, ord="fro")
        potential_error = norm(potential_func(xs) - potential_xs)
        pdf_error = norm(
            torch.exp(-potential_func(xs))
            - torch.exp(-potential_xs)
        )
        cov_error = norm(model.C - torch.cov(xs.T)) / norm(model.C)

        info = [
            f"{poly:16}",
            f"{method:16}",
            f"{transform_error:=16.5e}",
            f"{potential_error:=16.5e}",
            f"{pdf_error:=16.5e}",
            f"{cov_error:=16.5e}",
            f"{t1-t0:=16.5f}"
        ]
        print(" | ".join(info))

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        for ax in axes:
            ax.set_box_aspect(1)

        axes[0].hist((potential_func(xs) - potential_xs).abs().log())
        axes[0].set_xlabel("log(Error)")
        axes[0].set_title("Error in potential function")

        axes[1].scatter(potential_func(xs), potential_xs, s=4)
        axes[1].set_xlabel("True potential")
        axes[1].set_ylabel("FTT")
        axes[1].set_title("True potential vs FTT")
        
        plt.savefig(f"examples/ou_process/figures/01_gaussian_{poly}_{method}.pdf")