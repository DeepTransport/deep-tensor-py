"""Evaluates the quality of the FTT surrogate to some of the 
conditional distributions of the OU process.

"""

import time

from torch.linalg import norm

from examples.ou_process.setup_ou import *


subsets = ["first", "last"]

headers = [
    "Polynomial",
    "TT Method",
    "Subset",
    "Approx. Error",
    "Cov. Error",
    "Time (s)"
]
headers = [f"{h:16}" for h in headers]

print("")
print(" | ".join(headers))
print("-+-".join(["-" * 16] * len(headers)))

n_zs = 10_000
zs = torch.rand((n_zs, dim))

m = 2
indices_l = torch.arange(m)
indices_r = torch.arange(m, dim)

for poly in polys_dict:
    for method in options_dict:

        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        for ax in axes.flat:
            ax.set_box_aspect(1)

        for i, subset in enumerate(subsets):

            sirt: dt.TTSIRT = sirts[poly][method]

            if subset == "first":
                xs_cond = debug_x[:, indices_l]
                zs_cond = zs[:, indices_r]
                inds_cov = indices_r
            else:
                xs_cond = debug_x[:, indices_r]
                zs_cond = zs[:, indices_l]
                inds_cov = indices_l

            t0 = time.time()
            ys_cond_sirt, potential_cond_sirt = sirt.eval_cirt(
                xs_cond, 
                zs_cond,
                subset
            )
            t1 = time.time()
            
            if subset == "first":
                potential_cond_true = model.eval_potential_cond(
                    xs_cond, 
                    ys_cond_sirt, 
                    subset=subset
                )
            else:
                potential_cond_true = model.eval_potential_cond(
                    ys_cond_sirt,
                    xs_cond, 
                    subset=subset
                )

            fys_cond_sirt = torch.exp(-potential_cond_sirt)
            fys_cond_true = torch.exp(-potential_cond_true)
            approx_error = norm(fys_cond_true - fys_cond_sirt) 

            cov_cond_true = model.C[inds_cov[:, None], inds_cov[None, :]]
            cov_cond_sirt = torch.cov(ys_cond_sirt.T)
            cov_error = norm(cov_cond_true - cov_cond_sirt) / norm(cov_cond_true)

            info = [
                f"{poly:16}",
                f"{method:16}",
                f"{subset:16}",
                f"{approx_error:=16.5e}",
                f"{cov_error:=16.5e}",
                f"{t1-t0:=16.5e}"
            ]
            print(" | ".join(info))

            axes[i][0].hist((potential_cond_true - potential_cond_sirt).abs().log())
            axes[i][0].set_xlabel("log(Error)")
            axes[i][0].set_title("Error in potential function")

            axes[i][1].scatter(potential_cond_true, potential_cond_sirt, s=4)
            axes[i][1].set_xlabel("True potential")
            axes[i][1].set_ylabel("FTT")
            axes[i][1].set_title("True potential vs FTT")

        fname = f"examples/ou_process/figures/03_conditional_{poly}_{method}.png"
        plt.savefig(fname, dpi=500) 