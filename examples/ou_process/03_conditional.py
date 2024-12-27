"""TODO: write docstring."""

from torch.linalg import norm

from examples.ou_process.setup_ou import *


directions = {
    "forward": dt.Direction.FORWARD, 
    "backward": dt.Direction.BACKWARD
}

headers = [
    "Polynomial",
    "TT Method",
    "Direction",
    "Approx. Error",
    "Cov. Error"
]
headers = [f"{h:16}" for h in headers]

print("")
print(" | ".join(headers))
print("-+-".join(["-" * 16] * len(headers)))


zs = torch.rand((10_000, dim))

m = 8
indices_l = torch.arange(m)
indices_r = torch.arange(m, dim)

for poly in polys_dict:
    for method in options_dict:

        for i, direction in enumerate(directions):

            sirt: dt.TTSIRT = sirts[poly][method]

            if directions[direction] == dt.Direction.FORWARD:
                xs_cond = debug_x[:, indices_l]
                zs_cond = zs[:, indices_r]
                inds_cov = indices_r
                if sirt.int_dir == dt.Direction.BACKWARD:
                    sirt.marginalise(dt.Direction.FORWARD) 
            else:
                xs_cond = debug_x[:, indices_r]
                zs_cond = zs[:, indices_l]
                inds_cov = indices_l
                if sirt.int_dir == dt.Direction.FORWARD:
                    sirt.marginalise(dt.Direction.BACKWARD)

            ys_cond_sirt, neglogfys_cond_sirt = sirt.eval_cirt(
                xs_cond, 
                zs_cond
            )
            
            if directions[direction] == dt.Direction.FORWARD:
                neglogfys_cond_true = model.eval_potential_cond(
                    xs_cond, 
                    ys_cond_sirt, 
                    dir=directions[direction]
                )
            else:
                neglogfys_cond_true = model.eval_potential_cond(
                    ys_cond_sirt,
                    xs_cond, 
                    dir=directions[direction]
                )

            fys_cond_sirt = torch.exp(-neglogfys_cond_sirt)
            fys_cond_true = torch.exp(-neglogfys_cond_true)
            approx_error = norm(fys_cond_true - fys_cond_sirt) 

            cov_cond_true = model.C[inds_cov[:, None], inds_cov[None, :]]
            cov_cond_sirt = torch.cov(ys_cond_sirt.T)
            cov_error = norm(cov_cond_true - cov_cond_sirt) / norm(cov_cond_true)

            info = [
                f"{poly:16}",
                f"{method:16}",
                f"{direction:16}",
                f"{approx_error:=16.5e}",
                f"{cov_error:=16.5e}"
            ]
            print(" | ".join(info))

            # plt.scatter(torch.arange(10_000), torch.abs(torch.exp(-potential_ys_cond) - torch.exp(-fys_cond)))
            # plt.show()