import math
from pathlib import Path

from matplotlib import pyplot as plt
import torch
from torch import Tensor

import deep_tensor as dt
from examples.plotting import add_arrows, set_plot_style
from examples.shock import GammaNormalMapping, load_shock_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plot_path = Path(__file__).parent
torch.manual_seed(0)
set_plot_style()


data = load_shock_data(device)
failure_dists, censored = data.failure_dists, data.censored


def _neglogpost(xs: Tensor, params: Tensor) -> Tensor:

    bs, t2s = params[:, :-1], params[:, -1:]

    # Compute theta_1 values
    bxs = torch.sum(bs[:, 1:, None] * xs.T[None, :, :], dim=1)
    t1s = torch.exp(bs[:, :1] + bxs)

    # Add contribution of uncensored failure times
    neglogliks = (
        -torch.log(t2s / t1s[:, ~data.censored]) 
        - (t2s - 1.0) * torch.log(failure_dists[~censored] / t1s[:, ~censored]) 
        + (failure_dists[~censored] / t1s[:, ~censored]) ** t2s
    ).sum(dim=1)
    
    # Add contribution of censored failure times
    neglogliks += (
        (failure_dists[censored] / t1s[:, censored]) ** t2s
    ).sum(dim=1)

    neglogpris = (
        - (0.5*D + alpha) * t2s.log().flatten()
        + gamma * t2s.flatten()
        + 0.5 * torch.sum(t2s * (bs-ms)**2 / sds**2, dim=1)
    )
    
    neglogposts = neglogliks + neglogpris - 125.0
    return neglogposts


ftts = ["FTT", "EFTT (POD)", "EFTT (ACA)"]
colours = ["tab:blue", "tab:red", "tab:green"]

tt_options = dt.TTOptions(tt_method="amen", max_als=2, init_rank=8, verbose=2)
eftt_pod_options = dt.EFTTOptions(fibre_method="random", tol_svd=1.0e-2, num_snapshots=50)
eftt_aca_options = dt.EFTTOptions(fibre_method="aca", tol_aca=1.0e-2, num_aca=100)

Ds = torch.tensor([2, 4, 6, 8, 10, 12, 14]) 

num_replications = 10

dhells = torch.zeros((len(Ds), num_replications, len(ftts)))
evals = torch.zeros((len(Ds), num_replications, len(ftts)))
max_tuckers = torch.zeros((len(Ds), num_replications, len(ftts)))

# Define prior coefficients
alpha = torch.tensor(6.8757, device=device)
gamma = torch.tensor(2.2932, device=device)

reference = dt.GaussianReference()
basis = dt.Lagrange1(num_elems=29, device=device)

for i, D in enumerate(Ds):

    D = int(D)
    dim = D + 2

    # Define means and standard deviations of beta coefficients
    ms = torch.zeros((D+1,), device=device)
    ms[0] = math.log(30796)
    sds = torch.ones((D+1,), device=device)
    sds[0] = math.sqrt(0.1563)

    # Define bounds for parameters
    bounds = torch.zeros((D+2, 2), device=device)
    bounds[:-1] = torch.vstack((ms - 3.0*sds, ms + 3.0*sds)).T 
    bounds[-1] = torch.tensor([1e-8, 13.0], device=device)

    # Define preconditioner
    preconditioner = GammaNormalMapping(
        reference, bounds, 
        alpha, gamma, 
        ms, sds, dim
    )

    bases = dt.ApproxBases(basis, dim)

    for j in range(num_replications):
        
        # Generate covariates
        xs = torch.randn((failure_dists.numel(), D), device=device) / D

        def neglogpost(params: Tensor) -> Tensor:
            return _neglogpost(xs, params)

        target_func = dt.TargetFunc(neglogpost)

        rs = reference.random(n=5_000, d=dim, device=device)

        for k, ftt_name in enumerate(ftts):

            tt = dt.TT(tt_options, device=device)

            if ftt_name == "FTT":
                ftt = dt.FTT(bases, tt, device=device)
            elif ftt_name == "EFTT (POD)":
                ftt = dt.EFTT(bases, tt, eftt_pod_options, device=device)
            elif ftt_name == "EFTT (ACA)":
                ftt = dt.EFTT(bases, tt, eftt_aca_options, device=device)
            else:
                raise Exception(f"Unknown FTT type: '{ftt_name}'.")

            bridge = dt.SingleLayer()
            dirt = dt.DIRT(target_func, preconditioner, ftt, bridge, device=device)

            samples_dirt, potentials_dirt = dirt.eval_irt(rs)

            # Run an independence MCMC sampler
            potentials_true = neglogpost(samples_dirt)

            dhell = dt.compute_f_divergence(-potentials_dirt, -potentials_true).sqrt().item()

            dhells[i][j][k] = dhell 
            evals[i][j][k] = dirt.num_eval_construction
            if isinstance(dirt.sirts[0].ftt, dt.EFTT):
                max_tuckers[i][j][k] = dirt.sirts[0].ftt.basis_dims.max()

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))

dhells_mean = dhells.mean(dim=1)
dhells_sd = dhells.std(dim=1)

evals_mean = evals.mean(dim=1)
evals_sd = evals.std(dim=1)

max_tuckers_mean = max_tuckers.mean(dim=1)
max_tuckers_sd = max_tuckers.std(dim=1)

for i in range(len(ftts)):
    axes[0].errorbar(
        Ds, dhells_mean[:, i], yerr=dhells_sd[:, i], 
        fmt="-o", markersize=6, linewidth=1.5, elinewidth=1.5,
        capsize=3.0, capthick=1.5, label=ftts[i], c=colours[i]
    )
    axes[1].errorbar(
        Ds, evals_mean[:, i], yerr=evals_sd[:, i], 
        fmt="-o", markersize=6, linewidth=1.5, elinewidth=1.5,
        capsize=3.0, capthick=1.5, label=ftts[i], c=colours[i]
    )
    if "EFTT" in ftts[i]:
        axes[2].errorbar(
            Ds, max_tuckers_mean[:, i], yerr=max_tuckers_sd[:, i],
            fmt="-o", markersize=6, linewidth=1.5, elinewidth=1.5,
            capsize=3.0, capthick=1.5, label=ftts[i], c=colours[i]
        )

for ax in axes.flat:
    add_arrows(ax)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"$D$")

axes[0].set_ylabel(r"$\mathcal{D}_{\mathrm{H}}(\pi_{\beta, \theta_{2} | t}, \pi_{\hat{\beta}, \hat{\theta}_{2} | t})$")
axes[1].set_ylabel(r"Evaluations")
axes[2].set_ylabel(r"$\max_{k} R_{k}$")
axes[0].set_ylim(bottom=0.0)
axes[1].set_ylim(bottom=0.0)
axes[1].legend(fontsize=10)
axes[1].ticklabel_format(axis="y", scilimits=(0, 0))

save_path = plot_path.joinpath("plots", "dims.pdf").resolve()
plt.savefig(save_path)