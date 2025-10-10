from matplotlib import pyplot as plt
import torch 

import deep_tensor as dt

from benchmark_functions import *

from examples.plotting import set_plot_style


set_plot_style()
torch.manual_seed(0)


genz_funcs = {
    "Oscillatory": build_genz_1,
    "Corner Peak": build_genz_2,
    "Continuous": build_genz_3
}

ftt_names = ["EFTT (ACA)", "EFTT (rand)", "FTT"]

results = {
    func_name: {
        ftt_name: {
            "num_evals": [],
            "l2_errors": []
        } 
        for ftt_name in ftt_names
    } 
    for func_name in genz_funcs
}

dims = [20, 50, 100, 200, 300, 400, 500]

basis = dt.Chebyshev1st(order=99)
tt_options = dt.TTOptions(max_als=10, init_rank=1, tol_max_core_error=1e-01, verbose=0)

for func_name in genz_funcs:
    for dim in dims:

        genz = genz_funcs[func_name](dim)

        # Uniform samples...
        ls_l2 = 2.0 * torch.rand((10_000, dim)) - 1.0
        ys_l2 = genz(ls_l2)
        
        for ftt_name in ftt_names:
            
            bases = dt.ApproxBases(basis, dim=dim)
            tt = dt.TT(tt_options)
            
            if ftt_name == "EFTT (ACA)":
                options = dt.EFTTOptions(fibre_method="aca")
                ftt = dt.EFTT(bases, tt, options)
            elif ftt_name == "EFTT (rand)":
                options = dt.EFTTOptions(fibre_method="random")
                ftt = dt.EFTT(bases, tt, options)
            elif ftt_name == "FTT":
                ftt = dt.FTT(bases, tt)
            else: 
                raise Exception("Unknown FTT type")

            ftt.approximate(genz)

            ys_l2_ftt = ftt(ls_l2).flatten()
            l2_error = torch.linalg.norm(ys_l2 - ys_l2_ftt) / torch.linalg.norm(ys_l2)

            results[func_name][ftt_name]["num_evals"].append(ftt.num_eval)
            results[func_name][ftt_name]["l2_errors"].append(l2_error)
            
            print(f"{func_name} | {dim:3} | {ftt_name:12} | {ftt.num_eval:.2e} | {l2_error:.2e}")


fig, axes = plt.subplots(3, 2, figsize=(6, 8))

for i, func_name in enumerate(genz_funcs):
    
    axes[i][0].set_title(func_name)
    axes[i][1].set_title(func_name)
    axes[i][0].set_xlabel(r"$d$")
    axes[i][1].set_xlabel(r"$d$")
    axes[i][0].set_ylabel(r"$L^{2}$ error")
    axes[i][1].set_ylabel(r"Evaluations")

    axes[i][0].set_yscale("log")
    axes[i][1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0)) 

    for ftt_name in ftt_names:
        l2_errors = results[func_name][ftt_name]["l2_errors"]
        num_evals = results[func_name][ftt_name]["num_evals"]
        axes[i][0].plot(dims, l2_errors, label=ftt_name)
        axes[i][1].plot(dims, num_evals, label=ftt_name)

axes[0][0].legend()

# plt.tight_layout()
plt.savefig("genz.pdf")