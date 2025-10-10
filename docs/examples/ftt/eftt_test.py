from statistics import mean, stdev

import torch

import deep_tensor as dt
from benchmark_functions import *


# default max rank is 20 in constructor.m for FTT class in Strossner et al.
# number of collocation points in each dimension is 100 (see paper)


def geo_mean(x):
    if not isinstance(x, Tensor):
        x = torch.tensor(x)
    return float(x.log().mean().exp())

headers = [
    f"{'Function':12}", 
    f"{'Algorithm':12}", 
    f"{'mean(L2)':8}",
    f"{'sd(log(L2))':11}",
    f"{'mean(Evals)':10}", 
    f"{'sd(Evals)':9}",
    f"{'Max Rank':8}", 
    f"{'Max Basis':9}"
]

divider = r"\hline"  # "-+-".join(["-"*len(header) for header in headers])
print(" & ".join(headers) + r" \\")
print(divider)

tt_options = dt.TTOptions(max_als=10, tol_max_core_error=0.1, init_rank=1, max_rank=20, verbose=0)

ftt_names = ["EFTT (ACA)", "EFTT (rand)", "FTT"]

statistics = {
    func_name: {
        ftt_name: {
            "num_evals": [],
            "l2_errors": [],
            "max_ranks": [],
            "max_bases": []
        } 
        for ftt_name in ftt_names
    } for func_name in FUNCTIONS
}

num_runs = 10

for func_name in FUNCTIONS:

    func, dim = FUNCTIONS[func_name]

    basis = dt.Chebyshev1st(order=99)
    bases = dt.ApproxBases(basis, dim)

    # Uniform samples...
    ls_l2 = 2.0 * torch.rand((10_000, dim)) - 1.0
    ys_l2 = func(ls_l2)

    for ftt_name in ftt_names:

        for i in range(num_runs):

            tt = dt.TT(tt_options)

            if ftt_name == "EFTT (ACA)":
                options = dt.EFTTOptions(fibre_method="aca")
                ftt = dt.EFTT(bases, tt, options)
            elif ftt_name == "EFTT (rand)":
                options = dt.EFTTOptions(fibre_method="random")
                ftt = dt.EFTT(bases, tt, options)
            elif ftt_name == "FTT":
                ftt = dt.FTT(bases, tt)
            
            ftt.approximate(func)

            ys_l2_ftt = ftt(ls_l2).flatten()
            l2_error = torch.linalg.norm(ys_l2 - ys_l2_ftt) / torch.linalg.norm(ys_l2)

            if isinstance(ftt, dt.EFTT):
                num_eval = ftt.tt.num_eval + ftt.num_eval_fibres
            elif isinstance(ftt, dt.FTT):
                num_eval = ftt.tt.num_eval

            statistics[func_name][ftt_name]["l2_errors"].append(float(l2_error))
            statistics[func_name][ftt_name]["num_evals"].append(num_eval)
            statistics[func_name][ftt_name]["max_ranks"].append(float(ftt.tt.ranks.max()))
            if isinstance(ftt, dt.EFTT):
                max_basis = max(f.shape[1] for f in ftt.factors.values())
                statistics[func_name][ftt_name]["max_bases"].append(max_basis)

        num_evals = statistics[func_name][ftt_name]["num_evals"]
        l2_errors = statistics[func_name][ftt_name]["l2_errors"]
        log_l2_errors = [math.log(error) for error in l2_errors]
        max_ranks = statistics[func_name][ftt_name]["max_ranks"]

        print_name = func_name if ftt_name == "EFTT (ACA)" else ""

        diagnostics = [
            f"{print_name:12}",
            f"{ftt_name:12}",
            f"{geo_mean(l2_errors):8.1e}",
            f"{stdev(log_l2_errors):11.1e}",
            f"{round(mean(num_evals)):11}",
            f"{stdev(num_evals):9.1e}",
            f"{round(mean(max_ranks)):8}"
        ]
        if isinstance(ftt, dt.EFTT):
            max_bases = statistics[func_name][ftt_name]["max_bases"]
            diagnostics += [f"{round(mean(max_bases)):9}"]
        else:
            diagnostics += " "
        
        diagnostics = " & ".join(diagnostics) + r" \\"

        print(diagnostics)
    
    print(divider)