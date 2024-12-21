"""TODO: write docstring."""

from torch.linalg import norm

from examples.ou_process.setup_ou import * 


for i in range(len(bases_list)):
    for j in range(len(options_list)):

        zs = torch.rand((10_000, dim))
        xs, potential_xs = sirts[i][j].eval_irt_nograd(zs)
        z0 = sirts[i][j].eval_rt(xs)

        transform_error = norm(zs-z0, ord="fro")
        potential_error = norm(potential_func(xs) - potential_xs)
        pdf_error = norm(
            torch.exp(-potential_func(xs))
            - torch.exp(-potential_xs)
        )
        cov_error = norm(model.C - torch.cov(xs.T)) / norm(model.C)

        print(f"Polynomial {i}, option {j}:")
        print(f" - Transform error: {transform_error}.")
        print(f" - Potential error: {potential_error}.")
        print(f" - PDF error: {pdf_error}.")
        print(f" - Covariance error: {cov_error}.")

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        for ax in axes:
            ax.set_box_aspect(1)

        axes[0].hist(torch.abs(potential_func(xs) - potential_xs))
        axes[0].set_xlabel("Error")
        axes[0].set_title("Error in potential function")

        axes[1].scatter(potential_func(xs), potential_xs, s=10)
        axes[1].set_xlabel("True potential")
        axes[1].set_ylabel("FTT")
        axes[1].set_title("True potential vs FTT")
        
        plt.savefig(f"examples/ou_process/figures/01_gaussian_{i}{j}.pdf")