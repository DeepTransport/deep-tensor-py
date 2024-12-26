"""TODO: write docstring."""

from torch.linalg import norm

from examples.ou_process.setup_ou import *

directions = ["forward", "backward"]

headers = [
    "Polynomial",
    "TT Method",
    "Direction",
    "Transform Error",
    "Density Error",
    "Approx. Error"
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

        for i, direction in enumerate(directions):

            sirt: dt.TTSIRT = sirts[poly][method]

            if direction == "forward":
                # Forward marginalisation
                indices = torch.arange(8)
                if sirt.int_dir != dt.Direction.FORWARD:
                    sirt.marginalise(dt.Direction.FORWARD) 
            else:
                # Backward marginalisation
                indices = torch.arange(dim-1, 14, -1)
                if sirt.int_dir != dt.Direction.BACKWARD:
                    sirt.marginalise(dt.Direction.BACKWARD)

            xs, potential_xs = sirt.eval_irt_nograd(zs[:, indices])
            fxs = sirt.eval_pdf(xs)  # TODO: fix the bug in here...
            z0 = sirt.eval_rt(xs)
            fe = model.eval_potential_marginal(indices, xs)

            transform_error = norm(zs[:, indices] -z0, ord="fro")
            density_error = norm(torch.exp(-potential_xs) - fxs)
            approx_error = norm(torch.exp(-potential_xs) - torch.exp(-fe))
            info = [
                f"{poly:16}",
                f"{method:16}",
                f"{direction:16}",
                f"{transform_error:=16.5e}",
                f"{density_error:=16.5e}",
                f"{approx_error:=16.5e}"
            ]
            print(" | ".join(info))

            axes[i][0].scatter(torch.arange(10_000), torch.abs(fe - potential_xs), s=4)
            axes[i][0].set_ylim(bottom=0.0)
            axes[i][0].set_ylabel("Error")
            axes[i][0].set_title("Error in potential function")

            axes[i][1].scatter(fe, potential_xs, s=4)
            axes[i][1].set_xlabel("True potential")
            axes[i][1].set_ylabel("FTT")
            axes[i][1].set_title("True potential vs FTT")
            
        plt.savefig(f"examples/ou_process/figures/02_marginal_{poly}_{method}.pdf")

"""
% should test ind = 1, ind = 1:(d-1) for > 0
% should test ind = d, ind = 2:d for < 0
% sample
z = rand(d, 1E4);
for i = 1:size(irts,1)
    for j = 1:size(irts,2)
        figure;
        % test 1
        ind  = 1:8;
        if irts{i,j}.int_dir ~= 1, irts{i,j} = marginalise(irts{i,j}, 1); end
        tic;[r,p] = eval_irt(irts{i,j}, z(ind,:));toc
        fx = eval_pdf(irts{i,j}, r);
        tic;z0 = eval_rt(irts{i,j}, r);toc
        disp(' ')
        disp(['transform eror: ' num2str(norm(z(ind,:) - z0))])
        disp(['density eror: ' num2str(norm(exp(-p) - fx))])
        fe   = eval_potential_marginal(data, ind, r);
        disp(['approx eror: ' num2str(norm(exp(-p) - exp(-fe)))])
        disp(' ')
        %
        subplot(2,3,1);plot(abs(fe - p)/max(abs(fe)), '.');
        subplot(2,3,2);plot(fe , p, '.');
        title('actual poential function value vs fft')
        subplot(2,3,3);plot(data.C(ind, ind) - cov(r'))
        
        % test 2
        ind  = d:-1:15;
        if irts{i,j}.int_dir ~= -1, irts{i,j} = marginalise(irts{i,j}, -1); end
        tic;[r,p] = eval_irt(irts{i,j}, z(ind,:));toc
        fx = eval_pdf(irts{i,j}, r);
        tic;z0 = eval_rt(irts{i,j}, r);toc
        disp(' ')
        disp(['transform eror: ' num2str(norm(z(ind,:) - z0))])
        disp(['density eror: ' num2str(norm(exp(-p) - fx))])
        %
        fe   = eval_potential_marginal(data, ind, r);
        disp(['approx eror: ' num2str(norm(exp(-p) - exp(-fe)))])
        disp(' ')
        subplot(2,3,4);plot(abs(fe - p)/max(abs(fe)), '.');
        subplot(2,3,5);plot(fe , p, '.');
        title('actual potential function value vs fft')
        subplot(2,3,6);plot(data.C(ind, ind) - cov(r'))
    end
end"""