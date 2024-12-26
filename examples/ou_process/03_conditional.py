"""TODO: write docstring."""

from torch.linalg import norm

from examples.ou_process.setup_ou import *


directions = ["forward"]#, "backward"]

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

            if direction == "forward":
                xs_cond = debug_x[:, indices_l]
                zs_cond = zs[:, indices_r]
                if sirt.int_dir == dt.Direction.BACKWARD:
                    sirt.marginalise(dt.Direction.FORWARD) 
            else:
                xs_cond = debug_x[:, indices_r]
                zs_cond = zs[:, indices_l]
                if sirt.int_dir != dt.Direction.BACKWARD:
                    sirt.marginalise(dt.Direction.BACKWARD)

            ys_cond_sirt, neglogfys_cond_sirt = sirt.eval_cirt(xs_cond, zs_cond)

            neglogfys_cond_true = model.eval_potential_cond(
                xs_cond, 
                ys_cond_sirt, 
                dir=dt.Direction.FORWARD
            )

            fys_cond_sirt = torch.exp(-neglogfys_cond_sirt)
            fys_cond_true = torch.exp(-neglogfys_cond_true)
            approx_error = norm(fys_cond_true - fys_cond_sirt) 

            cov_cond_true = model.C[indices_r[:, None], indices_r[None, :]]
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




"""
z = rand(d, 1E4);
for i = 1:size(irts,1)
    for j = 1:min(2,size(irts,2))
        m = 8;
        ind1 = 1:m;
        ind2 = (m+1):d;
        xc1 = debug_x(ind1,:);
        xc2 = debug_x(ind2,:);
        % conditional sample
        zc1 = z(ind1, :);
        zc2 = z(ind2, :);
        %
        if irts{i,j}.int_dir ~= 1, irts{i,j} = marginalise(irts{i,j}, 1); end
        tic;[rc2,f] = eval_cirt(irts{i,j}, xc1, zc2);toc
        %
        figure;
        subplot(2,2,1);plot(abs(exp(-eval_potential_conditional(data, xc1, rc2, 1)) - exp(-f))); title('actual function value vs fft')
        subplot(2,2,2);plot(data.C(ind2, ind2) - cov(rc2')); title('actual covariance vs sample covariance')
        %
        disp(' ')
        disp(['approx eror: ' num2str(norm(exp(-eval_potential_conditional(data, xc1, rc2, 1)) - exp(-f)))])
        disp(['cov eror: ' num2str(norm(data.C(ind2, ind2) - cov(rc2'))/norm(data.C))])
        disp(' ')
        
        if irts{i,j}.int_dir ~= -1, irts{i,j} = marginalise(irts{i,j}, -1); end
        tic;[rc1,f] = eval_cirt(irts{i,j}, xc2, zc1);toc
        %
        subplot(2,2,3);plot(abs(exp(-eval_potential_conditional(data, rc1, xc2, -1)) - exp(-f))); title('actual function value vs fft')
        subplot(2,2,4);plot(data.C(ind1, ind1) - cov(rc1')); title('actual covariance vs sample covariance')
        disp(' ')
        disp(['approx eror: ' num2str(norm(exp(-eval_potential_conditional(data, rc1, xc2, -1)) - exp(-f)))])
        disp(['cov eror: ' num2str(norm(data.C(ind1, ind1) - cov(rc1'))/norm(data.C))])
        disp(' ')
    end
end"""