"""TODO: write docstring."""

from torch.linalg import norm

from examples.ou_process.setup_ou import *


zs = torch.rand((10_000, dim))

m = 8
indices_l = torch.arange(m)
indices_r = torch.arange(m, dim)

for poly in polys_dict:
    for method in options_dict:

        xs_cond_l = debug_x[:, indices_l]
        xs_cond_r = debug_x[:, indices_r]

        zs_cond_l = zs[:, indices_l]
        zs_cond_r = zs[:, indices_r]

        sirt: dt.TTSIRT = sirts[poly][method]
        
        if sirt.int_dir == dt.Direction.BACKWARD:
            sirt.marginalise(direction=dt.Direction.FORWARD)

        # if irts{i,j}.int_dir ~= 1, irts{i,j} = marginalise(irts{i,j}, 1); end
        ys_cond, fys_cond = sirt.eval_cirt(xs_cond_l, zs_cond_r)

        potential_ys_cond = model.eval_potential_cond(xs_cond_l, ys_cond, dir=dt.Direction.FORWARD)

        plt.scatter(torch.arange(10_000), torch.abs(torch.exp(-potential_ys_cond) - torch.exp(-fys_cond)))
        plt.show()




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