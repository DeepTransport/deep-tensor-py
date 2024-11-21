import torch

from deep_tensor.polynomials import Chebyshev1st, Chebyshev1stCDF


tau = 1e-2 # I think this is gamma in the papers(?)

# Returns product of exp(-potential) * (weight func)
def pdf_1(x):
    return torch.exp(-8 * (x - 0.2)**2)

def pdf_2(x):
    return torch.exp(-5 * torch.abs(x - 0.2))

# poly = Chebyshev2nd(20);
# poly_cdf = Chebyshev2ndCDF(poly);

poly = Chebyshev1st(order=20)
poly_cdf = Chebyshev1stCDF(poly=poly)

xs = torch.linspace(-1.0+1e-2, 1.0-1e-2, 10_000)

A, dA = poly_cdf.eval_int_basis_newton(xs)

weight = 0.5

fp = poly.node2basis @ (pdf_2(poly.nodes) / poly.eval_measure(poly.nodes))

pp = poly.eval_radon(fp, poly_cdf.nodes)

xs = torch.linspace(*poly.domain, 10_000)

# pdf and cdf?
fi = poly.eval(fp, xs)  # approximation of potential function
Fi = poly_cdf.eval_cdf(pp+tau, xs)

# Evaluate the CDF at each of the nodes
cdf_nodes = poly_cdf.cdf_basis2node @ (poly_cdf.node2basis @ pp)

# Normalise into the range [0, 1]
cdf_nodes = cdf_nodes - cdf_nodes[0]
cdf_nodes = cdf_nodes / cdf_nodes[-1]

# from matplotlib import pyplot as plt

# plt.plot(xs, fi)
# plt.plot(xs, pdf_2(xs))
# plt.plot(xs, Fi) 
# plt.plot(poly_cdf.sampling_nodes, cdf_nodes)

# plt.ylim((-0.1, 1.1))
# plt.show()

fp = poly.node2basis @ (pdf_2(poly.nodes) / poly.eval_measure(poly.nodes) ** 0.5)
pp = poly.eval_radon(fp, poly_cdf.nodes) ** 2

cdf_nodes = poly_cdf.cdf_basis2node @ (poly_cdf.node2basis @ pp)
cdf_nodes = cdf_nodes - cdf_nodes[0]
cdf_nodes = cdf_nodes / cdf_nodes[-1]

xs = torch.linspace(*poly.domain, 1_000)
fi = poly.eval_radon(fp, xs) ** 2 * poly.eval_measure(xs)
Fi = poly_cdf.eval_cdf(pp+tau, xs)

z = torch.rand((int(5e+5), 1))

r = poly_cdf.invert_cdf(pp+tau, z)

print(r)

# fp = poly.node2basis*( (pdf2(poly.nodes)./eval_measure(poly, poly.nodes)).^0.5 );
# pp = eval_radon(poly, fp, poly_cdf.nodes).^2;
# %
# cdf_nodes = poly_cdf.cdf_basis2node*(poly_cdf.node2basis*pp);
# cdf_nodes = cdf_nodes - cdf_nodes(1,:);
# cdf_nodes = cdf_nodes./cdf_nodes(end,:);

# xs = linspace(poly.domain(1), poly.domain(2), 1000);
# fi = eval_radon(poly, fp, xs).^2.*eval_measure(poly, xs(:));
# Fi = eval_cdf(poly_cdf, pp+tau, xs);


# z = rand(5E5,1);
# tic;r = invert_cdf(poly_cdf, pp+tau, z);toc
# tic; norm(eval_cdf(poly_cdf, pp+tau, r)-z), toc

# figure
# histogram(r, 'Normalization', 'pdf')
# hold on
# plot(xs, fi, xs, pdf2(xs))
# plot(xs, Fi, poly_cdf.sampling_nodes, cdf_nodes, 'o')
# set(gca, 'xlim', poly.domain)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# % multi cdf test
# n = 5E5;

# %%%%%%

# fp = poly.node2basis*( (pdf2(poly.nodes)./eval_measure(poly, poly.nodes)).^(0.5) );
# pp = eval_radon(poly, fp, poly_cdf.nodes).^2;
# pp = repmat(pp, 1, n);

# cdf_nodes = poly_cdf.cdf_basis2node*(poly_cdf.node2basis*pp);
# cdf_nodes = cdf_nodes - cdf_nodes(1,:);
# cdf_nodes = cdf_nodes./cdf_nodes(end,:);

# xs = linspace(poly.domain(1), poly.domain(2), n);
# fi = eval_radon(poly, fp, xs).^2.*eval_measure(poly, xs(:));
# Fi = eval_cdf(poly_cdf, pp+tau, xs);

# z = rand(n,1);
# taus = rand(1,n)*tau;
# tic;r = invert_cdf(poly_cdf, pp+tau, z);toc
# tic; norm(eval_cdf(poly_cdf, pp+tau, r)-z), toc


# figure
# histogram(r, 'Normalization', 'pdf')
# hold on
# plot(xs, fi, xs, pdf2(xs))
# hold on
# plot(xs, Fi, poly_cdf.sampling_nodes, cdf_nodes(:,1), 's')
# set(gca, 'xlim', poly.domain)