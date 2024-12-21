"""TODO: write docstring."""

from examples.ou_process.setup_ou import * 


zs = torch.rand((10_000, d))
xs, potential_xs = sirt.eval_irt_nograd(zs)
z0 = sirt.eval_rt(xs)

transform_error = torch.linalg.norm(zs-z0, ord="fro")
potential_error = torch.linalg.norm(potential_func(xs) - potential_xs)
pdf_error = torch.linalg.norm(torch.exp(-potential_func(xs)) - torch.exp(-potential_xs))

# print((zs-z0)[:2, :])

print(f"Transform error: {transform_error}")
print(f"Potential error: {potential_error}")
print(f"PDF error: {pdf_error}")

#disp(['cov eror: ' num2str(norm(data.C - cov(r'))/norm(data.C))])

plt.scatter(potential_func(xs), potential_xs, s=10)
plt.xlabel("Potential")
plt.ylabel("FTT")
plt.title("Actual potential function vs FTT")
plt.show()