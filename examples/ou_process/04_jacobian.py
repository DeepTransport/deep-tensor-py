"""TODO: write docstring."""

from examples.ou_process.setup_ou import * 


zs = torch.rand((100, dim))

for poly in polys_dict:
    for method in options_dict:

        sirt: dt.TTSIRT = sirts[poly][method]
        sirt.debug_jac(zs, dt.Direction.FORWARD)
        # sirt.debug_jac_autodiff(zs, dt.Direction.FORWARD)