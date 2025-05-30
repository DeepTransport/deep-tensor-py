import corner 
from matplotlib import pyplot as plt

import deep_tensor as dt

from setup import *
from plotting import *


# dirt = dt.SavedDIRT("dirt-coarse", preconditioner)

dirt_coarse = dt.SavedDIRT("dirt-coarse", preconditioner)
preconditioner_coarse = dt.DIRTPreconditioner(dirt_coarse)
dirt = dt.SavedDIRT("dirt-fine.h5", preconditioner_coarse)

samples_pri = dirt.reference.random(dirt.dim, 5000).numpy()[:, -10:]
samples_post = dirt.random(5000).numpy()[:, -10:] 
ms = m_true[-10:]

# plot_function(prior.transform(samples.mean(dim=0)), vmin=-9.0, vmax=-3.0)
# plt.show()

# for s in samples[:10]:
#     plot_function(prior.transform(s), vmin=-9.0, vmax=-3.0)
#     plt.show()

fig = corner.corner(samples_pri, color="k")
corner.corner(samples_post, fig=fig, color="tab:blue")
corner.overplot_lines(fig, xs=ms)

plt.savefig("marginals-fine-layer.pdf")