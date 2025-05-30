import deep_tensor as dt

from setup import *


xs = dirt.random(100_000)
neglogfxs_dirt = dirt.eval_potential(xs)
neglogfxs_exact = g.neglogpri(xs) + g.negloglik(xs)

dt.run_mcmc(xs, neglogfxs_dirt, neglogfxs_exact)