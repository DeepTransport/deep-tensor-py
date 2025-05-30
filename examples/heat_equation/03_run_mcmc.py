from setup import *

dirt = dt.SavedDIRT("dirt-coarse", preconditioner)
# preconditioner_coarse = dt.DIRTPreconditioner(dirt_coarse)
# dirt = dt.SavedDIRT("dirt-fine.h5", preconditioner_coarse)

xs = dirt.random(2_000)
neglogfxs_dirt = dirt.eval_potential(xs)
neglogfxs_exact = neglogpri(xs) + negloglik_coarse(xs)


print(neglogfxs_dirt)
print(neglogfxs_exact)

dt.run_mcmc(xs, neglogfxs_dirt, neglogfxs_exact)