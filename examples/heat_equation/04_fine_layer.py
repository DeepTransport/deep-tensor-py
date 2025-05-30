from setup import *

dirt_crse = dt.SavedDIRT("dirt-coarse", preconditioner)

poly = dt.Lagrange1(num_elems=20)
dirt_options = dt.DIRTOptions()
tt_options = dt.TTOptions(tt_method="amen", init_rank=12, max_rank=16, max_cross=2)
# bridge = dt.SingleLayer()

dirt = dt.DIRT(
    negloglik, 
    neglogpri,
    preconditioner,
    poly, 
    # bridge=bridge,
    tt_options=tt_options,
    dirt_options=dirt_options
)

dirt.save("dirt-fine-multilayer")

