from setup import *

import deep_tensor as dt


if __name__ == "__main__":

    poly = dt.Lagrange1(num_elems=20)
    dirt_options = dt.DIRTOptions(method="eratio")
    tt_options = dt.TTOptions(tt_method="amen", init_rank=12, max_rank=16, max_cross=1)
    bridge = dt.Tempering()

    dirt_crse = dt.DIRT(
        negloglik_coarse, 
        neglogpri,
        preconditioner,
        poly, 
        bridge=bridge,
        tt_options=tt_options,
        dirt_options=dirt_options
    )

    dirt_crse.save("dirt-coarse-eratio")