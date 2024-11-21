
def tt_cross(
    rel_stopping_tol: float=0.1,
    max_its: int=10
):
    """
    TT-Cross algorithm.

    From Dolgov et al, 2020.

    Approximates the tensor that results from discretising a given PDF 
    on the tensor product of univariate grids.
    """

    d = -1  # Number of dimensions of tensor

    i = 0
    while i < max_its: # Add condition on norm of pi

        for k in range(d):
            
            # (Optional) prepare enrichment set
            # Compute unfolding matrix
            # Compute new index set using maxvol algorithm and truncate
            pass

        for k in range(d-1, -1, -1):

            # (Optional) prepare enrichment set
            # Compute unfolding matrix
            # Compute new index set using maxvol algorithm and truncate
            pass

        i += 1

        pass 

    return