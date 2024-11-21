from typing import Callable

import numpy as np
from numpy.linalg import det, inv, norm


def maxvol(A: np.ndarray, delta=1e-2):
    """Implementation of the max-volume algorithm in Goreinov et al. 
    (2010).
    """

    n, r = A.shape 

    eye = np.eye(n)

    # Confirm that the initial submatrix is full rank
    if np.abs(det(A[:r])) < 1e-8:
        raise Exception("Initial submatrix is singular.")

    # Compute B
    B = A @ inv(A[:r])

    I = np.arange(n)

    while True:

        # Find entry of B with greatest absolute value
        i_max, j_max = np.unravel_index(np.argmax(np.abs(B), axis=None), B.shape)
        b_ij = B[i_max, j_max]

        # Check for convergence
        if np.abs(b_ij) < 1 + delta:
            return I[:r]
        
        # Update indices of max vol matrix
        # A[[i_max, j_max], :] = A[[j_max, i_max], :]
        I[[i_max, j_max]] = I[[j_max, i_max]]

        # Make rank 1 update to B matrix
        B1 = (B[r:][:, j_max] + eye[r:][:, i_max])[:, np.newaxis]
        B2 = (B[i_max, :] - eye[j_max, :r])[np.newaxis, :]

        B[r:] -= B1 @ B2 / b_ij


def low_rank_approx(
    A: np.ndarray, 
    r: int, 
    delta: float=1e-2
) -> np.ndarray:
    """The low-rank approximation algorithm for matrices (Algorithm 2 
    in Oseledets and Tyrtyshnikov, 2010).
    """

    n, m = A.shape

    # Initialisation
    J = np.arange(r)

    A_k = np.zeros((n, m))
    As = [A_k]

    while True:

        # Carry out column cycle
        R = A[:, J]
        Q, _ = np.linalg.qr(R)
        
        # Find max-volume submatrix in column R
        I = maxvol(Q)

        # Carry out row cycle
        C = A[I, :].T 
        Q, _ = np.linalg.qr(C)

        # Get indices of Q
        J = maxvol(Q)

        Q_hat = Q[J, :]

        A_k = A[:, J] @ (Q @ inv(Q_hat)).T
        As.append(A_k)

        if norm(As[-1] - As[-2]) <= delta * norm(As[-1]):
            return As[-1]


class Grid():
    """A set of univariate grids."""

    def __init__(self, grids: list[np.ndarray]):
        self.grids = grids 
        self.dims = np.array([len(grid) for grid in self.grids])
        self.ndim = len(grids)
    
    def get_xs(self, inds):
        xs = [self.grids[i][inds[i]] for i in range(self.ndim)]
        return np.array(xs)


class TensorTrain():

    def __init__(
        self, 
        grid: Grid, 
        cores: dict[np.ndarray], 
        ranks: dict[int]
    ):
        
        self.grid = grid
        self.cores = cores
        self.ranks = ranks
        self.ndim = len(cores.keys())
        self.dims = [core.shape[1] for core in cores] # TODO: tidy up
        return
    
    def __getitem__(self, inds):
        """Returns the element of a tensor at a specified index.
        """

        if len(inds) != self.ndim:
            msg = ("Length of indices does not match "
                   + "number of dimensions of tensor.")
            raise Exception(msg)

        # Extract the relevant set of TT cores
        cores = [self.cores[i][:, inds[i], :] for i in range(self.ndim)]

        # Take the product of the cores
        v = np.linalg.multi_dot(cores)
        return v.flat[0]

    def sample_irt(self, n):

        # Generate some uniform random variates
        unif_vars = np.random.sample((n, self.ndim))
        transformed_vars = np.zeros_like(unif_vars)

        # TODO: move this elsewhere (it can be precomputed before 
        # sampling)
        Ps = {}
        Ps[self.ndim] = np.array([1])

        for k in range(self.ndim, 0, -1):

            Ps[k] = np.zeros((self.ranks[k], self.ranks[k+1]))

            # Need to integrate the kth core over the grid
            for i in range(self.dims[k]-1):
                x0, x1 = self.grid.grids[k][i:i+2]
                dx = x1-x0 
                Ps[k] += (1 / dx) * 0.5 * (self.cores[k][:, i, :] + self.cores[k][:, i+1, :])
            
            Ps[k] = Ps[k] @ Ps[k+1]

            # Prepare grid intervals
            # dx_k = np.zeros(self.grid.dims[k])
            # dx_k[1:] = self.grid.grids[k][1:] - self.grid.grids[k][:-1]
            # Ps[k] = np.reshape(self.cores[k], (self.ranks[k]*self.dims[k], self.ranks[k+1]))
            # Ps[k] = Ps[k] @ Ps[k+1]
            # Ps[k] = np.reshape(Ps[k], (self.ranks[k], self.dims[k]))

        phis = {}
        psis = {}
        phis[0] = np.ones(n)

        for k in range(self.ndim):
            
            # Prepare deterministic part
            psis[k] = self.cores[k] @ Ps[k+1]

            for l in range(n):
                
                # Compute marginal pdf
                p_star = phis[k][l, :] @ 1 # TODO: finish




def tt_cross(
    func: Callable,
    grid: Grid,
    right_index_sets: list[list],
    rho: float=-1,  # Rank increasing parameter (TODO: figure out what this does)
    rel_stopping_tol: float=0.1, # TODO: implement this
    max_its: int=10
):
    r"""
    TT-Cross algorithm.

    Algorithm 1 from Dolgov et al (2020).

    Approximates the tensor that results from discretising a given 
    function on the tensor product of univariate grids.

    col_index_sets is a list of r_k indices that define (d-k) tuples 
    for each dimension k=0,1,2,\dots,d-1.

    Currently just fixed rank.
    TODO: add enrichment options?
    """

    # TT blocks (pi_hat)
    left_index_sets = {k: None for k in range(grid.ndim)}
    cores = {k: None for k in range(grid.ndim)}

    ranks = {k+1: len(ind_set) for k, ind_set in enumerate(right_index_sets)}
    ranks[0] = 1
    ranks[grid.ndim] = 1

    left_index_sets[0] = [[]]

    i = 0
    while i < max_its:  # TODO: add condition on norm of pdf

        # Forward iteration
        for k in range(grid.ndim-1):
            
            # Get indices (in format of original tensor) of rows of 
            # unfolding matrix
            row_inds = np.array([
                [*left_inds, j] 
                for left_inds in left_index_sets[k] 
                for j in range(grid.dims[k])
            ])

            # Extract the relevant set of rows and columns of unfolding 
            # matrix
            # TODO: should do some memoization to avoid computing the 
            # same elements of the tensor repeatedly
            submat = np.array([[
                func(grid.get_xs([*left_inds, j, *right_inds])) 
                for right_inds in right_index_sets[k]] 
                for left_inds in left_index_sets[k]
                for j in range(grid.dims[k])
            ])

            # Compute row index set using maxvol
            maxvol_inds = maxvol(submat)
            left_inds = row_inds[maxvol_inds]
            left_index_sets[k+1] = left_inds

            # Form the TT core
            core = submat @ inv(submat[maxvol_inds])
            cores[k] = np.reshape(core, (ranks[k], grid.dims[k], ranks[k+1]))
        
        # Form the final core
        k = grid.ndim-1

        submat = np.array([[
            func(grid.get_xs([*left_inds, j, *right_inds])) 
            for right_inds in right_index_sets[k]] 
            for left_inds in left_index_sets[k]
            for j in range(grid.dims[k])
        ])

        cores[k] = np.reshape(submat, (ranks[k], grid.dims[k], ranks[k+1]))

        # Backward iteration
        for k in range(grid.ndim-1, 0, -1):

            # Get indices (in format of original tensor) of columns of 
            # unfolding matrix
            col_inds = np.array([
                [j, *right_inds] 
                for j in range(grid.dims[k])
                for right_inds in right_index_sets[k] 
            ])

            # Extract the relevant set of rows and columns of unfolding 
            # matrix
            submat = np.array([[
                func(grid.get_xs([*left_inds, j, *right_inds])) 
                for j in range(grid.dims[k])
                for right_inds in right_index_sets[k]] 
                for left_inds in left_index_sets[k]
            ])

            # Compute col index set using maxvol
            maxvol_inds = maxvol(submat.T)
            right_index_sets[k-1] = col_inds[maxvol_inds]

            # Form new core
            core = inv(submat[:, maxvol_inds]) @ submat
            cores[k] = np.reshape(core, (ranks[k], grid.dims[k], ranks[k+1]))

        k = 0
        submat = np.array([[
            func(grid.get_xs([*left_inds, j, *right_inds])) 
            for j in range(grid.dims[k])
            for right_inds in right_index_sets[k]] 
            for left_inds in left_index_sets[k]
        ])

        cores[k] = np.reshape(submat, (ranks[k], grid.dims[k], ranks[k+1]))
        i += 1

    return TensorTrain(grid, cores, ranks)