from typing import Callable, Dict, Tuple
import warnings

import torch
from torch import Tensor
from torch import linalg
from torch.distributions import Categorical

from .directions import Direction, REVERSE_DIRECTIONS
from ..interpolation import deim, maxvol
from ..linalg import (
    cartesian_prod, 
    fold_left, fold_right, 
    unfold_left, unfold_right,
    tsvd
)
from ..options import TTOptions
from ..tools.printing import als_info


INTERPOLATION_METHODS = {"deim": deim, "maxvol": maxvol}
MAX_CONDITION_NUM = 1.0e+5


class Grid():

    def __init__(
        self, 
        points: Dict[int, Tensor], 
        weights: Dict[int, Tensor] | None = None
    ):

        self.points = points 
        self.indices = {k: torch.arange(points[k].numel()) for k in self.points}
        self.dim = len(points.keys())

        if weights is None:
            weights = {k: torch.ones_like(points[k]) for k in self.points}

        self.weights = weights  # unnormalised
        self.point_densities = {k: Categorical(self.weights[k]) for k in self.weights}

        return 
    
    def sample_indices(self, n: int) -> Tensor:
        """Returns a sample of indices. indices are chosen 
        proportionally to their weights..
        
        TODO: ideally it wouldn't be possible to have the same sample 
        multiple times...
        """

        sample = torch.vstack([
            self.point_densities[k].sample((n,))
            for k in range(self.dim)
        ]).T

        return sample
    
    def indices2points(self, inds: Tensor) -> Tensor:
        """Converts a tensor of indices to the corresponding points."""
        
        points = torch.vstack([
            self.points[k][inds_k]
            for k, inds_k in enumerate(inds.T)
        ]).T

        return points


class TT():
    """Computes a tensor train factorisation of the discretisation of 
    an arbitrary function on a tensor-product grid using the 
    alternating cross approximation algorithm. 

    user could possibly call this from their own custom ftt 
    implementation.
    """

    def __init__(
        self, 
        target_func: Callable[[Tensor], Tensor],
        grid: Grid,
        options: TTOptions
    ):

        self.target_func = target_func
        self.grid = grid 
        self.dim = grid.dim
        self.indices = grid.indices 
        self.points = grid.points
        self.options = options

        self.errors = torch.zeros(self.dim)
        self.num_eval = 0

        self.direction = Direction.FORWARD
        self.index_sets: Dict[int, Tensor] = {}
        self.cores: Dict[int, Tensor] = {}
        
        return
    
    @property
    def ranks(self) -> Tensor:
        """The ranks of the tensor cores (excluding rank 0 and rank d).
        """
        ranks = torch.tensor([self.cores[k].shape[2] 
                              for k in range(self.dim-1)])
        return ranks
    
    @staticmethod
    def _get_error_local(H_new: Tensor, H_old: Tensor) -> float:
        """Returns the error between the current and previous 
        coefficient tensors.
        """
        return float((H_new-H_old).abs().max() / H_new.abs().max())  

    def _initialise(self) -> None:
        """Initialises the cores and interpolation points in each 
        dimension.
        """

        for k in range(self.dim):

            core_shape = (
                1 if k == 0 else self.options.init_rank, 
                self.indices[k].numel(),
                1 if k == self.dim-1 else self.options.init_rank
            )
            self.cores[k] = torch.zeros(core_shape)

            inds_sample = self.grid.sample_indices(self.options.init_rank)
            self.index_sets[k] = inds_sample[:, k:]

        self.index_sets[-1] = torch.tensor([])
        self.index_sets[self.dim] = torch.tensor([])
        return
    
    def _reverse_direction(self) -> None:
        """Reverses the direction in which the dimensions of the 
        function are iterated over.
        """
        self.direction = REVERSE_DIRECTIONS[self.direction]
        return
    
    def _get_local_index(
        self,
        index_set_prev: Tensor,
        indices_k: Tensor,
        indices_global: Tensor
    ) -> Tensor:
        """Updates the set of interpolation points for the current 
        dimension.
        
        Parameters
        ----------
        basis:
            The polynomial basis for the current dimension of the 
            approximation.
        ls_int_p: 
            The previous set of interpolation points.
        inds:
            The set of indices of the maximum-volume submatrix of the 
            current (unfolded) tensor core.
        
        Returns
        -------
        ls_int_k:
            The set of updated interpolation points for the current 
            dimension.
        
        """

        if index_set_prev.numel() == 0:
            index_set_k = indices_k[indices_global][:, None]
            return index_set_k

        n_k = indices_k.numel()

        # TODO: the naming could be improved here...
        index_set_prev = index_set_prev[indices_global // n_k].clone()
        index_set_k = indices_k[indices_global % n_k][:, None]

        if self.direction == Direction.FORWARD:
            index_set_k = torch.hstack((index_set_prev, index_set_k))
        else:
            index_set_k = torch.hstack((index_set_k, index_set_prev))

        return index_set_k
 
    def _select_points(self, U: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Selects a square submatrix within a tall matrix.

        Parameters
        ----------
        U:
            A tall matrix.
        
        Returns
        -------
        inds:
            The set of row indices of U corresponding to the selected 
            submatrix.
        B:
            The product of U and the inverse of the selected submatrix, 
            UU[I, :]^{-1}.
        U_sub:
            The selected submatrix, U[I, :].
        
        """
        inds, B = INTERPOLATION_METHODS[self.options.int_method](U)
        U_sub = U[inds]
        if (cond := linalg.cond(U_sub)) > MAX_CONDITION_NUM:
            msg = f"Poor condition number in interpolation: {cond}."
            warnings.warn(msg)
        return inds, B, U_sub

    def build_cross_block_fixed(self, k: int) -> None:
        
        H = self.compute_block(self.index_sets[k-1], self.index_sets[k+1], k) 
        self.errors[k] = TT._get_error_local(H, self.cores[k])
        self._build_basis_svd(H, k)
        return

    def build_block_final(self) -> None:
        """Computes the final block of the FTT approximation to the 
        target function.
        """

        if self.direction == Direction.FORWARD:
            k = self.dim - 1
        else:
            k = 0

        H = self.compute_block(self.index_sets[k-1], self.index_sets[k+1], k)
        self.errors[k] = TT._get_error_local(H, self.cores[k])
        self.cores[k] = H
        return
       
    def _truncate_local(
        self, 
        H: Tensor, 
        tol: float | None = None,
        max_rank: int | None = None
    ) -> Tuple[Tensor, Tensor, int]:
        """Computes the truncated SVD for a given tensor block.

        Parameters
        ----------
        H:
            The unfolding matrix of evaluations of the target function 
            evaluated at a set of interpolation points.
        tol:
            The error tolerance used when truncating the singular 
            values.
        
        Returns
        -------
        Ur:
            Matrix containing the left singular vectors of F after 
            truncation.
        sVhr: 
            Matrix containing the transpose of the product of the 
            singular values and the right-hand singular vectors after
            truncation. 
        rank:
            The number of singular values of H that were retained.

        """
        if tol is None: 
            tol = self.options.local_tol
        if max_rank is None:
            max_rank = self.options.max_rank
        Ur, sr, Vhr, rank = tsvd(H, tol, max_rank)
        sVhr = sr[:, None] * Vhr
        return Ur, sVhr, rank
    
    def _build_basis_svd(
        self, 
        H: Tensor, 
        k: int, 
        tol: float | None = None,
        max_rank: int | None = None
    ) -> None:
        """Computes the coefficients of the kth tensor core.
        
        Parameters
        ----------
        H:
            An r_{k-1} * n_{k} * r_{k} tensor containing the 
            coefficients of the kth TT block.
        k:
            The index of the dimension corresponding to the basis 
            being constructed.
        tol:
            The tolerance to use when applying truncated SVD to the 
            unfolding matrix of H.
        max_rank:
            The maximum number of singular values to retain when 
            applying truncated SVD to the unfolding matrix of H.

        Returns
        -------
        None
            
        """

        k_prev = k - self.direction.value
        k_next = k + self.direction.value
        r_p, n_k, r_k = H.shape
        
        index_set_prev = self.index_sets[k_prev]
        A_next = self.cores[k_next]

        if self.direction == Direction.FORWARD:
            H = unfold_left(H)
        else: 
            H = unfold_right(H)

        # tol = 0.0  # TEMP!!
        U, sVh, rank = self._truncate_local(H, tol, max_rank)

        # Select a set of interpolation points
        indices_global, B, U_interp = self._select_points(U)
        index_set_k = self._get_local_index(index_set_prev, self.indices[k], indices_global)
        couple = U_interp @ sVh

        # Form the current coefficient tensor and update the next one
        if self.direction == Direction.FORWARD:
            A = fold_left(B, (r_p, n_k, rank))
            r_next = A_next.shape[0]
            A_next = torch.einsum("il, ljk", couple[:, :r_next], A_next)
        else:
            A = fold_right(B, (rank, n_k, r_k))
            r_next = A_next.shape[2]
            A_next = torch.einsum("ijl, kl", A_next, couple[:, :r_next])

        self.cores[k] = A
        self.cores[k_next] = A_next
        self.index_sets[k] = index_set_k
        return

    def compute_block(
        self, 
        inds_left: Tensor, 
        inds_right: Tensor, 
        k: int
    ) -> Tensor:
        """Evaluates the function being approximated at a (reduced) set 
        of interpolation points, and returns the corresponding
        local coefficient matrix.
        """

        r_p = 1 if inds_left.numel() == 0 else inds_left.shape[0]
        r_k = 1 if inds_right.numel() == 0 else inds_right.shape[0]
        n_k = self.points[k].numel()

        inds = cartesian_prod(inds_left, self.indices[k][:, None], inds_right)
        ls = self.grid.indices2points(inds)
        H = self.target_func(ls).reshape(r_p, n_k, r_k)
        self.num_eval += H.numel()

        return H
    
    def round(
        self, 
        tol: float | None = None, 
        max_rank: int | None = None
    ) -> None:
        """Rounds the TT cores. Applies double rounding to get back 
        to the starting direction.

        Parameters
        ----------
        tol:
            The tolerance to use when applying truncated SVD to round 
            each core.
        
        """

        if tol is None:
            tol = self.options.local_tol

        for _ in range(2):
            
            self._reverse_direction()

            if self.direction == Direction.FORWARD:
                inds = range(self.dim-1)
            else:
                inds = range(self.dim-1, 0, -1)

            for k in inds:
                self._build_basis_svd(self.cores[k], k, tol, max_rank)

        # if self.use_amen:
        #     self.tt_data.res_w = {}
        #     self.tt_data.res_x = {}
        return

    def sweep(self):
        """Runs a single cross iteration.
        NOTE: start this without any adaptivity (then add enrichment in later).
        """

        if self.cores == {}:
            self._initialise()
        else:
            self._reverse_direction()
        
        # if self.use_amen:
        #     self._initialise_amen()

        if self.direction == Direction.FORWARD:
            inds = range(self.dim-1)
        else:
            inds = range(self.dim-1, 0, -1)
        
        for i, k in enumerate(inds):
            
            if self.options.verbose > 1:
                msg = f"Building block {i+1} / {self.dim}..."
                als_info(msg, end="\r")
            
            # TODO: support enrichment methods...
            self.build_cross_block_fixed(k)
            # if self.options.tt_method == "fixed_rank":
            #     self._compute_cross_block_fixed(k)
            # elif self.options.tt_method == "random":
            #     self._compute_cross_block_random(k)
            # elif self.options.tt_method == "amen":
            #     self._compute_cross_block_amen(k)
        
        if self.options.verbose > 1:
            msg = f"Building block {self.dim} / {self.dim}..."
            als_info(msg, end="\r")
        self.build_block_final()

        return

