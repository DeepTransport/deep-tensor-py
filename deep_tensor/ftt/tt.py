from __future__ import annotations

from typing import Callable, Dict, Tuple
import warnings

import torch
from torch import Tensor
from torch import linalg
from torch.distributions import Categorical

from .directions import Direction, REVERSE_DIRECTIONS
from .tt_options import TTOptions
from ..interpolation import deim, maxvol
from ..linalg import (
    cartesian_prod, 
    fold_left, fold_right,
    unfold_left, unfold_right, 
    mode_n_folding, mode_n_unfolding, n_mode_prod,
    tsvd
)
from ..tools.printing import als_info


INTERPOLATION_METHODS = {"deim": deim, "maxvol": maxvol}
MAX_CONDITION_NUM = 1e+05


class Grid():

    def __init__(
        self, 
        points: Dict[int, Tensor], 
        weights: Dict[int, Tensor] | None = None
    ):
        
        if weights is None:
            weights = {k: torch.ones_like(points[k]) for k in points}
        
        self._points = points 
        self._indices = {k: torch.arange(points[k].numel()) for k in points}
        self._dim = len(points.keys())
        self._weights = weights  # unnormalised
        self._point_densities = {
            k: Categorical(self._weights[k]) for k in self._weights
        }
        return 
    
    def _sample_indices(self, n: int) -> Tensor:
        """Returns a sample of indices, where indices are chosen 
        proportionally to their weights. 

        Parameters
        ----------
        n: 
            The number of samples to generate.

        Returns
        -------
        sample:
            An n * d matrix where each row contains a set of sampled 
            indices.
        
        """
        # TODO: ideally it wouldn't be possible to have the same sample 
        # multiple times.

        sample = torch.vstack([
            self._point_densities[k].sample((n,)).to(device=self._points[0].device)
            for k in range(self._dim)
        ]).T

        return sample
    
    def _indices2points(self, inds: Tensor) -> Tensor:
        """Converts a tensor of indices to the corresponding 
        interpolation points.
        """
        points = torch.vstack([
            self._points[k][inds_k]
            for k, inds_k in enumerate(inds.T)
        ]).T
        return points


class TT():
    r"""A tensor train factorisation.
    
    This class computes and stores a tensor train factorisation of the 
    discretisation of an arbitrary function on a tensor-product grid, 
    using the alternating cross approximation algorithm outlined in 
    @Oseledets2010. 

    Parameters
    ----------
    options:
        Parameters which control the construction of the tensor train 
        factorisation.
    device:
        The device to use when carrying out computations.

    Attributes
    ----------
    ranks:
        The core ranks $\{r_{k}\}_{k=1}^{d-1}$ (note that $r_{0}=r_{d}=1$).

    """

    def __init__(
        self, 
        options: TTOptions | None = None,
        device: torch.device = torch.get_default_device()
    ):
        if options is None:
            options = TTOptions()
        self._options = options
        self._num_eval = 0
        self._direction = Direction.FORWARD
        self._index_sets: Dict[int, Tensor] = {}
        self._cores: Dict[int, Tensor] = {}
        # AMEn
        self._res_l = {}
        self._res_w = {}
        self._device = device
        return
    
    @property
    def ranks(self) -> Tensor:
        """The ranks of the tensor cores (excluding rank 0 and rank d)."""
        ranks = [self._cores[k].shape[2] for k in range(self._dim-1)]
        return torch.tensor(ranks, device=self._device)
    
    @staticmethod
    def _compute_local_error(H_new: Tensor, H_old: Tensor) -> float:
        """Returns the error between the current and previous 
        coefficient tensors.
        """
        return float((H_new-H_old).abs().max() / H_new.abs().max())  

    def initialise(
        self, 
        target_func: Callable[[Tensor], Tensor], 
        grid: Grid
    ) -> None:
        r"""Initialises the TT.
        
        Parameters
        ----------
        target_func:
            A function that takes an $n \times d$ matrix with rows 
            containing samples from the tensor product grid, and 
            returns an $n$-dimensional vector containing the values of 
            the target function evaluated at each sample.
        grid:
            The tensor product grid used to construct the TT.

        """
        self._target_func = target_func
        self._grid = grid 
        self._dim = grid._dim
        self._indices = grid._indices 
        self._points = grid._points
        self._errors = torch.zeros(self._dim, device=self._device)
        if self._cores != {}:
            self.round(max_rank=self._options.init_rank)
        return

    def _initialise_cores(self) -> None:
        """Initialises the cores and interpolation points in each 
        dimension.
        """

        for k in range(self._dim):
            core_shape = (
                1 if k == 0 else self._options.init_rank, 
                self._indices[k].numel(),
                1 if k == self._dim-1 else self._options.init_rank
            )
            self._cores[k] = torch.zeros(core_shape, device=self._device)
            inds_sample = self._grid._sample_indices(self._options.init_rank)
            self._index_sets[k] = inds_sample[:, k:]

        self._index_sets[-1] = torch.tensor([], device=self._device)
        self._index_sets[self._dim] = torch.tensor([], device=self._device)
        return
    
    def _initialise_res_l(self) -> None:
        """Initialises the residual coordinates for AMEn."""

        for k in range(self._dim):
            samples = self._grid._sample_indices(self._options.kick_rank)
            if self._direction == Direction.FORWARD:
                self._res_l[k] = samples[:, k:]
            else:
                self._res_l[k] = samples[:, :(k+1)]

        self._res_l[-1] = torch.tensor([], device=self._device)
        self._res_l[self._dim] = torch.tensor([], device=self._device)
        return
    
    def _initialise_res_w(self) -> None:
        """Initialises the residual blocks for AMEn."""

        kick_rank = self._options.kick_rank

        if self._direction == Direction.FORWARD:
            
            shape_0 = (kick_rank, self._cores[0].shape[-1])
            self._res_w[0] = torch.ones(shape_0, device=self._device)
            
            for k in range(1, self._dim):
                shape_k = (self._cores[k].shape[0], kick_rank)
                self._res_w[k] = torch.ones(shape_k, device=self._device)

        else:

            for k in range(self._dim-1):
                shape_k = (kick_rank, self._cores[k].shape[-1])
                self._res_w[k] = torch.ones(shape_k, device=self._device)

            shape_d = (self._cores[self._dim-1].shape[0], kick_rank)
            self._res_w[self._dim-1] = torch.ones(shape_d, device=self._device)

        self._res_w[-1] = torch.tensor([[1.0]], device=self._device)
        self._res_w[self._dim] = torch.tensor([[1.0]], device=self._device)
        return

    def _initialise_amen(self) -> None:
        """Initialises the residual coordinates and residual blocks 
        for AMEn.
        """
        if self._res_l == {}:
            self._initialise_res_l()
        if self._res_w == {}:
            self._initialise_res_w()
        return
    
    def _reverse_direction(self) -> None:
        """Reverses the direction in which the dimensions of the 
        function are iterated over.
        """
        self._direction = REVERSE_DIRECTIONS[self._direction]
        return
    
    def _merge_indices(
        self,
        index_set_prev: Tensor,
        indices_k: Tensor,
        indices_global: Tensor
    ) -> Tensor:
        """Updates the set of interpolation points for the current 
        dimension.
        """

        if index_set_prev.numel() == 0:
            index_set_k = indices_k[indices_global][:, None]
            return index_set_k

        n_k = indices_k.numel()

        # TODO: the naming could be improved here...
        index_set_prev = index_set_prev[indices_global // n_k].clone()
        index_set_k = indices_k[indices_global % n_k][:, None]

        if self._direction == Direction.FORWARD:
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
        inds, B = INTERPOLATION_METHODS[self._options.int_method](U)
        U_sub = U[inds]
        if (cond := linalg.cond(U_sub)) > MAX_CONDITION_NUM:
            msg = f"Poor condition number in interpolation: {cond}."
            warnings.warn(msg)
        return inds, B, U_sub
    
    @staticmethod
    def _check_block(block: Tensor) -> None:
        if block.max() == 0.0:
            msg = (
                "The entries of the current TT core are uniformly zero "
                "(possibly the result of underflow). Consider rescaling "
                "the target function. If you are confident the target "
                "function is scaled appropriately, consider using a "
                "refined grid, larger core ranks, or an increased "
                "number of bridging densities."
            )
            warnings.warn(msg)
        return 

    def _compute_block(
        self, 
        index_set_left: Tensor, 
        inds_k: Tensor,
        index_set_right: Tensor
    ) -> Tensor:
        """Evaluates the function being approximated at a (reduced) set 
        of interpolation points, and returns the corresponding
        local coefficient matrix.
        """

        r_p = 1 if index_set_left.numel() == 0 else index_set_left.shape[0]
        r_k = 1 if index_set_right.numel() == 0 else index_set_right.shape[0]
        n_k = inds_k.numel()

        indices_prod = cartesian_prod(index_set_left, inds_k, index_set_right)

        ls = self._grid._indices2points(indices_prod)
        block = self._target_func(ls).reshape(r_p, n_k, r_k)
        self._num_eval += block.numel()

        self._check_block(block)
        return block
    
    def _build_core_amen(self, k: int) -> None:
        
        index_set_left = self._index_sets[k-1]
        inds = self._indices[k][:, None]
        index_set_right = self._index_sets[k+1]
        r_left = self._res_l[k-1]
        r_right = self._res_l[k+1]

        # Evaluate the interpolant function at l_k nodes
        H = self._compute_block(index_set_left, inds, index_set_right)
        self._errors[k] = self._compute_local_error(H, self._cores[k])

        # Evaluate the residual function at r_k nodes
        H_res = self._compute_block(r_left, inds, r_right)

        if self._direction == Direction.FORWARD and k > 0:
            H_up = self._compute_block(index_set_left, inds, r_right)
        elif self._direction == Direction.BACKWARD and k < self._dim-1: 
            H_up = self._compute_block(r_left, inds, index_set_right)
        else:
            H_up = H_res.clone()

        self._build_basis_amen(H, H_res, H_up, k)
        return 

    def _build_core_random(self, k: int) -> None:

        index_set_left = self._index_sets[k-1]
        inds = self._indices[k][:, None]
        index_set_right = self._index_sets[k+1]
        index_set_enrich = self._grid._sample_indices(n=self._options.kick_rank)
        
        block = self._compute_block(index_set_left, inds, index_set_right) 
        self._errors[k] = TT._compute_local_error(block, self._cores[k])

        if self._direction == Direction.FORWARD:
            block_indices = (index_set_left, inds, index_set_enrich[:, k+1:])
            block_enrich = self._compute_block(*block_indices)
            block = torch.concatenate((block, block_enrich), dim=2)
        else:
            block_indices = (index_set_enrich[:, :k], inds, index_set_right)
            block_enrich = self._compute_block(*block_indices)
            block = torch.concatenate((block, block_enrich), dim=0)
        
        self._build_basis_svd(block, k)

        return
    
    def _build_core_fixed(self, k: int) -> None:

        index_set_left = self._index_sets[k-1]
        inds = self._indices[k][:, None]
        index_set_right = self._index_sets[k+1]

        block = self._compute_block(index_set_left, inds, index_set_right)
        self._errors[k] = TT._compute_local_error(block, self._cores[k])
        self._build_basis_svd(block, k)
        
        return

    def _build_core(self, k: int) -> None:
        """Builds a single TT core."""
        match self._options.tt_method:
            case "fixed_rank":
                self._build_core_fixed(k)
            case "random":
                self._build_core_random(k)
            case "amen":
                self._build_core_amen(k)
            case _:
                msg = f"Unknown TT method: {self._options.tt_method}."
                raise Exception(msg)
        return

    def _build_block_final(self, k: int) -> None:
        """Computes the final block of the FTT approximation to the 
        target function.
        """

        index_set_prev = self._index_sets[k-1]
        inds = self._indices[k][:, None]
        index_set_next = self._index_sets[k+1]

        block = self._compute_block(index_set_prev, inds, index_set_next)
        self._errors[k] = TT._compute_local_error(block, self._cores[k])
        self._cores[k] = block

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
        max_rank:
            The maximum allowable rank of the resulting tensor core.
        
        Returns
        -------
        Ur:
            Matrix containing the left singular vectors of H after 
            truncation.
        sVhr: 
            Matrix containing the transpose of the product of the 
            singular values and the right-hand singular vectors after
            truncation. 
        rank:
            The number of singular values of H that were retained.

        """
        if tol is None: 
            tol = self._options.tol_svd
        if max_rank is None:
            max_rank = self._options.max_rank
        Ur, sr, Vhr, rank = tsvd(H, tol, max_rank)
        sVhr = sr[:, None] * Vhr
        return Ur, sVhr, rank
    
    def _build_basis_svd(
        self, 
        T: Tensor, 
        k: int, 
        tol: float | None = None,
        max_rank: int | None = None
    ) -> None:
        """Computes the coefficients of the kth tensor core.
        
        Parameters
        ----------
        T:
            An r_{k-1} * n_{k} * r_{k} tensor containing the 
            coefficients of the kth TT block.
        k:
            The index of the dimension corresponding to the basis 
            being constructed.
        tol:
            The tolerance to use when applying truncated SVD to the 
            unfolding matrix of T.
        max_rank:
            The maximum number of singular values to retain when 
            applying truncated SVD to the unfolding matrix of T.

        Returns
        -------
        None
            
        """

        k_prev = k - self._direction.value
        k_next = k + self._direction.value

        if self._direction == Direction.BACKWARD:
            T = T.swapdims(0, 2)
            self._cores[k_next] = self._cores[k_next].swapdims(0, 2)

        r_prev, n_k = T.shape[:2]
        r_next = self._cores[k_next].shape[0]

        T = mode_n_unfolding(T, n=2)
        U, sVh, rank = self._truncate_local(T, tol, max_rank)

        # Select a set of interpolation points
        indices_global, UU_inv, U_sub = self._select_points(U)
        core_shape = (r_prev, n_k, rank)
        self._cores[k] = mode_n_folding(UU_inv, n=2, newshape=core_shape)
        
        self._index_sets[k] = self._merge_indices(
            self._index_sets[k_prev], 
            self._indices[k], 
            indices_global
        )

        couple = (U_sub @ sVh)[:, :r_next]
        self._cores[k_next] = n_mode_prod(self._cores[k_next], couple, n=0)

        if self._direction == Direction.BACKWARD:
            self._cores[k] = self._cores[k].swapdims(0, 2)
            self._cores[k_next] = self._cores[k_next].swapdims(0, 2)

        return

    def _build_basis_amen(
        self, 
        H: Tensor,
        H_res: Tensor,
        H_up: Tensor,
        k: int
    ) -> None:
        """Computes the coefficients of the kth tensor core."""
        
        k_prev = k - self._direction.value
        k_next = k + self._direction.value

        res_w_prev = self._res_w[k-1]
        res_w_next = self._res_w[k+1]

        A_next = self._cores[k_next]

        n_left, n_k, n_right = H.shape
        r_0_next, _, r_1_next = A_next.shape

        if self._direction == Direction.FORWARD:
            H = unfold_left(H)
            H_up = unfold_left(H_up)
        else:
            H = unfold_right(H)
            H_up = unfold_right(H_up)
        
        U, sVh, rank = self._truncate_local(H)

        if self._direction == Direction.FORWARD:
            temp_l = fold_left(U, (n_left, n_k, rank))
            temp_l = torch.einsum("il, ljk", res_w_prev, temp_l)
            temp_r = sVh @ res_w_next
            H_up -= U @ temp_r
            H_res -= torch.einsum("ijl, lk", temp_l, temp_r)
            H_res = unfold_left(H_res)

        else: 
            temp_r = fold_right(U, (rank, n_k, n_right))
            temp_r = torch.einsum("ijl, lk", temp_r, res_w_next)
            temp_lt = sVh @ res_w_prev.T
            H_up -= U @ temp_lt
            H_res -= torch.einsum("li, ljk", temp_lt, temp_r)
            H_res = unfold_right(H_res)
        
        # Enrich basis
        T = torch.cat((U, H_up), dim=1)
        U, R = linalg.qr(T)
        r_new = U.shape[1]

        indices_global, B, U_interp = self._select_points(U)
        couple = U_interp @ R[:r_new, :rank] @ sVh

        self._index_sets[k] = self._merge_indices(
            self._index_sets[k_prev], 
            self._indices[k], 
            indices_global
        )

        U_res = self._truncate_local(H_res, tol=0.0)[0]
        indices_res_global = self._select_points(U_res)[0]
        
        self._res_l[k] = self._merge_indices(
            self._res_l[k_prev], 
            self._indices[k],
            indices_res_global
        )

        if self._direction == Direction.FORWARD:
            
            A = fold_left(B, (n_left, n_k, r_new))

            temp = torch.einsum("il, ljk", res_w_prev, A)
            temp = unfold_left(temp)
            res_w = temp[indices_res_global]

            couple = couple[:, :r_0_next]
            A_next = torch.einsum("il, ljk", couple, A_next)

        else:
            
            A = fold_right(B, (r_new, n_k, n_right))

            temp = torch.einsum("ijl, lk", A, res_w_next)
            temp = unfold_right(temp)
            res_w = temp[indices_res_global].T

            couple = couple[:, :r_1_next]
            A_next = torch.einsum("ijl, kl", A_next, couple)

        self._cores[k] = A 
        self._cores[k_next] = A_next
        self._res_w[k] = res_w 
        return

    def round(
        self, 
        tol: float | None = None, 
        max_rank: int | None = None
    ) -> None:
        """Rounds the TT cores. 
        
        Applies double rounding to get back to the starting direction.

        Parameters
        ----------
        tol:
            The tolerance to use when applying truncated SVD to round 
            each core.
        max_rank:
            The maximum allowable rank of each tensor core.
            
        """

        if tol is None:
            tol = self._options.tol_svd

        for _ in range(2):
            self._reverse_direction()
            if self._direction == Direction.FORWARD:
                inds = range(self._dim-1)
            else:
                inds = range(self._dim-1, 0, -1)
            for k in inds:
                self._build_basis_svd(self._cores[k], k, tol, max_rank)

        if self._options.tt_method == "amen":
            self._res_l = {}
            self._res_w = {}
        return

    def sweep(self):
        """Runs a single cross iteration.
        
        If cross iterations have been run previously, this performs a 
        sweep over the cores in the opposite order to the previous 
        sweep.
        
        """

        if self._cores == {}:
            self._initialise_cores()
        else:
            self._reverse_direction()     
        if self._options.tt_method == "amen":
            self._initialise_amen()
        if self._direction == Direction.FORWARD:
            inds = range(self._dim)
        else:
            inds = range(self._dim-1, -1, -1)
        
        for i, k in enumerate(inds):
            if self._options.verbose > 1:
                msg = f"Building block {i+1} / {self._dim}..."
                als_info(msg, end="\r")
            if i < self._dim-1:
                self._build_core(k)
            else:
                self._build_block_final(k)
        return