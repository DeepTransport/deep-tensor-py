from __future__ import annotations

from typing import Callable, Dict, Tuple
import warnings

import torch
from torch import linalg
from torch import Tensor

from .directions import Direction
from .eftt_options import EFTTOptions
from .tt import Grid, TT
from ..constants import EPS
from ..domains import Domain
from ..interpolation import deim, maxvol
from ..linalg import batch_mul, n_mode_prod, tsvd
from ..polynomials import Basis1D, Spectral
from ..references import Reference
from ..tools.printing import als_info


def _compute_weights(
    grid_points: Dict, 
    domain: Domain, 
    reference: Reference
) -> Dict[int, Tensor]:
    """Computes the weights used when selecting the initial index sets 
    in each dimension, by evaluating the reference density at each grid 
    point in each dimension.
    
    Parameters
    ----------
    grid_points:
        A dictionary containing the interpolation points in each 
        dimension (mapped to the local domain of the basis being used).
    domain:
        The domain of the reference density.
    reference:
        The reference density.
    
    Returns
    -------
    weights:
        A dictionary containing the evaluation of the reference density 
        at each element of the weights.
        
    """
    reference_weights = {}
    for k in grid_points:
        nodes_approx_k = domain._local2approx(grid_points[k])[0]
        reference_weights[k] = reference.eval_pdf(nodes_approx_k)[0].sqrt()
    return reference_weights


class FTT():
    r"""A functional tensor train, defined on $[-1, 1]^{d}$.

    Parameters
    ----------
    basis:
        A set of basis functions for each dimension of the FTT.
    tt: 
        A tensor train object.
    num_error_samples:
        The number of samples to use to estimate the $L^{2}$ error of 
        the FTT during its construction.
    
    """

    def __init__(
        self, 
        basis: Basis1D, 
        tt: TT | None = None,
        num_error_samples: int = 1000,
        device: torch.device = torch.get_default_device()
    ):
        self._tt = TT(device=device) if tt is None else tt
        self._basis = basis
        self._num_error_samples = num_error_samples
        self._device = device
        self._l2_error = None
        self._cores = {}
        return
    
    @property
    def _direction(self) -> Direction:
        return self._tt._direction

    @property 
    def _ranks(self) -> Tensor:
        return self._tt.ranks
    
    @property 
    def _num_eval_tt(self) -> int:
        return self._tt._num_eval

    @property
    def num_eval(self) -> int:
        return self._tt._num_eval + self._num_error_samples
    
    @property
    def _num_eval_construction(self) -> int:
        return self._tt._num_eval
    
    @property
    def _is_finished(self) -> bool:
        max_core_error = float(self._tt._errors.max())
        is_finished = max_core_error < self._tt._options.tol_max_core_error
        if self._l2_error:
            error_target_met = self._l2_error < self._tt._options.tol_l2_error
            is_finished = is_finished or error_target_met
        return is_finished

    @property 
    def _l2_error_samples(self) -> bool:
        """Whether to form a sample-based estimate of the L2 error."""
        return self._num_error_samples > 0

    def __call__(self, ls: Tensor, direction: Direction | None = None) -> Tensor:
        """Syntax sugar for self.eval()."""
        return self.eval(ls, direction)

    @staticmethod
    def _check_sample_dim(xs: Tensor, dim: int, strict: bool = False) -> None:
        """Checks that a set of samples is two-dimensional and that the 
        dimension does not exceed the expected dimension.
        """

        if xs.ndim != 2:
            msg = "Samples should be two-dimensional."
            raise Exception(msg)
        
        if strict and xs.shape[1] != dim:
            msg = (
                "Dimension of samples must be equal to dimension of "
                "approximation."
            )
            raise Exception(msg)

        if xs.shape[1] > dim:
            msg = (
                "Dimension of samples may not exceed dimension of "
                "approximation."
            )
            raise Exception(msg)

        return
    
    def _check_direction(self, xs: Tensor, direction: Direction | None) -> None:
        if xs.shape[1] != self._dim and direction is None:
            msg = (
                "A marginal function is being evaluated, but no "
                "direction has been provided."
            )
            raise Exception(msg)
        return
    
    def _print_info_header(self) -> None:
        info_headers = ["Iter", "Func Evals", "Max Rank", 
                        "Max Core Error", "Mean Core Error"]
        if self._l2_error_samples:
            info_headers += ["L2 Error"]
        als_info(" | ".join(info_headers))
        return

    def _print_info(self, cross_iter: int) -> None:
        """Prints diagnostics for the current cross iteration."""
        diagnostics = [
            f"{cross_iter+1:=4}", 
            f"{self.num_eval:=10}",
            f"{self._ranks.max():=8}",
            f"{self._tt._errors.max():=14.2e}",
            f"{self._tt._errors.mean():=15.2e}"
        ]
        if self._l2_error_samples:
            diagnostics += [f"{self._l2_error:=8.2e}"]
        als_info(" | ".join(diagnostics))
        return

    def _initialise_l2_error_samples(self) -> None:
        sample_size = (self._num_error_samples, self._dim)
        rs_unif = torch.rand(sample_size, device=self._device)
        self._ls_error = 2.0 * rs_unif - 1.0
        self._fls_error = self._target_func(self._ls_error)
        return    

    def _estimate_l2_error(self) -> None:
        """Computes the relative error between the value of the FTT 
        approximation to the target function and the true value for the 
        set of debugging samples.
        """
        fls_ftt = self(self._ls_error).flatten()
        numer = linalg.norm(self._fls_error - fls_ftt)
        denom = linalg.norm(self._fls_error)
        self._l2_error = numer / denom
        return

    def _eval_forward(self, ls: Tensor) -> Tensor:
        """Evaluates the FTT approximation to the target function for 
        the first k variables.
        """
        d_ls = ls.shape[1]
        Gs = [FTT._eval_core(self._basis, self._cores[k], ls[:, k])
              for k in range(d_ls)]
        Gs_prod = batch_mul(*Gs).squeeze(dim=1)
        return Gs_prod
    
    def _eval_backward(self, ls: Tensor) -> Tensor:
        """Evaluates the FTT approximation to the target function for 
        the last k variables.
        """
        d_ls = ls.shape[1]
        Gs = [FTT._eval_core(self._basis, self._cores[k], ls[:, i])
              for i, k in enumerate(range(self._dim-d_ls, self._dim))]
        Gs_prod = batch_mul(*Gs).squeeze(dim=2)
        return Gs_prod
    
    @staticmethod
    def _eval_core(basis: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates a tensor core."""
        r_p, n_k, r_k = A.shape
        n_ls = ls.numel()
        coeffs = A.permute(1, 0, 2).reshape(n_k, r_p * r_k)
        Gs = basis._eval_radon(coeffs, ls).reshape(n_ls, r_p, r_k)
        return Gs
    
    @staticmethod
    def _eval_core_rev(basis: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        return FTT._eval_core(basis, A, ls).swapdims(1, 2)
    
    @staticmethod
    def _eval_core_deriv(basis: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the derivative of a tensor core."""
        r_p, n_k, r_k = A.shape 
        n_ls = ls.numel()
        coeffs = A.permute(1, 0, 2).reshape(n_k, r_p * r_k)
        dGdls = basis._eval_radon_deriv(coeffs, ls).reshape(n_ls, r_p, r_k)
        return dGdls
    
    @staticmethod
    def _eval_core_deriv_rev(basis: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        return FTT._eval_core_deriv(basis, A, ls).swapdims(1, 2)

    def eval(self, ls: Tensor, direction: Direction | None = None) -> Tensor:
        r"""Evaluates the FTT.
        
        Returns the functional tensor train approximation to the target 
        function for either the first or last $k$ variables, for a set 
        of points mapped to the domain of the basis functions.
        
        Parameters
        ----------
        ls:
            An $n \times k$ matrix containing a set of samples mapped 
            to the domain of the FTT basis functions.
        direction:
            The direction in which to iterate over the cores.
        
        Returns
        -------
        Gs_prod:
            An $n \times r$ matrix, where each row contains the product 
            of the first or last (depending on direction) $k$ tensor 
            cores evaluated at the corresponding sample in `ls`.
            
        """
        self._check_sample_dim(ls, self._dim)
        self._check_direction(ls, direction)
        if direction in (Direction.FORWARD, None):
            return self._eval_forward(ls) 
        return self._eval_backward(ls)

    def _round(
        self, 
        tol: float | None = None, 
        max_rank: int | None = None
    ) -> None:
        self._tt.round(tol, max_rank)
        return
     
    def _compute_cores(self) -> None:
        """(Re)-computes the FTT cores from the TT cores."""
        for k in range(self._dim):
            core = self._tt._cores[k].clone()
            if isinstance(basis := self._basis, Spectral):
                core = n_mode_prod(core, basis._node2basis, n=1)
            self._cores[k] = core
        return
    
    def _construct_tt(self, grid: Grid) -> None:
        """Constructs the underlying tensor train approximation to the 
        discretisation of the function on the tensor-product grid 
        formed from the collocation points.
        """
        
        self._tt.initialise(self._target_func, grid)
        if self._l2_error_samples:
            self._initialise_l2_error_samples()
        if self._tt._options.verbose > 0:
            self._print_info_header()

        for num_iter in range(self._tt._options.max_als): 
            self._tt.sweep()
            self._compute_cores()
            if self._l2_error_samples:
                self._estimate_l2_error()
            if self._tt._options.verbose > 0:
                self._print_info(num_iter)
            if self._is_finished:
                break        

        if self._tt._options.verbose > 0:
            als_info("ALS complete.")
        if self._tt._options.verbose > 1:
            als_info(f"Maximum TT rank: {self._tt.ranks.max()}.")

        return

    def approximate(
        self, 
        target_func: Callable[[Tensor], Tensor],
        dim: int,
        reference: Reference | None = None
    ) -> None:
        r"""Constructs an FTT approximation to a target function.

        Parameters
        ----------
        target_func: 
            The target function, $f : [-1, 1]^{d} \rightarrow \mathbb{R}$.
        dim:
            The dimension of the target function.
        reference:
            The reference measure. If provided, this will be used to 
            generate the initial index sets for the underlying TT. 
            Otherwise, these sets will be generated by sampling 
            uniformly from the underlying tensor grid.
        
        """
        self._target_func = target_func
        self._dim = dim

        points = {k: self._basis._nodes for k in range(self._dim)}
        weights = (_compute_weights(points, reference._domain, reference)
                   if isinstance(reference, Reference)
                   else None)
        grid = Grid(points, weights)
        
        self._construct_tt(grid)
        return
    
    def _clone(self) -> FTT:

        tt = TT(self._tt._options, device=self._device)
        tt._cores = {k: self._tt._cores[k].clone() for k in self._tt._cores}
        tt._index_sets = {k: self._tt._index_sets[k].clone() for k in self._tt._index_sets}
        tt._direction = self._tt._direction

        ftt = FTT(self._basis, tt, self._num_error_samples, self._device)
        return ftt


class EFTT(FTT):
    r"""An extended functional tensor train, defined on $[-1, 1]^{d}$.
    
    Parameters
    ----------
    basis:
        The basis function used for each dimension of the FTT.
    tt: 
        A tensor train object.
    options: 
        A set of tuning parameters used during the construction of the 
        EFTT.

    Attributes
    ----------
    num_eval:
        The number of function evaluations required to construct the 
        EFTT.

    """

    def __init__(
        self, 
        basis: Basis1D,
        tt: TT,
        options: EFTTOptions | None = None,
        device: torch.device = torch.get_default_device()
    ):
        if options is None:
            options = EFTTOptions()
        FTT.__init__(self, basis, tt, options.num_error_samples, device=device)
        self._options = options
        self._num_eval_fibres = 0
        self._tucker_inds: Dict[int, Tensor] = {}
        self._factors: Dict[int, Tensor] = {}
        return
    
    @property
    def num_eval(self) -> int:
        return self._num_error_samples + self._num_eval_fibres + self._tt._num_eval
    
    @property 
    def _num_eval_construction(self) -> int:
        return self._num_eval_fibres + self._tt._num_eval
    
    @property 
    def _basis_dims(self) -> Tensor:
        """Returns a tensor containing the dimension of the reduced 
        basis for each coordinate.
        """
        basis_dims = [self._factors[k].shape[1] for k in range(self._dim)]
        return torch.tensor(basis_dims, device=self._device)
    
    def _compute_fibre_submatrix_random(
        self, 
        grid: Grid, 
        reference: Reference | None,
        k: int
    ) -> Tensor:
        
        n_k = grid._points[k].numel()
        sample_size = (self._options.num_snapshots, self._dim)

        if reference is not None:
            point_samples = reference.random(*sample_size)
            point_samples = reference._domain._approx2local(point_samples)[0]
        else:
            point_samples = 2.0 * torch.rand(sample_size, device=self._device) - 1.0

        point_samples = point_samples.repeat((n_k, 1))
        point_samples[:, k] = grid._points[k].repeat_interleave(self._options.num_snapshots)

        # Note: each column is a fibre
        fibre_matrix = self._target_func(point_samples)
        fibre_matrix = fibre_matrix.reshape(n_k, self._options.num_snapshots)
        self._num_eval_fibres += fibre_matrix.numel()

        return fibre_matrix
    
    @staticmethod
    def _find_evaluated_points(
        new_inds: Tensor, 
        inds_eval: Tensor,
        vals_eval: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Returns a mask of elements of a set of indices that have 
        been computed previously, as well as the computed values.
        """
        diffs = (new_inds[:, None, :] - inds_eval[None, ...]).abs().sum(dim=2)
        inds_prev = diffs.argmin(dim=1)
        mask = diffs.min(dim=1).values < EPS
        mask_vals = vals_eval[inds_prev[mask]]
        return mask, mask_vals
    
    def _generate_points_aca(self, n: int, grid: Grid) -> Tuple[Tensor, Tensor]:
        """Returns a set of random indices and the corresponding 
        function values.
        """
        inds_rand = grid._sample_indices(n)
        random_points = grid._indices2points(inds_rand)
        func_vals = self._target_func(random_points)
        self._num_eval_fibres += func_vals.numel()
        return inds_rand, func_vals
    
    def _initialise_index_set_aca(self, grid: Grid) -> Tuple[Tensor, Tensor]:
        """Initialises the index set defining the current cross by 
        sampling from the coefficient tensor at random. This is 
        repeated multiple times in case the sampled elements are 
        uniformly zero (see also implementation by Strossner et al.).
        """

        num_initialisation_batches = 5
        num_aca = self._options.num_aca

        for _ in range(num_initialisation_batches):
            inds_rand, func_vals = self._generate_points_aca(num_aca, grid)
            if func_vals.abs().max() > 0.0:
                break
        
        if func_vals.abs().max() == 0.0:
            msg = (
                "ACA: None of the sampled fibre elements are nonzero. "
                "Consider rescaling the target function. If you are "
                "confident the target function is scaled appropriately, "
                "consider using a refined grid, larger core ranks, an "
                "increased number of bridging densities, or a larger "
                "value for num_aca."
            )
            warnings.warn(msg)
        
        max_residual_index = func_vals.abs().argmax()
        inds = torch.atleast_2d(inds_rand[max_residual_index])
        vals = torch.atleast_1d(func_vals[max_residual_index])
        return inds, vals
    
    def _compute_fibre_submatrix_aca(self, grid: Grid, k: int) -> Tensor:

        num_aca = self._options.num_aca
        inds, vals = self._initialise_index_set_aca(grid)

        # Keep track of elements of the cross that have been evaluated
        inds_eval = inds.clone()
        vals_eval = vals.clone()

        max_abs_func = torch.tensor(0.0)

        for _ in range(1, self._options.max_fibres):

            num_inds = inds.shape[0]
            inds_rand, func_vals = self._generate_points_aca(num_aca, grid)
            max_abs_func = torch.max(max_abs_func, func_vals.abs().max())

            inds_int = inds.repeat(num_inds, 1)
            inds_int[:, k] = inds[:, k].repeat_interleave(num_inds, dim=0)
            inds_row = inds_rand.repeat(num_inds, 1)
            inds_row[:, k] = inds[:, k].repeat_interleave(num_aca, dim=0)
            inds_col = inds.repeat(self._options.num_aca, 1)
            inds_col[:, k] = inds_rand[:, k].repeat_interleave(num_inds, dim=0)

            points_int = grid._indices2points(inds_int)
            points_row = grid._indices2points(inds_row)
            points_col = grid._indices2points(inds_col)

            mask, mask_vals = self._find_evaluated_points(
                inds_int, inds_eval, vals_eval
            )

            # Form intersection submatrix (avoiding the evaluation 
            # of function values that were previously computed)
            B_int = torch.zeros(inds_int.shape[0])
            B_int[mask] = mask_vals
            if (~mask).any():
                B_int[~mask] = self._target_func(points_int[~mask])
            
            B_rows = self._target_func(points_row)
            B_cols = self._target_func(points_col)
            
            B_int = B_int.reshape(num_inds, num_inds)
            B_rows = B_rows.reshape(num_inds, num_aca)
            B_cols = B_cols.reshape(num_aca, num_inds)
            
            inds_eval = inds_int.clone()
            vals_eval = B_int.flatten()
            
            num_eval_int = int((~mask).sum())
            self._num_eval_fibres += (
                num_eval_int + B_rows.numel() + B_cols.numel()
            )

            # Check for (near-)singularity of intersection matrix
            # (also done in implementation by Strossner et al.).
            # This occurs for functions where the fibre matrices 
            # are exactly low rank.
            if linalg.cond(B_int) > 1.0 / EPS:
                break

            # Note: Strossner et al. just take the maximum residual, 
            # but I think the below error is easier to work with 
            # because it is invariant to rescalings of the target 
            # function
            cross_vals = B_cols @ linalg.solve(B_int, B_rows)
            residuals = (func_vals - cross_vals.diag()).abs()
            error = residuals.max() / max_abs_func
            if error < self._options.tol_aca:
                break

            # Update index set
            max_index = inds_rand[residuals.argmax(), :]
            inds = torch.vstack((inds, max_index))
        
        n_k = self._basis._cardinality
        num_inds = inds.shape[0]

        fibre_inds = inds.repeat(n_k, 1)
        ii = torch.arange(n_k, device=self._device)
        fibre_inds[:, k] = ii.repeat_interleave(num_inds, dim=0)
        fibre_points = grid._indices2points(fibre_inds)

        mask, mask_vals = self._find_evaluated_points(
            fibre_inds, inds_eval, vals_eval
        )

        fibre_matrix = torch.zeros((n_k*num_inds,))
        fibre_matrix[mask] = mask_vals
        fibre_matrix[~mask] = self._target_func(fibre_points[~mask])
        fibre_matrix = fibre_matrix.reshape(n_k, num_inds)

        num_eval_new = int((~mask).sum())
        self._num_eval_fibres += num_eval_new

        return fibre_matrix

    def _compute_reduced_indices(
        self, 
        reference: Reference | None = None
    ) -> None:
        """Computes the reduced index set in each dimension."""

        points = {k: self._basis._nodes for k in range(self._dim)}
        grid = Grid(points)

        for k in range(self._dim):

            if self._tt._options.verbose > 1:
                msg = (
                    "Computing reduced basis for dimension "
                    f"{k+1} / {self._dim}..."
                )
                als_info(msg, end="\r")

            if self._options.fibre_method == "random":
                fibre_matrix = self._compute_fibre_submatrix_random(grid, reference, k)
                basis_k = tsvd(fibre_matrix, tol=self._options.tol_svd)[0]
                inds_k, factor_k = deim(basis_k)

            elif self._options.fibre_method == "aca":
                fibre_matrix = self._compute_fibre_submatrix_aca(grid, k)
                U_k = linalg.qr(fibre_matrix).Q
                inds_k = maxvol(U_k)[0]
                factor_k = linalg.solve(U_k[inds_k], U_k, left=False)
            
            self._tucker_inds[k] = inds_k
            self._factors[k] = factor_k
        
        if self._tt._options.verbose > 1:
            basis_dims = [dim for dim in self._basis_dims]
            msg = (
                "Maximum reduced basis dimension: "
                + f"{max(basis_dims)}."
            )
            als_info(msg.ljust(60))

        return
    
    def _compute_cores(self) -> None:
        """(Re)-computes the FTT cores from the TT cores."""
        for k in range(self._dim):
            core = n_mode_prod(self._tt._cores[k], self._factors[k], n=1)
            if isinstance(basis := self._basis, Spectral):
                core = n_mode_prod(core, basis._node2basis, n=1)
            self._cores[k] = core
        return

    def approximate(
        self, 
        target_func: Callable[[Tensor], Tensor], 
        dim: int,
        reference: Reference | None = None
    ) -> None:
        r"""Constructs an FTT approximation to a target function.

        Parameters
        ----------
        target_func: 
            The target function, $f : [-1, 1]^{d} \rightarrow \mathbb{R}$. 
        dim:
            The dimension of the target function.
        reference:
            The reference measure. If provided, this will be used to 
            generate the samples to build the fibre matrix bases and 
            generate the initial index sets for the underlying TT. 
            Otherwise, the samples will be drawn uniformly.
        
        """

        self._target_func = target_func
        self._dim = dim
        self._compute_reduced_indices(reference)

        deim_nodes = {
            k: self._basis._nodes[self._tucker_inds[k]] 
            for k in range(self._dim)
        }
        if reference is not None:
            weights = (_compute_weights(deim_nodes, reference._domain, reference)
                   if isinstance(reference, Reference)
                   else None)
            deim_grid = Grid(deim_nodes, weights)
        else:
            deim_grid = Grid(deim_nodes)
        self._construct_tt(deim_grid)
        return
    
    def _clone(self) -> EFTT:
        # Note: we cannot copy the cores and index sets over, because 
        # the indices corresponding to the DEIM projection onto the 
        # reduced bases in each dimension can change. Instead we start 
        # from scratch.
        tt = TT(self._tt._options, device=self._device)
        ftt = EFTT(self._basis, tt, self._options, device=self._device)
        return ftt