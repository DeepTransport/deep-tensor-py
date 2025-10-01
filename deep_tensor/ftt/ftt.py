from typing import Callable, Dict

import torch
from torch import linalg
from torch import Tensor

from .approx_bases import ApproxBases
from .directions import Direction
from .eftt_options import EFTTOptions
from .tt import Grid, TT
from ..constants import EPS
from ..domains import Domain
from ..interpolation import deim
from ..linalg import batch_mul, n_mode_prod, tsvd
from ..polynomials import Basis1D, Spectral
from ..references import Reference
from ..tools.printing import als_info


def compute_weights(
    grid_points: Dict, 
    domain: Domain, 
    reference: Reference
) -> Dict[int, Tensor]:
    # TODO: this won't have quite the indended effect when the 
    # collocation points are not spaced equally--need to fix.
    
    reference_weights = {}
    for k in grid_points:
        nodes_approx_k = domain.local2approx(grid_points[k])[0]
        reference_weights[k] = reference.eval_pdf(nodes_approx_k)[0]
    
    return reference_weights


class FTT():
    r"""A functional tensor train, defined on $[-1, 1]^{d}$.

    Parameters
    ----------
    bases:
        A set of basis functions for each dimension of the FTT.
    tt: 
        A tensor train object.
    num_error_samples:
        The number of samples to use to estimate the $L_{2}$ error of 
        the FTT during its construction.
    
    """

    def __init__(
        self, 
        bases: ApproxBases, 
        tt: TT | None = None,
        num_error_samples: int = 1000
    ):
        self.tt = TT() if tt is None else tt
        self.bases = bases 
        self.dim = bases.dim
        self.num_error_samples = num_error_samples
        self.l2_error = None
        self.cores = {}
        return
    
    @property
    def direction(self) -> Direction:
        return self.tt.direction

    @property 
    def ranks(self) -> Tensor:
        return self.tt.ranks

    @property
    def num_eval(self) -> int:
        return self.tt.num_eval + self.num_error_samples
    
    @property
    def is_finished(self) -> bool:
        max_core_error = float(self.tt.errors.max())
        is_finished = max_core_error < self.tt.options.tol_max_core_error
        if self.l2_error:
            error_target_met = self.l2_error < self.tt.options.tol_l2_error
            is_finished = is_finished or error_target_met
        return is_finished

    @property 
    def l2_error_samples(self) -> bool:
        """Whether to form a sample-based estimate of the L2 error."""
        return self.num_error_samples > 0

    def __call__(self, ls: Tensor, direction: Direction | None = None) -> Tensor:
        """Syntax sugar for self.eval()."""
        return self.eval(ls, direction)

    @staticmethod
    def check_sample_dim(xs: Tensor, dim: int, strict: bool = False) -> None:
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
    
    @staticmethod
    def eval_core(basis: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates a tensor core."""
        r_p, n_k, r_k = A.shape
        n_ls = ls.numel()
        coeffs = A.permute(1, 0, 2).reshape(n_k, r_p * r_k)
        Gs = basis.eval_radon(coeffs, ls).reshape(n_ls, r_p, r_k)
        return Gs
    
    @staticmethod
    def eval_core_rev(basis: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        return FTT.eval_core(basis, A, ls).swapdims(1, 2)
    
    @staticmethod
    def eval_core_deriv(basis: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the derivative of a tensor core."""
        r_p, n_k, r_k = A.shape 
        n_ls = ls.numel()
        coeffs = A.permute(1, 0, 2).reshape(n_k, r_p * r_k)
        dGdls = basis.eval_radon_deriv(coeffs, ls).reshape(n_ls, r_p, r_k)
        return dGdls
    
    @staticmethod
    def eval_core_deriv_rev(basis: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        return FTT.eval_core_deriv(basis, A, ls).swapdims(1, 2)

    def print_info_header(self) -> None:

        info_headers = [
            "Iter", 
            "Func Evals",
            "Max Rank", 
            "Max Core Error", 
            "Mean Core Error"
        ]
        if self.l2_error_samples:
            info_headers += ["L2 Error"]

        als_info(" | ".join(info_headers))
        return

    def print_info(self, cross_iter: int) -> None:
        """Prints some diagnostic information about the current cross 
        iteration.
        """

        diagnostics = [
            f"{cross_iter+1:=4}", 
            f"{self.num_eval:=10}",
            f"{self.ranks.max():=8}",
            f"{self.tt.errors.max():=14.2e}",
            f"{self.tt.errors.mean():=15.2e}"
        ]
        if self.l2_error_samples:
            diagnostics += [f"{self.l2_error:=8.2e}"]

        als_info(" | ".join(diagnostics))
        return

    def estimate_l2_error(self) -> None:
        """Computes the relative error between the value of the FTT 
        approximation to the target function and the true value for the 
        set of debugging samples.
        """
        fls_ftt = self(self.ls_error).flatten()
        numer = linalg.norm(self.fls_error - fls_ftt)
        denom = linalg.norm(self.fls_error)
        self.l2_error = numer / denom
        return

    def eval_forward(self, ls: Tensor) -> Tensor:
        """Evaluates the FTT approximation to the target function for 
        the first k variables.
        """
        d_ls = ls.shape[1]
        Gs = [
            FTT.eval_core(self.bases[k], self.cores[k], ls[:, k])
            for k in range(d_ls)
        ]
        Gs_prod = batch_mul(*Gs).squeeze(dim=1)
        return Gs_prod
    
    def eval_backward(self, ls: Tensor) -> Tensor:
        """Evaluates the FTT approximation to the target function for 
        the last k variables.
        """
        d_ls = ls.shape[1]
        Gs = [
            FTT.eval_core(self.bases[k], self.cores[k], ls[:, i])
            for i, k in enumerate(range(self.dim-d_ls, self.dim))
        ]
        Gs_prod = batch_mul(*Gs).squeeze(dim=2)
        return Gs_prod

    def eval(self, ls: Tensor, direction: Direction | None = None) -> Tensor:
        r"""Evaluates the FTT.
        
        Returns the functional tensor train approximation to the target 
        function for either the first or last $k$ variables, for a set 
        of points mapped to the domain of the basis functions.
        
        Parameters
        ----------
        ls:
            An $n \times d$ matrix containing a set of samples mapped 
            to the domain of the FTT basis functions.
        direction:
            The direction in which to iterate over the cores.
        
        Returns
        -------
        Gs_prod:
            An $n \times n_{k}$ matrix, where each row contains the 
            product of the first or last (depending on direction) $k$ 
            tensor cores evaluated at the corresponding sample in `ls`.
            
        """
        
        self.check_sample_dim(ls, self.dim)
        
        # TODO: tidy this up.
        if ls.shape[1] != self.dim and direction is None:
            msg = "Need to give direction if marginal is being evaluated."
            raise Exception(msg)

        if direction in (Direction.FORWARD, None):
            Gs_prod = self.eval_forward(ls)
        else: 
            Gs_prod = self.eval_backward(ls)
        
        return Gs_prod
    
    def initialise_l2_error_samples(self):
        # TODO: figure out whether these should be drawn from the 
        # measure associated with the basis in each dimension.
        # self.ls_error = self.bases.sample_measure(self.num_error_samples)[0]
        sample_size = (self.num_error_samples, self.dim)
        self.ls_error = 2.0 * torch.rand(sample_size) - 1.0
        self.fls_error = self.target_func(self.ls_error)
        return 

    def round(
        self, 
        tol: float | None = None, 
        max_rank: int | None = None
    ) -> None:
        self.tt.round(tol, max_rank)
        return
     
    def compute_cores(self) -> None:
        """(Re)-computes the FTT cores from the TT cores."""
        for k in range(self.dim):
            core = self.tt.cores[k].clone()
            if isinstance(basis := self.bases[k], Spectral):
                core = n_mode_prod(core, basis.node2basis, n=1)
            self.cores[k] = core
        return
    
    def construct_tt(self, grid: Grid) -> None:
        """Constructs the underlying tensor train approximation to the 
        discretisation of the function on the tensor-product grid 
        formed from the collocation points.
        """
        
        self.tt.initialise(self.target_func, grid)
        if self.l2_error_samples:
            self.initialise_l2_error_samples()
        if self.tt.options.verbose > 0:
            self.print_info_header()

        for num_iter in range(self.tt.options.max_als): 
            self.tt.sweep()
            self.compute_cores()
            if self.l2_error_samples:
                self.estimate_l2_error()
            if self.tt.options.verbose > 0:
                self.print_info(num_iter)
            if self.is_finished:
                break        

        if self.tt.options.verbose > 0:
            als_info("ALS complete.")
        if self.tt.options.verbose > 1:
            als_info(f"Maximum TT rank: {self.tt.ranks.max()}.")

        return

    def approximate(
        self, 
        target_func: Callable[[Tensor], Tensor],
        reference: Reference | None = None
    ) -> None:
        r"""Constructs a FTT approximation to a target function.

        Parameters
        ----------
        target_func: 
            The target function, $f : [-1, 1]^{d} \rightarrow \mathbb{R}$.
        reference:
            The reference measure. If provided, this will be used to 
            generate the initial index sets for the underlying TT. 
            Otherwise, these sets will be generated by sampling 
            uniformly from the underlying tensor grid.
        
        """

        self.target_func = target_func

        points = {k: self.bases[k].nodes for k in range(self.dim)}
        if reference is None:
            grid = Grid(points)
        else:
            weights = compute_weights(points, reference.domain, reference)
            grid = Grid(points, weights)

        self.construct_tt(grid)
        return
    
    def clone(self):

        tt = TT(self.tt.options)
        tt.cores = {k: self.tt.cores[k].clone() for k in self.tt.cores}
        tt.index_sets = {k: self.tt.index_sets[k].clone() for k in self.tt.index_sets}
        tt.direction = self.tt.direction

        ftt = FTT(self.bases, tt)
        return ftt


class EFTT(FTT):
    r"""An extended functional tensor train, defined on $[-1, 1]^{d}$.
    
    Parameters
    ----------
    bases:
        A set of basis functions for each dimension of the EFTT.
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
        bases: ApproxBases,
        tt: TT,
        options: EFTTOptions | None = None
    ):
        if options is None:
            options = EFTTOptions()
        FTT.__init__(self, bases, tt, options.num_error_samples)
        self.options = options
        self.num_eval_fibres = 0
        self.deim_inds: Dict[int, Tensor] = {}
        self.factors: Dict[int, Tensor] = {}
        return
    
    @property
    def num_eval(self) -> int:
        return self.num_error_samples + self.num_eval_fibres + self.tt.num_eval
    
    @property 
    def basis_dims(self) -> Tensor:
        """Returns a tensor containing the dimension of the reduced 
        basis for each coordinate.
        """
        basis_dims = torch.tensor([self.factors[k].shape[1] 
                                   for k in range(self.dim)])
        return basis_dims
    
    def compute_fibre_submatrix_random(
        self, 
        grid: Grid, 
        reference: Reference | None,
        k: int
    ) -> Tensor:
        
        n_k = grid.points[k].numel()

        if reference is None:
            sample_size = (self.options.num_snapshots, self.dim)
            point_samples = 2.0 * torch.rand(sample_size) - 1.0
        else:
            point_samples = reference.random(self.options.num_snapshots, self.dim)
            point_samples = reference.domain.approx2local(point_samples)[0]

        point_samples = point_samples.repeat((n_k, 1))
        point_samples[:, k] = grid.points[k].repeat_interleave(self.options.num_snapshots)

        # Note: each column is a fibre
        fibre_matrix = self.target_func(point_samples)
        fibre_matrix = fibre_matrix.reshape(n_k, self.options.num_snapshots)
        self.num_eval_fibres += fibre_matrix.numel()

        return fibre_matrix
    
    def compute_fibre_submatrix_aca(self, grid: Grid, k: int) -> Tensor:

        for iter in range(self.options.max_fibres):
                
            random_inds = grid.sample_indices(self.options.num_aca)
            random_points = grid.indices2points(random_inds)

            M_vals = self.target_func(random_points)
            
            if iter == 0:

                max_residual = M_vals.max()
                max_residual_index = M_vals.abs().argmax()
                max_index = random_inds[max_residual_index, :]

                inds = torch.atleast_2d(max_index)

            else:

                num_inds = inds.shape[0]

                # Compute intersection matrix (NOTE: some of this 
                # will have actually been computed at previous 
                # iterations...)
                inds_int = inds.repeat(num_inds, 1)
                inds_int[:, k] = inds[:, k].repeat_interleave(num_inds, dim=0)

                inds_row = random_inds.repeat(num_inds, 1)
                inds_row[:, k] = inds[:, k].repeat_interleave(self.options.num_aca, dim=0)

                inds_col = inds.repeat(self.options.num_aca, 1)
                inds_col[:, k] = random_inds[:, k].repeat_interleave(num_inds, dim=0)

                points_int = grid.indices2points(inds_int)
                points_row = grid.indices2points(inds_row)
                points_col = grid.indices2points(inds_col)

                B_int = self.target_func(points_int)
                B_int = B_int.reshape(num_inds, num_inds)
                B_rows = self.target_func(points_row)
                B_rows = B_rows.reshape(num_inds, self.options.num_aca)
                B_cols = self.target_func(points_col)
                B_cols = B_cols.reshape(self.options.num_aca, num_inds)

                self.num_eval_fibres += (
                    2 * num_inds * self.options.num_aca 
                    + num_inds ** 2
                )

                # Check for (near-)singularity of intersection matrix
                # (also done in implementation by Strossner et al.).
                if linalg.cond(B_int) > 1.0 / EPS:
                    break

                # Update index set with index of maximum residual
                B_vals = B_cols @ linalg.solve(B_int, B_rows)
                residuals = torch.diag(M_vals - B_vals).abs()
                max_residual = residuals.max()
                max_residual_index = residuals.abs().argmax()
                max_index = random_inds[max_residual_index, :]
                inds = torch.vstack((inds, max_index))
            
            if max_residual < self.options.tol_aca and iter > 1:
                break
        
        n_k = self.bases[k].cardinality
        num_inds = inds.shape[0]

        fibre_inds = inds.repeat(n_k, 1)
        fibre_inds[:, k] = torch.arange(n_k).repeat_interleave(num_inds, dim=0)

        fibre_points = grid.indices2points(fibre_inds)
        fibre_matrix = self.target_func(fibre_points).reshape(n_k, num_inds)
        self.num_eval_fibres += n_k * num_inds

        return fibre_matrix

    def compute_reduced_indices(
        self, 
        reference: Reference | None = None
    ) -> None:
        """Computes the POD bases in each dimension."""

        points = {k: self.bases[k].nodes for k in range(self.dim)}
        grid = Grid(points)

        for k in range(self.dim):

            if self.tt.options.verbose > 1:
                msg = (
                    "Computing reduced basis for dimension "
                    f"{k+1} / {self.dim}..."
                )
                als_info(msg, end="\r")

            if self.options.fibre_method == "random":
                fibre_matrix = self.compute_fibre_submatrix_random(grid, reference, k)
            elif self.options.fibre_method == "aca":
                fibre_matrix = self.compute_fibre_submatrix_aca(grid, k)
            
            basis_k = tsvd(fibre_matrix, tol=self.options.tol_svd)[0]
            self.deim_inds[k], self.factors[k] = deim(basis_k)
        
        if self.tt.options.verbose > 1:
            basis_dims = [dim for dim in self.basis_dims]
            msg = (
                "Maximum reduced basis dimension: "
                + f"{max(basis_dims)}."
            )
            als_info(msg.ljust(60))

        return
    
    def compute_cores(self) -> None:
        """(Re)-computes the FTT cores from the TT cores."""
        for k in range(self.dim):
            core = n_mode_prod(self.tt.cores[k], self.factors[k], n=1)
            if isinstance(basis := self.bases[k], Spectral):
                core = n_mode_prod(core, basis.node2basis, n=1)
            self.cores[k] = core
        return

    def approximate(
        self, 
        target_func: Callable[[Tensor], Tensor], 
        reference: Reference | None = None
    ) -> None:
        r"""Constructs a FTT approximation to a target function.

        Parameters
        ----------
        target_func: 
            The target function, $f : [-1, 1]^{d} \rightarrow \mathbb{R}$. 
        reference:
            The reference measure. If provided, this will be used to 
            generate the samples to build the fibre matrix bases and 
            generate the initial index sets for the underlying TT. 
            Otherwise, the samples will be drawn uniformly.
        
        """

        self.target_func = target_func
        self.compute_reduced_indices(reference)

        deim_nodes = {
            k: self.bases[k].nodes[self.deim_inds[k]] 
            for k in range(self.dim)
        }
        deim_grid = Grid(deim_nodes)
        self.construct_tt(deim_grid)
        return
    
    def clone(self):
        # Note: can't copy the cores over because the number of 
        # collocation points is not fixed. Could try the index sets 
        # (and direction) though (although would need to adjust their 
        # size at some point)...
        tt = TT(self.tt.options)
        ftt = EFTT(self.bases, tt, self.options)
        return ftt