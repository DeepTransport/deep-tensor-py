import abc
from typing import Callable, Dict

import torch
from torch import Tensor

from .approx_bases import ApproxBases
from .directions import Direction
from .input_data import InputData
from .tt import Grid, TT
from ..domains import Domain
from ..interpolation import deim
from ..linalg import batch_mul, n_mode_prod, tsvd
from ..options import TTOptions
from ..polynomials import Basis1D, Spectral
from ..references import Reference
from ..tools.printing import als_info


def compute_weights(
    grid_points: Dict, 
    domain: Domain, 
    reference: Reference
) -> Dict[int, Tensor]:
    
    reference_weights = {}
    for k in grid_points:
        nodes_approx_k = domain.local2approx(grid_points[k])[0]
        reference_weights[k] = reference.eval_pdf(nodes_approx_k)[0]
    
    return reference_weights


class TTFunc(abc.ABC):
    """Base class for functional tensor trains."""

    def __init__(
        self,
        target_func: Callable[[Tensor], Tensor], 
        bases: ApproxBases, 
        options: TTOptions, 
        input_data: InputData, 
        reference: Reference,
        tt_data = None
    ):

        self.target_func = target_func
        self.bases = bases
        self.dim = self.bases.dim
        self.options = options
        self.input_data = input_data  # TODO: get rid of this... only using the debugging samples.
        self.reference = reference
        self.l2_err = torch.inf
        self.linf_err = torch.inf
        
        grid_points = {k: self.bases[k].nodes for k in range(self.dim)}
        reference_weights = compute_weights(grid_points, bases.domain, reference)

        self.grid = Grid(grid_points, reference_weights)
        self.tt = TT(target_func, self.grid, options)
        self.cores = {}
        self.tt_data = tt_data

        # Generate debugging samples.
        if self.input_data.is_debug:
            self.input_data.set_debug(self.target_func, self.bases)

        return
    
    @property
    def direction(self) -> Direction:
        return self.tt.direction

    @property 
    def ranks(self) -> Tensor:
        return self.tt.ranks
    
    @property
    def is_finished(self) -> bool:
        max_core_error = self.tt.errors.max().item()
        is_finished = (
            max_core_error < self.options.als_tol 
            or self.l2_err < self.options.als_tol
        )
        return is_finished

    @property
    @abc.abstractmethod
    def num_eval(self) -> int:
        pass

    @staticmethod
    def _check_sample_dim(xs: Tensor, dim: int, strict: bool = False) -> None:
        """Checks that a set of samples is two-dimensional and that the 
        dimension does not exceed the expected dimension.
        """

        if xs.ndim != 2:
            msg = "Samples should be two-dimensional."
            raise Exception(msg)
        
        if strict and xs.shape[1] != dim:
            msg = ("Dimension of samples must be equal to dimension "
                   + "of approximation.")
            raise Exception(msg)

        if xs.shape[1] > dim:
            msg = ("Dimension of samples may not exceed dimension "
                   + "of approximation.")
            raise Exception(msg)

        return
    
    @staticmethod
    def eval_core(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates a tensor core.
        """
        r_p, n_k, r_k = A.shape
        n_ls = ls.numel()
        coeffs = A.permute(1, 0, 2).reshape(n_k, r_p * r_k)
        Gs = poly.eval_radon(coeffs, ls).reshape(n_ls, r_p, r_k)
        return Gs
    
    @staticmethod
    def _eval_core_213(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates a tensor core.
        """
        r_p, n_k, r_k = A.shape
        n_ls = ls.numel()
        coeffs = A.permute(1, 0, 2).reshape(n_k, r_p * r_k)
        Gs = poly.eval_radon(coeffs, ls).reshape(n_ls, r_p, r_k)
        return Gs

    @staticmethod
    def _eval_core_213_deriv(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the derivative of a tensor core.
        """
        r_p, n_k, r_k = A.shape 
        n_ls = ls.numel()
        coeffs = A.permute(1, 0, 2).reshape(n_k, r_p * r_k)
        dGdls = poly.eval_radon_deriv(coeffs, ls).reshape(n_ls, r_p, r_k)
        return dGdls

    @staticmethod
    def _eval_core_231(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates a tensor core.
        """
        return TTFunc._eval_core_213(poly, A, ls).swapdims(1, 2)
    
    @staticmethod
    def _eval_core_231_deriv(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the derivative of a tensor core.
        """
        return TTFunc._eval_core_213_deriv(poly, A, ls).swapdims(1, 2)

    def _print_info_header(self) -> None:

        info_headers = [
            "Iter", 
            "Func Evals",
            "Max Rank", 
            "Max Local Error", 
            "Mean Local Error"
        ]
        
        if self.input_data.is_debug:
            info_headers += ["Max Debug Error", "Mean Debug Error"]

        als_info(" | ".join(info_headers))
        return

    def _print_info(self, cross_iter: int) -> None:
        """Prints some diagnostic information about the current cross 
        iteration.
        """

        diagnostics = [
            f"{cross_iter+1:=4}", 
            f"{self.num_eval:=10}",
            f"{self.ranks.max():=8}",
            f"{self.tt.errors.max():=15.5e}",
            f"{self.tt.errors.mean():=16.5e}"
        ]

        if self.input_data.is_debug:
            diagnostics += [
                f"{self.linf_err:=15.5e}",
                f"{self.l2_err:=16.5e}"
            ]

        als_info(" | ".join(diagnostics))
        return

    def compute_error_estimates(self) -> None:
        """Computes the relative error between the value of the FTT 
        approximation to the target function and the true value for the 
        set of debugging samples.
        """

        if not self.input_data.is_debug:
            return
        
        ps_approx = self._eval_local(self.input_data.ls_debug, self.direction)
        ps_approx = ps_approx.flatten()
        self.l2_err, self.linf_err = self.input_data.relative_error(ps_approx)
        return

    def _eval_local_forward(self, ls: Tensor) -> Tensor:
        """Evaluates the FTT approximation to the target function for 
        the first k variables.
        """
        d_ls = ls.shape[1]
        Gs = [
            FTT._eval_core_213(self.bases[k], self.cores[k], ls[:, k])
            for k in range(d_ls)
        ]
        Gs_prod = batch_mul(*Gs).squeeze(dim=1)
        return Gs_prod
    
    def _eval_local_backward(self, ls: Tensor) -> Tensor:
        """Evaluates the FTT approximation to the target function for 
        the last k variables.
        """
        d_ls = ls.shape[1]
        Gs = [
            FTT._eval_core_213(self.bases[k], self.cores[k], ls[:, i])
            for i, k in enumerate(range(self.dim-d_ls, self.dim))
        ]
        Gs_prod = batch_mul(*Gs).squeeze(dim=2)
        return Gs_prod

    def _eval_local(self, ls: Tensor, direction: Direction) -> Tensor:
        """Evaluates the functional tensor train approximation to the 
        target function for either the first or last k variables, for a 
        set of points in the local domain ([-1, 1]).
        
        Parameters
        ----------
        ls:
            A n * d matrix containing a set of samples from the local 
            domain.
        direction:
            The direction in which to iterate over the cores.
        
        Returns
        -------
        Gs_prod:
            An n * n_k matrix, where each row contains the product of 
            the first or last (depending on direction) k tensor cores 
            evaluated at the corresponding sample in ls.
            
        """
        self._check_sample_dim(ls, self.dim)
        if direction == Direction.FORWARD:
            Gs_prod = self._eval_local_forward(ls)
        else: 
            Gs_prod = self._eval_local_backward(ls)
        return Gs_prod

    def eval(self, xs: Tensor) -> Tensor:
        """Evaluates the target function at a set of points in the 
        approximation domain.
        
        Parameters
        ----------
        xs:
            An n * d matrix containing samples from the approximation 
            domain.
            
        Returns
        -------
        gs:
            An n-dimensional vector containing the values of the 
            approximation to the target function function at each x 
            value.
        
        """
        FTT._check_sample_dim(xs, self.dim, strict=True)
        ls = self.bases.approx2local(xs)[0]
        gs = self._eval_local(ls, self.direction).flatten()
        return gs

    def _round(
        self, 
        tol: float | None = None, 
        max_rank: int | None = None
    ) -> None:
        self.tt.round(tol, max_rank)
        return


class FTT(TTFunc):
    """A multivariate functional tensor-train.

    General idea:
        Build TT. at the end of each TT sweep, evaluate the debug 
        samples to give an error estimate.

    """
    
    @property
    def direction(self) -> Direction:
        return self.tt.direction

    @property 
    def ranks(self) -> Tensor:
        """The ranks of each tensor core."""
        return self.tt.ranks
    
    @property
    def num_eval(self) -> int:
        return self.tt.num_eval
     
    def compute_cores(self) -> None:
        """(Re)-computes the FTT cores from the TT cores.
        """
        for k in range(self.dim):
            core = self.tt.cores[k].clone()
            if isinstance(basis := self.bases[k], Spectral):
                core = n_mode_prod(core, basis.node2basis, n=1)
            self.cores[k] = core
        return

    def build(self) -> None:
        """Builds the FTT approximation."""

        if self.options.verbose > 0:
            self._print_info_header()

        for num_iter in range(self.options.max_als): 

            self.tt.sweep()
            self.compute_cores()
            self.compute_error_estimates()

            if self.options.verbose > 0:
                self._print_info(num_iter)
            if self.is_finished:
                break
            
        if self.options.verbose > 0:
            als_info("ALS complete.")
        if self.options.verbose > 1:
            ranks = "-".join([str(int(r)) for r in self.ranks])
            msg = f"Final TT ranks: {ranks}."
            als_info(msg)
        
        return


class EFTT(TTFunc):
    """Extended functional tensor train.
    
    TODO: it could be nice if this could work with alternative TT 
    construction algorithms.
    """

    def __init__(
        self, 
        target_func: Callable[[Tensor], Tensor], 
        bases: ApproxBases, 
        options: TTOptions, 
        input_data: InputData, 
        reference: Reference,
        tt_data = None
    ):
        
        TTFunc.__init__(
            self, 
            target_func, 
            bases, 
            options, 
            input_data, 
            reference, 
            tt_data
        )

        self.num_eval_pod = 0
        self.pod_bases: Dict[int, Tensor] = {}
        self.deim_inds: Dict[int, Tensor] = {}     # DEIM indices for interpolating the reduced basis in each dimension
        self.factors: Dict[int, Tensor] = {}       # Tucker factor matrices in each dimension 
        return
    
    @property
    def num_eval(self) -> int:
        return self.tt.num_eval + self.num_eval_pod
    
    @property 
    def basis_dims(self) -> Tensor:
        """Returns a tensor containing the dimension of the reduced 
        basis for each coordinate.
        """
        basis_dims = torch.tensor([self.pod_bases[k].shape[1] 
                                   for k in range(self.dim)])
        return basis_dims
    
    def _compute_pod_bases(self):
        """Computes the POD bases in each dimension.
        
        TODO: add an option to set the number of samples here.
        TODO: add an option to set the tolerance here.

        TODO: give this a more descriptive name--it also does the DEIM 
        indices too... (perhaps this part of the code could be a 
        separate function).

        TODO: the samples could be drawn directly from the 
        reference rather than the (weighted) grid.

        """

        N = 25  # number of snaphots

        for k in range(self.dim):
            
            n_k = self.grid.points[k].numel()

            index_samples = self.grid.sample_indices(N)
            point_samples = self.grid.indices2points(index_samples)

            point_samples = point_samples.repeat((n_k, 1))
            point_samples[:, k] = self.grid.points[k].repeat_interleave(N)

            # Note: each column is a fibre
            fibre_matrix = self.target_func(point_samples).reshape(n_k, N)
            self.num_eval_pod += fibre_matrix.numel()

            # NOTE: if the matrix is wide (i.e., more POD samples than 
            # interpolation points), computing the eigendecomposition
            # of FF' is better here.
            tol = 1e-6
            basis_k = tsvd(fibre_matrix, tol=tol)[0]

            msg = f"Computing reduced basis for dimension {k+1} / {self.dim}..."
            als_info(msg, end="\r")

            self.pod_bases[k] = basis_k
            self.deim_inds[k], self.factors[k] = deim(basis_k)

        print("", end="\r")
        
        if self.options.verbose > 1:
            basis_dims = f"-".join([str(int(d)) for d in self.basis_dims])
            als_info(f"Reduced basis dimensions: {basis_dims}.")

        return
    
    def compute_cores(self) -> None:
        """(Re)-computes the FTT cores from the TT cores.
        """
        for k in range(self.dim):
            core = n_mode_prod(self.tt.cores[k], self.factors[k], n=1)
            if isinstance(basis := self.bases[k], Spectral):
                core = n_mode_prod(core, basis.node2basis, n=1)
            self.cores[k] = core
        return

    def build(self):
        """Steps to take:
        
        1. build snapshot matrices in each dimension
        2. compute bases in each dimension
        3. compute index sets using DEIM...
        4. compute TT decomposition of reduced tensor.

        """ 

        self._compute_pod_bases()

        deim_nodes = {k: self.bases[k].nodes[self.deim_inds[k]] for k in range(self.dim)}
        deim_weights = compute_weights(deim_nodes, self.bases.domain, self.reference)
        deim_grid = Grid(deim_nodes, deim_weights)

        self.tt = TT(self.target_func, deim_grid, self.options)

        if self.options.verbose > 0:
            self._print_info_header()

        num_iter = 0

        for num_iter in range(self.options.max_als): 

            self.tt.sweep()
            self.compute_cores()
            self.compute_error_estimates()
            
            if self.options.verbose > 0:
                self._print_info(num_iter)

            if self.is_finished:
                break
            
        if self.options.verbose > 0:
            als_info("ALS complete.")
        if self.options.verbose > 1:
            ranks = "-".join([str(int(r)) for r in self.ranks])
            msg = f"Final TT ranks: {ranks}."
            als_info(msg)
        
        return
