from typing import Callable, Dict

import torch
from torch import linalg
from torch import Tensor

from .approx_bases import ApproxBases
from .directions import Direction
from .tt import Grid, TT
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
    
    reference_weights = {}
    for k in grid_points:
        nodes_approx_k = domain.local2approx(grid_points[k])[0]
        reference_weights[k] = reference.eval_pdf(nodes_approx_k)[0]
    
    return reference_weights


class FTT():
    r"""A functional tensor train.

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
        is_finished = max_core_error < self.tt.options.als_tol
        if self.l2_error:
            error_target_met = self.l2_error < self.tt.options.als_tol
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
    def eval_core(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates a tensor core."""
        r_p, n_k, r_k = A.shape
        n_ls = ls.numel()
        coeffs = A.permute(1, 0, 2).reshape(n_k, r_p * r_k)
        Gs = poly.eval_radon(coeffs, ls).reshape(n_ls, r_p, r_k)
        return Gs
    
    @staticmethod
    def eval_core_deriv(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the derivative of a tensor core."""
        r_p, n_k, r_k = A.shape 
        n_ls = ls.numel()
        coeffs = A.permute(1, 0, 2).reshape(n_k, r_p * r_k)
        dGdls = poly.eval_radon_deriv(coeffs, ls).reshape(n_ls, r_p, r_k)
        return dGdls

    @staticmethod
    def _eval_core_213(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates a tensor core.
        """
        return FTT.eval_core(poly, A, ls)

    @staticmethod
    def _eval_core_213_deriv(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the derivative of a tensor core.
        """
        return FTT._eval_core_213_deriv(poly, A, ls)

    @staticmethod
    def _eval_core_231(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates a tensor core.
        """
        return FTT.eval_core(poly, A, ls).swapdims(1, 2)
    
    @staticmethod
    def _eval_core_231_deriv(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the derivative of a tensor core.
        """
        return FTT.eval_core_deriv(poly, A, ls).swapdims(1, 2)

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
        fls_ftt = self.eval(self.ls_error).flatten()
        self.l2_error = (
            linalg.norm(self.fls_error - fls_ftt) / linalg.norm(self.fls_error)
        )
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
        self.ls_error = self.bases.sample_measure(self.num_error_samples)[0]
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
        
        """

        self.target_func = target_func

        # Build grid (TODO: add an option to weight by the reference 
        # measure when sampling initial index sets).
        points = {k: self.bases[k].nodes for k in range(self.dim)}
        if reference is None:
            print("warning, no reference passed in..")
            grid = Grid(points)
        else:
            weights = compute_weights(points, reference.domain, reference)
            grid = Grid(points, weights)

        self.tt.initialise(target_func, grid)
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
            ranks = "-".join([str(int(r)) for r in self.ranks])
            msg = f"Final TT ranks: {ranks}."
            als_info(msg)
        
        return
    
    def clone(self):

        tt = TT(self.tt.options)
        tt.cores = {k: self.tt.cores[k].clone() for k in self.tt.cores}
        tt.index_sets = {k: self.tt.index_sets[k].clone() for k in self.tt.index_sets}
        tt.direction = self.tt.direction

        ftt = FTT(self.bases, tt)
        return ftt


class EFTT(FTT):
    """Extended functional tensor train.
    
    Parameters
    ----------
    TODO

    """

    def __init__(
        self, 
        bases: ApproxBases,
        tt: TT,
        num_error_samples: int = 1000, 
        num_pod_samples: int = 30,
        tol_pod: float = 1e-12
    ):
        FTT.__init__(self, bases, tt, num_error_samples)
        self.num_pod_samples = num_pod_samples
        self.tol_pod = tol_pod
        self.num_eval_pod = 0
        self.pod_bases: Dict[int, Tensor] = {}
        self.deim_inds: Dict[int, Tensor] = {}
        self.factors: Dict[int, Tensor] = {}
        return
    
    @property
    def num_eval(self) -> int:
        return self.num_error_samples + self.num_eval_pod + self.tt.num_eval
    
    @property 
    def basis_dims(self) -> Tensor:
        """Returns a tensor containing the dimension of the reduced 
        basis for each coordinate.
        """
        basis_dims = torch.tensor([self.pod_bases[k].shape[1] 
                                   for k in range(self.dim)])
        return basis_dims
    
    def compute_reduced_indices(self):
        """Computes the POD bases in each dimension.

        TODO: the samples should be drawn directly from the 
        reference (or from [-1, 1]^d) rather than the (weighted or 
        unweighted) grid.

        """

        points = {k: self.bases[k].nodes for k in range(self.dim)}
        grid = Grid(points)

        for k in range(self.dim):
            
            n_k = grid.points[k].numel()

            # index_samples = grid.sample_indices(self.num_pod_samples)
            # point_samples = grid.indices2points(index_samples)
            # TEMP...
            # TODO: eventually these should probably be from the reference.
            point_samples = 2.0 * torch.rand((self.num_pod_samples, self.dim)) - 1.0

            point_samples = point_samples.repeat((n_k, 1))
            point_samples[:, k] = grid.points[k].repeat_interleave(self.num_pod_samples)

            # Note: each column is a fibre
            fibre_matrix = self.target_func(point_samples).reshape(n_k, self.num_pod_samples)
            self.num_eval_pod += fibre_matrix.numel()

            # NOTE: if the matrix is wide (i.e., more POD samples than 
            # interpolation points), computing the eigendecomposition
            # of FF' is better here.
            basis_k = tsvd(fibre_matrix, tol=self.tol_pod)[0]

            if self.tt.options.verbose > 1:
                msg = f"Computing reduced basis for dimension {k+1} / {self.dim}..."
                als_info(msg, end="\r")

            self.pod_bases[k] = basis_k
            self.deim_inds[k], self.factors[k] = deim(basis_k)
        
        if self.tt.options.verbose > 1:
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

    def approximate(self, target_func: Callable[[Tensor], Tensor]) -> None:

        self.target_func = target_func
        self.compute_reduced_indices()

        deim_nodes = {k: self.bases[k].nodes[self.deim_inds[k]] for k in range(self.dim)}
        # deim_weights = compute_weights(deim_nodes, self.bases.domain, self.reference)
        deim_grid = Grid(deim_nodes)#, deim_weights)

        # TODO: all of the below is common to the FTT and EFTT 
        # implementations and could go into the parent class.
        self.tt.initialise(self.target_func, deim_grid)
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
            ranks = "-".join([str(int(r)) for r in self.ranks])
            msg = f"Final TT ranks: {ranks}."
            als_info(msg)
        return