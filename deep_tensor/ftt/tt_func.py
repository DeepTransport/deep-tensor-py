from typing import Callable, Tuple
import warnings

import torch
from torch import Tensor
from torch import linalg
from torch.autograd.functional import jacobian

from .approx_bases import ApproxBases
from .directions import Direction
from .input_data import InputData
from .tt_data import TTData
from ..constants import EPS
from ..linalg import (
    batch_mul, 
    cartesian_prod, 
    fold_left, fold_right, 
    unfold_left, unfold_right,
    tsvd
)
from ..options import TTOptions
from ..polynomials import Basis1D, Piecewise, Spectral
from ..tools import deim, maxvol
from ..tools.printing import als_info


INTERPOLATION_METHODS = {"deim": deim, "maxvol": maxvol}
MAX_COND = 1.0e+5


def _check_condition_number(U: Tensor) -> None:
    if (cond := linalg.cond(U)) > MAX_COND:
        msg = f"Poor condition number in interpolation: {cond}."
        warnings.warn(msg)
    return


class AbstractTTFunc(object):

    @property 
    def bases(self) -> ApproxBases:
        return self._bases 
    
    @bases.setter 
    def bases(self, value: ApproxBases) -> None:
        self._bases = value 
        return
    
    @property
    def options(self) -> TTOptions:
        return self._options
    
    @options.setter 
    def options(self, value: TTOptions) -> None:
        self._options = value 
        return
    
    @property
    def input_data(self) -> InputData:
        return self._input_data 
    
    @input_data.setter 
    def input_data(self, value: InputData) -> None:
        self._input_data = value 
        return
    
    @property
    def tt_data(self) -> TTData:
        return self._tt_data 
    
    @tt_data.setter 
    def tt_data(self, value: TTData) -> None:
        self._tt_data = value 
        return
    
    @property
    def dim(self) -> int:
        return self.bases.dim

    @property 
    def rank(self) -> Tensor:
        """The ranks of each tensor core."""
        return self.tt_data.rank

    @property
    def use_amen(self) -> bool:
        """Whether to use AMEN."""
        return self.options.tt_method.lower() == "amen"
        
    @property
    def sample_size(self):
        """An upper bound on the total number of samples required to 
        construct a FTT approximation to the target function.
        """
        n = self.dim * (self.options.init_rank 
                        + self.options.kick_rank * (self.options.max_als + 1))
        return n

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
    def unfold(H: Tensor, direction: Direction) -> Tensor:
        """Unfolds a tensor.
        """
        if direction == Direction.FORWARD:
            H = unfold_left(H)
        else: 
            H = unfold_right(H)
        return H
    
    @staticmethod
    def eval_core_213(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates a tensor core.

        Parameters
        ----------
        poly:
            The basis functions associated with the core.
        A:
            The coefficient tensor associated with the current core.
        ls: 
            An n-dimensional vector of points at which to evaluate the 
            current core.

        Returns
        -------
        Gs:
            A matrix of dimension n * r_{k-1} * r_{k}, containing the 
            evaluation of the kth core at each value of ls.
        
        """
        r_p, n_k, r_k = A.shape
        n_ls = ls.numel()
        coeffs = A.permute(1, 0, 2).reshape(n_k, r_p * r_k)
        Gs = poly.eval_radon(coeffs, ls).reshape(n_ls, r_p, r_k)
        return Gs

    @staticmethod
    def eval_core_213_deriv(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the derivative of a tensor core.

        Parameters
        ----------
        poly:
            The basis functions associated with the current dimension.
        A:
            The coefficient tensor associated with the current core.
        ls: 
            A vector of points at which to evaluate the current core.

        Returns
        -------
        dGdls:
            A matrix of dimension n_{k} * r_{k-1} * r_{k}, 
            corresponding to evaluations of the derivative of the kth 
            core at each value of ls.
        
        """
        r_p, n_k, r_k = A.shape 
        n_ls = ls.numel()
        coeffs = A.permute(1, 0, 2).reshape(n_k, r_p * r_k)
        dGdls = poly.eval_radon_deriv(coeffs, ls).reshape(n_ls, r_p, r_k)
        return dGdls

    @staticmethod
    def eval_core_231(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the kth tensor core at a given set of values.

        Parameters
        ----------
        poly:
            The basis functions associated with the current dimension.
        A:
            The coefficient tensor associated with the current core.
        ls: 
            A vector of points at which to evaluate the current core.

        Returns
        -------
        Gs:
            A tensor of dimension n_{k} * r_{k} * r_{k-1}, 
            corresponding to evaluations of the kth core at each value 
            of ls.
        
        """
        r_p, n_k, r_k = A.shape
        n_ls = ls.numel()
        coeffs = A.permute(1, 2, 0).reshape(n_k, r_p * r_k)
        Gs = poly.eval_radon(coeffs, ls).reshape(n_ls, r_k, r_p)
        return Gs
    
    @staticmethod
    def eval_core_231_deriv(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates the derivative of the kth tensor core at a given 
        set of values.

        Parameters
        ----------
        poly:
            The basis functions associated with the current dimension.
        A:
            The coefficient tensor associated with the current core.
        ls: 
            A vector of points at which to evaluate the current core.

        Returns
        -------
        dGdls:
            A tensor of dimension n_{k} * r_{k} * r_{k-1}, 
            corresponding to evaluations of the derivative of the kth 
            core at each value of ls.
        
        """
        r_p, n_k, r_k = A.shape
        n_ls = ls.numel()
        coeffs = A.permute(1, 2, 0).reshape(n_k, r_p * r_k)
        dGdls = poly.eval_radon_deriv(coeffs, ls).reshape(n_ls, r_k, r_p)
        return dGdls
    
    @staticmethod
    def _apply_mass_R(poly: Basis1D, H: Tensor) -> Tensor:

        # Mass matrix for spectral polynomials is the identity
        if isinstance(poly, Spectral):
            return H
        
        nr_k = H.shape[0]
        H = poly.mass_R @ H.T.reshape(-1, poly.cardinality).T
        H = H.T.reshape(-1, nr_k).T
        return H

    @staticmethod
    def _apply_mass_R_inv(poly: Basis1D, U: Tensor) -> Tensor:
        
        # Mass matrix for spectral polynomials is the identity
        if isinstance(poly, Spectral):
            return U

        nr_k = U.shape[0]
        U = U.T.reshape(-1, poly.cardinality).T
        U = linalg.solve(poly.mass_R, U)
        U = U.T.reshape(-1, nr_k).T
        return U

    def _eval_local_forward(self, ls: Tensor) -> Tensor:
        """Evaluates the FTT approximation to the target function for 
        the first k variables.
        """
        d_ls = ls.shape[1]
        polys = self.bases.polys
        cores = self.tt_data.cores
        Gs = [
            TTFunc.eval_core_213(polys[k], cores[k], ls[:, k])
            for k in range(d_ls)
        ]
        # Take the product of the cores and remove the second dimension 
        # (which will always be r_0 = 1)
        Gs_prod = batch_mul(*Gs).sum(dim=1)
        return Gs_prod
    
    def _eval_local_backward(self, ls: Tensor) -> Tensor:
        """Evaluates the FTT approximation to the target function for 
        the last k variables.
        """
        d_ls = ls.shape[1]
        polys = self.bases.polys
        cores = self.tt_data.cores
        Gs = [
            TTFunc.eval_core_213(polys[k], cores[k], ls[:, i])
            for i, k in enumerate(range(self.dim-d_ls, self.dim))
        ]
        # Take the product of the cores and remove the third dimension 
        # (which will always be r_0 = 1)
        Gs_prod = batch_mul(*Gs).sum(dim=2)
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

    def _eval(self, xs: Tensor) -> Tensor:
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
        TTFunc._check_sample_dim(xs, self.dim, strict=True)
        ls = self.bases.approx2local(xs)[0]
        gs = self._eval_local(ls, self.tt_data.direction).flatten()
        return gs

    def _grad_local(self, ls: Tensor) -> Tensor:
        """Evaluates the gradient of the approximation to the target 
        function for a set of samples in the local domain.

        Parameters
        ----------
        ls:
            An n * d matrix containing a set of samples in the local 
            domain.
        
        Returns
        -------
        dfdls:
            An n * d matrix containing the gradient of the FTT 
            approximation to the target function evaluated at each 
            element in ls.

        """

        polys = self.bases.polys
        cores = self.tt_data.cores
        n_ls = ls.shape[0]
        
        dGdls = {k: torch.ones((n_ls, 1, 1)) for k in range(self.dim)}
        
        for k in range(self.dim):
            Gs_k = TTFunc.eval_core_213(polys[k], cores[k], ls[:, k])
            dGdls_k = TTFunc.eval_core_213_deriv(polys[k], cores[k], ls[:, k])
            for j in range(self.dim):
                if k == j:
                    dGdls[j] = batch_mul(dGdls[j], dGdls_k)
                else:
                    dGdls[j] = batch_mul(dGdls[j], Gs_k)
        
        dfdls = torch.zeros_like(ls)
        for k in range(self.dim):
            dfdls[:, k] = dGdls[k].sum(dim=(1, 2))
        return dfdls
    
    def _grad_autodiff(self, xs: Tensor) -> Tensor:
        
        n_xs = xs.shape[0]

        def _grad(xs: Tensor) -> Tensor:
            xs = xs.reshape(n_xs, self.dim)
            return self._eval(xs).sum(dim=0)
        
        derivs = jacobian(_grad, xs.flatten(), vectorize=True)
        return derivs.reshape(n_xs, self.dim)

    def _grad(self, xs: Tensor, method: str = "autodiff") -> Tensor:
        """Evaluates the gradient of the approximation to the target 
        function at a set of points in the approximation domain.
        
        Parameters
        ----------
        xs: 
            A matrix containing n sets of d-dimensional input 
            variables in the approximation domain. Each row contains a
            single input variable.

        Returns
        -------
        dfdxs:
            An n * d matrix containing the gradient of the FTT 
            approximation to the target function evaluated at each 
            element in ls.

        """

        method = method.lower()
        if method not in ("manual", "autodiff"):
            raise Exception("Unknown method.")
        
        TTFunc._check_sample_dim(xs, self.dim, strict=True)

        if method == "autodiff":
            dfdxs = self._grad_autodiff(xs)
            return dfdxs
        
        ls, dldxs = self.bases.approx2local(xs)
        dfdls = self._grad_local(ls)
        dfdxs = dfdls * dldxs
        return dfdxs

    def _round(self, tol: float | Tensor | None = None) -> None:
        """Rounds the TT cores. Applies double rounding to get back to 
        the starting direction.

        Parameters
        ----------
        tol:
            The tolerance to use when applying truncated SVD to round 
            each core.
        
        """

        if tol is None:
            tol = self.options.local_tol

        for _ in range(2):
            
            self.tt_data._reverse_direction()

            if self.tt_data.direction == Direction.FORWARD:
                inds = range(self.dim-1)
            else:
                inds = range(self.dim-1, 0, -1)

            for k in inds:
                self._build_basis_svd(self.tt_data.cores[k], k, tol)

        if self.use_amen:
            self.tt_data.res_w = {}
            self.tt_data.res_x = {}
        return
    
    def _truncate_local(
        self, 
        H: Tensor, 
        tol: float | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
        Ur, sr, Vhr, rank = tsvd(H, tol, self.options.max_rank)
        sVhr = sr[:, None] * Vhr
        return Ur, sVhr, rank
    
    def _build_basis_svd(
        self, 
        H: Tensor, 
        k: int, 
        tol: float | None = None
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

        Returns
        -------
        None
            
        """

        k_prev = k - self.tt_data.direction.value
        k_next = k + self.tt_data.direction.value
        r_p, n_k, r_k = H.shape
        
        poly = self.bases.polys[k]
        ls_int_p = self.tt_data.interp_ls[k_prev]
        A_next = self.tt_data.cores[k_next]

        if self.tt_data.direction == Direction.FORWARD:
            H = unfold_left(H)
        else: 
            H = unfold_right(H)

        # Compute an M-orthogonal basis and truncate
        H = self._apply_mass_R(poly, H)
        U, sVh, rank = self._truncate_local(H, tol)
        U = self._apply_mass_R_inv(poly, U)

        # Select a set of interpolation points
        inds, B, U_interp = self._select_points(U, k)
        ls_int_k = self._get_local_index(poly, ls_int_p, inds)
        couple = U_interp @ sVh

        # Form the current coefficient tensor and update the dimensions 
        # of the next one
        if self.tt_data.direction == Direction.FORWARD:
            self.tt_data.cores[k] = fold_left(B, (r_p, n_k, rank))
            r_next = A_next.shape[0]
            A_next = torch.einsum("il, ljk", couple[:, :r_next], A_next)
        else:
            self.tt_data.cores[k] = fold_right(B, (rank, n_k, r_k))
            r_next = A_next.shape[2]
            A_next = torch.einsum("ijl, kl", A_next, couple[:, :r_next])

        self.tt_data.cores[k_next] = A_next
        self.tt_data.interp_ls[k] = ls_int_k 
        return
    
    def _build_basis_amen(
        self, 
        H: Tensor,
        H_res: Tensor,
        H_up: Tensor,
        k: Tensor | int
    ) -> None:
        """Computes the coefficients of the kth tensor core."""
        
        k = int(k)
        k_prev = int(k - self.tt_data.direction.value)
        k_next = int(k + self.tt_data.direction.value)
        
        poly = self.bases.polys[k]
        interp_ls_prev = self.tt_data.interp_ls[k_prev]
        res_x_prev = self.tt_data.res_x[k_prev]

        res_w_prev = self.tt_data.res_w[k-1]
        res_w_next = self.tt_data.res_w[k+1]

        A_next = self.tt_data.cores[k_next]

        n_left, n_k, n_right = H.shape
        r_0_next, _, r_1_next = A_next.shape

        H = TTFunc.unfold(H, self.tt_data.direction)
        H = self._apply_mass_R(poly, H)
        U, sVh, rank = self._truncate_local(H)
        U = self._apply_mass_R_inv(poly, U)

        H_up = TTFunc.unfold(H_up, self.tt_data.direction)

        if self.tt_data.direction == Direction.FORWARD:
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

        T = self._apply_mass_R(poly, T)
        U, R = linalg.qr(T)
        U = self._apply_mass_R_inv(poly, U)

        r_new = U.shape[1]

        indices, B, U_interp = self._select_points(U, k)
        couple = U_interp @ R[:r_new, :rank] @ sVh

        interp_ls = self._get_local_index(poly, interp_ls_prev, indices)

        error_tol = self.options.local_tol * EPS
        U_res = self._truncate_local(H_res, error_tol)[0]
        inds_res = self._select_points(U_res, k)[0]
        res_x = self._get_local_index(poly, res_x_prev, inds_res)

        if self.tt_data.direction == Direction.FORWARD:
            
            A = fold_left(B, (n_left, n_k, r_new))

            temp = torch.einsum("il, ljk", res_w_prev, A)
            temp = unfold_left(temp)
            res_w = temp[inds_res]

            couple = couple[:, :r_0_next]
            A_next = torch.einsum("il, ljk", couple, A_next)

        else:
            
            A = fold_right(B, (r_new, n_k, n_right))

            temp = torch.einsum("ijl, lk", A, res_w_next)
            temp = unfold_right(temp)
            res_w = temp[inds_res].T

            couple = couple[:, :r_1_next]
            A_next = torch.einsum("ijl, kl", A_next, couple)

        self.tt_data.cores[k] = A 
        self.tt_data.cores[k_next] = A_next
        self.tt_data.interp_ls[k] = interp_ls
        self.tt_data.res_w[k] = res_w 
        self.tt_data.res_x[k] = res_x
        return

    def _get_local_index(
        self,
        poly: Basis1D, 
        ls_int_p: Tensor,
        inds: Tensor
    ) -> Tensor:
        """Updates the set of interpolation points for the current 
        dimension.
        
        Parameters
        ----------
        poly:
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

        if ls_int_p.numel() == 0:
            ls_int_k = poly.nodes[inds][:, None]
            return ls_int_k

        n_k = poly.cardinality
        ls_prev = ls_int_p[inds // n_k]
        ls_nodes = poly.nodes[inds % n_k][:, None]

        if self.tt_data.direction == Direction.FORWARD:
            ls_int_k = torch.hstack((ls_prev, ls_nodes))
        else:
            ls_int_k = torch.hstack((ls_nodes, ls_prev))

        return ls_int_k

    def _select_points_piecewise(
        self,
        U: Tensor,
        poly: Piecewise
    ) -> Tuple[Tensor, Tensor, Tensor]:
        
        inds, B = INTERPOLATION_METHODS[self.options.int_method](U)
        U_interp = U[inds]
        _check_condition_number(U_interp)
        
        return inds, B, U_interp
    
    def _select_points_spectral(
        self,
        U: Tensor,
        poly: Spectral
    ) -> Tuple[Tensor, Tensor, Tensor]:

        n_k = poly.cardinality
        r_p = round(U.shape[0] / n_k)

        # Undo the previous application of the node2basis 
        nodes = poly.basis2node @ U.T.reshape(-1, n_k).T
        nodes = nodes.T.reshape(-1, n_k * r_p).T

        inds, _ = INTERPOLATION_METHODS[self.options.int_method](nodes)
        U_interp = nodes[inds]
        _check_condition_number(U_interp)

        # Compute unfolded coefficient tensor B = UU[I, :]^{-1}
        B = linalg.solve(U_interp, U, left=False)
        return inds, B, U_interp

    def _select_points(self, U: Tensor, k: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Builds the cross indices.

        Parameters
        ----------
        U:
            The set of left singular vectors obtained from a truncated 
            SVD of the unfolding matrix of tensor H (which contains a
            set of evaluations of the target function at the current 
            set of interpolation points).
        k: 
            The index of the current dimension.
        
        Returns
        -------
        inds:
            The set of indices of the (approximate) maximum volume 
            submatrix of H.
        B:
            The corresponding (unfolded) tensor core.
        U_interp:
            The nodes of the basis of the current dimension 
            corresponding to the set of indices of the maximum volume
            submatrix.
        
        """

        poly = self.bases.polys[k]

        if isinstance(poly, Piecewise):
            return self._select_points_piecewise(U, poly)
        elif isinstance(poly, Spectral):
            return self._select_points_spectral(U, poly)
    
        raise Exception("Unknown polynomial encountered.")


class TTFunc(AbstractTTFunc):
    """A multivariate functional tensor-train.

    Parameters
    ----------
    target_func:
        Maps an n * d matrix containing samples from the local domain 
        to an n-dimensional vector containing the values of the target 
        function at each sample.
    bases:
        The bases associated with the approximation domain.
    options:
        Options used when constructing the FTT approximation to the 
        target function.
    input_data:
        Data used for initialising and evaluating the quality of the 
        FTT approximation to the target function.
    tt_data:
        Data used to construct the FTT approximation to the target
        function.

    """

    def __init__(
        self, 
        target_func: Callable[[Tensor], Tensor], 
        bases: ApproxBases, 
        options: TTOptions, 
        input_data: InputData,
        tt_data: TTData | None = None
    ):

        if tt_data is None:
            tt_data = TTData()
        
        self.target_func = target_func
        self.bases = bases 
        self.options = options
        self.input_data = input_data
        self.tt_data = tt_data
        self.num_eval = 0
        self.errors = torch.zeros(self.dim)
        self.l2_err = torch.inf
        self.linf_err = torch.inf

        self.input_data.set_samples(self.bases, self.sample_size)
        if self.input_data.is_debug:
            self.input_data.set_debug(self.target_func, self.bases)
        
        return

    def _initialise_cores(self) -> None:
        """Initialises the cores and interpolation points in each 
        dimension.
        """

        for k in range(self.dim):

            core_shape = [
                1 if k == 0 else self.options.init_rank, 
                self.bases.polys[k].cardinality,
                1 if k == self.dim-1 else self.options.init_rank
            ]
            self.tt_data.cores[k] = torch.zeros(core_shape)

            samples = self.input_data.get_samples(self.options.init_rank)
            self.tt_data.interp_ls[k] = samples[:, k:]

        self.tt_data.interp_ls[-1] = torch.tensor([])
        self.tt_data.interp_ls[self.dim] = torch.tensor([])
        return

    def _initialise_res_x(self) -> None:
        """Initialises the residual coordinates for AMEN."""

        for k in range(self.dim-1, -1, -1):
            samples = self.input_data.get_samples(self.options.kick_rank)
            if self.tt_data.direction == Direction.FORWARD:
                self.tt_data.res_x[k] = samples[:, k:]
            else:
                self.tt_data.res_x[k] = samples[:, :(k+1)]

        self.tt_data.res_x[-1] = torch.tensor([])
        self.tt_data.res_x[self.dim] = torch.tensor([])
        return
    
    def _initialise_res_w(self) -> None:
        """Initialises the residual blocks for AMEN."""

        if self.tt_data.direction == Direction.FORWARD:
            
            core_0 = self.tt_data.cores[0]
            shape_0 = (self.options.kick_rank, core_0.shape[-1])
            self.tt_data.res_w[0] = torch.ones(shape_0)
            
            for k in range(1, self.dim):
                core_k = self.tt_data.cores[k].shape[0]
                shape_k = (core_k, self.options.kick_rank)
                self.tt_data.res_w[k] = torch.ones(shape_k)

        else:

            for k in range(self.dim-1):
                core_k = self.tt_data.cores[k]
                shape_k = (self.options.kick_rank, core_k.shape[-1])
                self.tt_data.res_w[k] = torch.ones(shape_k)

            core_d = self.tt_data.cores[self.dim-1]
            shape_d = (core_d.shape[0], self.options.kick_rank)
            self.tt_data.res_w[self.dim-1] = torch.ones(shape_d)

        self.tt_data.res_w[-1] = torch.tensor([[1.0]])
        self.tt_data.res_w[self.dim] = torch.tensor([[1.0]])
        return

    def _initialise_amen(self) -> None:
        """Initialises the residual coordinates and residual blocks 
        for AMEN.
        """
        if self.tt_data.res_x == {}:
            self._initialise_res_x()
        if self.tt_data.res_w == {}:
            self._initialise_res_w()
        return

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

    def _print_info(self, cross_iter: int, indices: Tensor) -> None:
        """Prints some diagnostic information about the current cross 
        iteration.
        """

        diagnostics = [
            f"{cross_iter:=4}", 
            f"{self.num_eval:=10}",
            f"{torch.max(self.rank):=8}",
            f"{torch.max(self.errors[indices]):=15.5e}",
            f"{torch.mean(self.errors[indices]):=16.5e}"
        ]

        if self.input_data.is_debug:
            diagnostics += [
                f"{self.linf_err:=15.5e}",
                f"{self.l2_err:=16.5e}"
            ]

        als_info(" | ".join(diagnostics))
        return

    def _compute_relative_error(self) -> None:
        """Computes the relative error between the value of the FTT 
        approximation to the target function and the true value for the 
        set of debugging samples.
        """

        if not self.input_data.is_debug:
            return
        
        ps_approx = self._eval_local(self.input_data.ls_debug, self.tt_data.direction)
        ps_approx = ps_approx.flatten()
        self.l2_err, self.linf_err = self.input_data.relative_error(ps_approx)
        return

    def _build_block_local(
        self, 
        ls_left: Tensor,
        ls_right: Tensor,
        k: int | Tensor
    ) -> Tensor:
        """Evaluates the function being approximated at a (reduced) set 
        of interpolation points, and returns the corresponding
        local coefficient matrix.

        Parameters
        ----------
        ls_left:
            An r_{k-1} * {k-1} matrix containing a set of interpolation
            points for dimensions 1, ..., {k-1}.
        ls_right:
            An r_{k+1} * {k+1} matrix containing a set of interpolation 
            points for dimensions {k+1}, ..., d.
        k:
            The dimension in which interpolation is being carried out.

        Returns
        -------
        H: 
            An r_{k-1} * n_{k} * r_{k} tensor containing the values of 
            the function evaluated at each interpolation point.

        References
        ----------
        Cui and Dolgov (2022). Deep composition of tensor-trains using 
        squared inverse Rosenblatt transports.
        
        """

        poly = self.bases.polys[int(k)]
        r_p = 1 if ls_left.numel() == 0 else ls_left.shape[0]
        r_k = 1 if ls_right.numel() == 0 else ls_right.shape[0]
        n_k = poly.n_nodes

        ls = cartesian_prod((ls_left, poly.nodes[:, None], ls_right))
        H = self.target_func(ls).reshape(r_p, n_k, r_k)

        # Convert from a tensor containing the target function value at 
        # each interpolation point to a tensor containing the 
        # coefficients associated with each basis function for each 
        # interpolation point. Note that node2basis for piecewise 
        # polynomials is an identity operator.
        if isinstance(poly, Spectral): 
            H = torch.einsum("jl, ilk", poly.node2basis, H)

        self.num_eval += ls.shape[0]
        return H

    def _get_error_local(self, H: Tensor, k: int) -> float:
        """Returns the error between the current core and the tensor 
        formed by evaluating the target function at the current set of 
        interpolation points corresponding to the core.

        Parameters
        ----------
        H:
            The tensor formed by evaluating the target function at the 
            current set of interpolation points corresponding to the 
            kth core.
        k:
            The current dimension.

        Returns
        -------
        error:
            The greatest absolute difference between an element of H 
            and the corresponding element of the core divided by the 
            absolute value of the element of H.

        """
        core = self.tt_data.cores[k]
        return float((core-H).abs().max() / H.abs().max())

    def _is_finished(self, cross_iter: int, indices: Tensor) -> bool:
        """Returns True if the maximum number of cross iterations has 
        been reached or the desired error tolerance is met, and False 
        otherwise.
        """
        max_iters = cross_iter == self.options.max_als
        max_error_tol = torch.max(self.errors[indices]) < self.options.als_tol
        l2_error_tol = self.l2_err < self.options.als_tol
        return bool(max_iters or max_error_tol or l2_error_tol)

    def _compute_cross_block_fixed(self, k: Tensor) -> None:
        
        ls_left = self.tt_data.interp_ls[k-1]
        ls_right = self.tt_data.interp_ls[k+1]
        
        H = self._build_block_local(ls_left, ls_right, k) 
        self.errors[k] = self._get_error_local(H, k)
        self._build_basis_svd(H, k)
        return
    
    def _compute_cross_block_random(self, k: Tensor) -> None:
        
        ls_left = self.tt_data.interp_ls[k-1].clone()
        ls_right = self.tt_data.interp_ls[k+1].clone()
        enrich = self.input_data.get_samples(self.options.kick_rank)

        H = self._build_block_local(ls_left, ls_right, k)
        self.errors[k] = self._get_error_local(H, k)

        if self.tt_data.direction == Direction.FORWARD:
            H_enrich = self._build_block_local(ls_left, enrich[:, k+1:], k)
            H_full = torch.concatenate((H, H_enrich), dim=2)
        else:
            H_enrich = self._build_block_local(enrich[:, :k], ls_right, k)
            H_full = torch.concatenate((H, H_enrich), dim=0)

        self._build_basis_svd(H_full, k)
        return
    
    def _compute_cross_block_amen(self, k: Tensor) -> None:
        
        ls_left = self.tt_data.interp_ls[k-1]
        ls_right = self.tt_data.interp_ls[k+1]
        r_left = self.tt_data.res_x[k-1]
        r_right = self.tt_data.res_x[k+1]

        # Evaluate the interpolant function at x_k nodes
        H = self._build_block_local(ls_left, ls_right, k)
        self.errors[k] = self._get_error_local(H, k)

        # Evaluate residual function at x_k nodes
        H_res = self._build_block_local(r_left, r_right, k)

        if self.tt_data.direction == Direction.FORWARD and k > 0:
            H_up = self._build_block_local(ls_left, r_right, k)
        elif self.tt_data.direction == Direction.BACKWARD and k < self.dim-1: 
            H_up = self._build_block_local(r_left, ls_right, k)
        else:
            H_up = H_res.clone()

        self._build_basis_amen(H, H_res, H_up, k)
        return 

    def _cross(self) -> None:
        """Builds the FTT using cross iterations."""

        cross_iter = 0

        if self.options.verbose > 0:
            self._print_info_header()

        if self.tt_data.cores == {}:
            self._initialise_cores()
        else:
            # Prepare for the next iteration
            self.tt_data._reverse_direction()
        
        if self.use_amen:
            self._initialise_amen()

        while True:

            if self.tt_data.direction == Direction.FORWARD:
                indices = torch.arange(self.dim-1)
            else:
                indices = torch.arange(self.dim-1, 0, -1)
            
            for i, k in enumerate(indices):
                if self.options.verbose > 1:
                    msg = f"Building block {i+1} / {self.dim}..."
                    als_info(msg, end="\r")
                if self.options.tt_method == "fixed_rank":
                    self._compute_cross_block_fixed(int(k))
                elif self.options.tt_method == "random":
                    self._compute_cross_block_random(int(k))
                elif self.options.tt_method == "amen":
                    self._compute_cross_block_amen(int(k))

            cross_iter += 1
            finished = self._is_finished(cross_iter, indices)
            
            if finished:
                if self.options.verbose > 1:
                    msg = f"Building block {self.dim} / {self.dim}..."
                    als_info(msg, end="\r")
                self._compute_final_block()
            
            self._compute_relative_error()
            if self.options.verbose > 0:
                self._print_info(cross_iter, indices)

            if finished:
                if self.options.verbose > 0:
                    als_info("ALS complete.")
                if self.options.verbose > 1:
                    ranks = "-".join([str(int(r)) for r in self.rank])
                    msg = f"Final TT ranks: {ranks}."
                    als_info(msg)
                return
            
            self.tt_data._reverse_direction()

    def _compute_final_block(self) -> None:
        """Computes the final block of the FTT approximation to the 
        target function.
        """

        if self.tt_data.direction == Direction.FORWARD:
            k = self.dim-1 
        else:
            k = 0

        ls_left = self.tt_data.interp_ls[k-1]
        ls_right = self.tt_data.interp_ls[k+1]
        H = self._build_block_local(ls_left, ls_right, k)
        self.errors[k] = self._get_error_local(H, k)
        self.tt_data.cores[k] = H
        return
    

class SavedTTFunc(TTFunc):
    """A saved functional tensor train.
    
    TODO: transfer statistics (number of function evaluations for 
    construction, etc.) from the previous FTT.
    """

    def __init__(
        self, 
        bases: ApproxBases, 
        options: TTOptions,
        input_data: InputData, 
        tt_data: TTData
    ):
        
        self.bases = bases 
        self.options = options
        self.input_data = input_data
        self.tt_data = tt_data
        return