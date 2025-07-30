from typing import Callable, Dict, Tuple, Sequence
import warnings

import torch
from torch import Tensor
from torch import linalg
from torch.autograd.functional import jacobian

from .approx_bases import ApproxBases
from .directions import Direction
from .input_data import InputData
from .tt_data import TTData
from ..linalg import (
    batch_mul, cartesian_prod, 
    fold_left, fold_right, 
    unfold_left, unfold_right,
    tsvd
)
from ..options import TTOptions
from ..polynomials import Basis1D, Spectral
from ..tools import deim, maxvol
from ..tools.printing import als_info


INTERPOLATION_METHODS = {"deim": deim, "maxvol": maxvol}
MAX_COND = 1.0e+5


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
        return self.tt_data._rank

    @property
    def _use_amen(self) -> bool:
        return self.options.tt_method.lower() == "amen"
        
    @property
    def _sample_size(self) -> int:
        """An upper bound on the total number of samples required to 
        construct a FTT approximation to the target function.
        """
        n = self.dim * (self.options.init_rank 
                        + self.options.kick_rank * (self.options.max_als + 1))
        return n
    
    @property 
    def cores(self) -> Dict[int, Tensor]:
        return self.tt_data.cores
    
    @property
    def coefs(self) -> Dict[int, Tensor]:
        return self.tt_data.coefs

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
    def _eval_core_213(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
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
    def _eval_core_213_deriv(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
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
    def _eval_core_231(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
        """Evaluates a tensor core.

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
        return TTFunc._eval_core_213(poly, A, ls).swapdims(1, 2)
    
    @staticmethod
    def _eval_core_231_deriv(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
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
            A tensor of dimension n_{k} * r_{k} * r_{k-1}, 
            corresponding to evaluations of the derivative of the kth 
            core at each value of ls.
        
        """
        return TTFunc._eval_core_213_deriv(poly, A, ls).swapdims(1, 2)

    @staticmethod
    def _coef2core(H: Tensor, basis: Basis1D) -> Tensor:
        """Converts from a tensor containing the target function value 
        at each interpolation point to a tensor containing the 
        coefficients associated with each basis function for each 
        interpolation point.
        """
        if isinstance(basis, Spectral): 
            return torch.einsum("jl, ilk", basis.node2basis, H)
        # node2basis is an identity operator for piecewise polynomials 
        return H.clone()

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

    @staticmethod
    def _get_error_local(H_new: Tensor, H_old: Tensor) -> float:
        """Returns the error between the current and previous 
        coefficient tensors.
        """
        return float((H_new-H_old).abs().max() / H_new.abs().max())

    def _eval_local_forward(self, ls: Tensor) -> Tensor:
        """Evaluates the FTT approximation to the target function for 
        the first k variables.
        """
        d_ls = ls.shape[1]
        Gs = [
            TTFunc._eval_core_213(self.bases[k], self.cores[k], ls[:, k])
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
            TTFunc._eval_core_213(self.bases[k], self.cores[k], ls[:, i])
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

        n_ls = ls.shape[0]
        
        dGdls = {k: torch.ones((n_ls, 1, 1)) for k in range(self.dim)}
        
        for k in range(self.dim):
            Gs_k = TTFunc._eval_core_213(self.bases[k], self.cores[k], ls[:, k])
            dGdls_k = TTFunc._eval_core_213_deriv(self.bases[k], self.cores[k], ls[:, k])
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
            return self.eval(xs).sum(dim=0)
        
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
        method: 
            'autodiff' or 'manual'.

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

    def _round(self, tol: float | None = None) -> None:
        """Rounds the TT cores.

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
                self._build_basis_svd(self.coefs[k], k, tol)

        if self._use_amen:
            self.tt_data.res_w = {}
            self.tt_data.res_x = {}
        return
    
    def _truncate_local(
        self, 
        H: Tensor, 
        tol: float | None = None
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
        tol:
            The tolerance to use when applying truncated SVD to the 
            unfolding matrix of H.

        Returns
        -------
        None
            
        """

        k_prev = k - self.tt_data.direction.value
        k_next = k + self.tt_data.direction.value
        r_p, n_k, r_k = H.shape
        
        ls_int_p = self.tt_data.interp_ls[k_prev]
        A_next = self.coefs[k_next]

        if self.tt_data.direction == Direction.FORWARD:
            H = unfold_left(H)
        else: 
            H = unfold_right(H)

        # Compute an M-orthogonal basis and truncate
        H = self._apply_mass_R(self.bases[k], H)
        U, sVh, rank = self._truncate_local(H, tol)
        U = self._apply_mass_R_inv(self.bases[k], U)

        # Select a set of interpolation points
        inds, B, U_interp = self._select_points(U)
        ls_int_k = self._get_local_index(self.bases[k], ls_int_p, inds)
        couple = U_interp @ sVh

        # Form the current coefficient tensor and update the next one
        if self.tt_data.direction == Direction.FORWARD:
            A = fold_left(B, (r_p, n_k, rank))
            r_next = A_next.shape[0]
            A_next = torch.einsum("il, ljk", couple[:, :r_next], A_next)
        else:
            A = fold_right(B, (rank, n_k, r_k))
            r_next = A_next.shape[2]
            A_next = torch.einsum("ijl, kl", A_next, couple[:, :r_next])

        self.coefs[k] = A
        self.coefs[k_next] = A_next
        self.tt_data.interp_ls[k] = ls_int_k 
        self.cores[k] = TTFunc._coef2core(A, self.bases[k])
        self.cores[k_next] = TTFunc._coef2core(A_next, self.bases[k_next])
        return
    
    def _build_basis_amen(
        self, 
        H: Tensor,
        H_res: Tensor,
        H_up: Tensor,
        k: int
    ) -> None:
        """Computes the coefficients of the kth tensor core."""
        
        k_prev = k - self.tt_data.direction.value
        k_next = k + self.tt_data.direction.value
        
        basis = self.bases[k]
        interp_ls_prev = self.tt_data.interp_ls[k_prev]
        res_x_prev = self.tt_data.res_x[k_prev]

        res_w_prev = self.tt_data.res_w[k-1]
        res_w_next = self.tt_data.res_w[k+1]

        A_next = self.coefs[k_next]

        n_left, n_k, n_right = H.shape
        r_0_next, _, r_1_next = A_next.shape

        if self.tt_data.direction == Direction.FORWARD:
            H = unfold_left(H)
            H_up = unfold_left(H_up)
        else:
            H = unfold_right(H)
            H_up = unfold_right(H_up)

        H = self._apply_mass_R(basis, H)
        U, sVh, rank = self._truncate_local(H)
        U = self._apply_mass_R_inv(basis, U)

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

        T = self._apply_mass_R(basis, T)
        U, R = linalg.qr(T)
        U = self._apply_mass_R_inv(basis, U)

        r_new = U.shape[1]

        indices, B, U_interp = self._select_points(U)
        couple = U_interp @ R[:r_new, :rank] @ sVh

        interp_ls = self._get_local_index(basis, interp_ls_prev, indices)

        U_res = self._truncate_local(H_res, tol=0.0)[0]
        inds_res = self._select_points(U_res)[0]
        res_x = self._get_local_index(basis, res_x_prev, inds_res)

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

        self.coefs[k] = A 
        self.coefs[k_next] = A_next
        self.tt_data.interp_ls[k] = interp_ls
        self.tt_data.res_w[k] = res_w 
        self.tt_data.res_x[k] = res_x
        self.cores[k] = TTFunc._coef2core(A, self.bases[k])
        self.cores[k_next] = TTFunc._coef2core(A_next, self.bases[k_next])
        return

    def _get_local_index(
        self,
        basis: Basis1D, 
        ls_int_p: Tensor,
        inds: Tensor
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

        if ls_int_p.numel() == 0:
            ls_int_k = basis.nodes[inds][:, None]
            return ls_int_k

        n_k = basis.cardinality
        ls_p = ls_int_p[inds // n_k]
        ls_nodes = basis.nodes[inds % n_k][:, None]

        if self.tt_data.direction == Direction.FORWARD:
            ls_int_k = torch.hstack((ls_p, ls_nodes))
        else:
            ls_int_k = torch.hstack((ls_nodes, ls_p))

        return ls_int_k

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
        if (cond := linalg.cond(U_sub)) > MAX_COND:
            msg = f"Poor condition number in interpolation: {cond}."
            warnings.warn(msg)
        return inds, B, U_sub


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
        self.n_eval = 0
        self.errors = torch.zeros(self.dim)
        self.l2_err = torch.inf
        self.linf_err = torch.inf

        self.input_data.set_samples(self.bases, self._sample_size)
        if self.input_data.is_debug:
            self.input_data.set_debug(self.target_func, self.bases)
        
        return

    def _initialise_coefs(self) -> None:
        """Initialises the cores and interpolation points in each 
        dimension.
        """

        for k in range(self.dim):

            coef_shape = [
                1 if k == 0 else self.options.init_rank, 
                self.bases[k].cardinality,
                1 if k == self.dim-1 else self.options.init_rank
            ]
            self.coefs[k] = torch.zeros(coef_shape)
            self.cores[k] = torch.zeros(coef_shape)

            ls_samp = self.input_data.get_samples(self.options.init_rank)
            self.tt_data.interp_ls[k] = ls_samp[:, k:]

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
            
            coef_0 = self.coefs[0]
            shape_0 = (self.options.kick_rank, coef_0.shape[-1])
            self.tt_data.res_w[0] = torch.ones(shape_0)
            
            for k in range(1, self.dim):
                coef_k = self.coefs[k].shape[0]
                shape_k = (coef_k, self.options.kick_rank)
                self.tt_data.res_w[k] = torch.ones(shape_k)

        else:

            for k in range(self.dim-1):
                coef_k = self.coefs[k]
                shape_k = (self.options.kick_rank, coef_k.shape[-1])
                self.tt_data.res_w[k] = torch.ones(shape_k)

            coef_d = self.coefs[self.dim-1]
            shape_d = (coef_d.shape[0], self.options.kick_rank)
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

    def _print_info(self, cross_iter: int, inds: Sequence) -> None:
        """Prints some diagnostic information about the current cross 
        iteration.
        """

        diagnostics = [
            f"{cross_iter:=4}", 
            f"{self.n_eval:=10}",
            f"{self.rank.max():=8}",
            f"{self.errors[inds].max():=15.5e}",
            f"{self.errors[inds].mean():=16.5e}"
        ]

        if self.input_data.is_debug:
            diagnostics += [
                f"{self.linf_err:=15.5e}",
                f"{self.l2_err:=16.5e}"
            ]

        als_info(" | ".join(diagnostics))
        return

    def _compute_rel_error(self) -> None:
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

    def _build_block_local(self, ls_left: Tensor, ls_right: Tensor, k: int) -> Tensor:
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
        
        """

        r_p = 1 if ls_left.numel() == 0 else ls_left.shape[0]
        r_k = 1 if ls_right.numel() == 0 else ls_right.shape[0]
        n_k = self.bases[k].cardinality

        ls = cartesian_prod((ls_left, self.bases[k].nodes[:, None], ls_right))
        H = self.target_func(ls).reshape(r_p, n_k, r_k)
        self.n_eval += ls.shape[0]
        return H

    def _is_finished(self, cross_iter: int, inds: Sequence) -> bool:

        max_iters = cross_iter == self.options.max_als
        max_error_tol = bool(self.errors[inds].max() < self.options.als_tol)
        l2_error_tol = self.l2_err < self.options.als_tol
        
        return max_iters or max_error_tol or l2_error_tol

    def _compute_cross_block_fixed(self, k: int) -> None:
        
        ls_left = self.tt_data.interp_ls[k-1]
        ls_right = self.tt_data.interp_ls[k+1]
        
        H = self._build_block_local(ls_left, ls_right, k) 
        self.errors[k] = TTFunc._get_error_local(H, self.coefs[k])
        self._build_basis_svd(H, k)
        return
    
    def _compute_cross_block_random(self, k: int) -> None:
        
        ls_left = self.tt_data.interp_ls[k-1].clone()
        ls_right = self.tt_data.interp_ls[k+1].clone()
        enrich = self.input_data.get_samples(self.options.kick_rank)

        H = self._build_block_local(ls_left, ls_right, k)
        self.errors[k] = TTFunc._get_error_local(H, self.coefs[k])

        if self.tt_data.direction == Direction.FORWARD:
            H_enrich = self._build_block_local(ls_left, enrich[:, k+1:], k)
            H_full = torch.concatenate((H, H_enrich), dim=2)
        else:
            H_enrich = self._build_block_local(enrich[:, :k], ls_right, k)
            H_full = torch.concatenate((H, H_enrich), dim=0)

        self._build_basis_svd(H_full, k)
        return
    
    def _compute_cross_block_amen(self, k: int) -> None:
        
        ls_left = self.tt_data.interp_ls[k-1]
        ls_right = self.tt_data.interp_ls[k+1]
        r_left = self.tt_data.res_x[k-1]
        r_right = self.tt_data.res_x[k+1]

        # Evaluate the interpolant function at x_k nodes
        H = self._build_block_local(ls_left, ls_right, k)
        self.errors[k] = TTFunc._get_error_local(H, self.coefs[k])

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
        
        self.errors[k] = TTFunc._get_error_local(H, self.coefs[k])
        self.coefs[k] = H
        self.cores[k] = self._coef2core(H, self.bases[k])

        return

    def _cross(self) -> None:
        """Builds the FTT using cross iterations."""

        previously_constructed = self.coefs != {}
        n_iter = 0

        if self.options.verbose > 0:
            self._print_info_header()

        if not previously_constructed:
            self._initialise_coefs()
        else:
            self.tt_data._reverse_direction()
        
        if self._use_amen:
            self._initialise_amen()

        while True:

            if self.tt_data.direction == Direction.FORWARD:
                inds = range(self.dim-1)
            else:
                inds = range(self.dim-1, 0, -1)
            
            for i, k in enumerate(inds):
                if self.options.verbose > 1:
                    msg = f"Building block {i+1} / {self.dim}..."
                    als_info(msg, end="\r")
                if self.options.tt_method == "fixed_rank":
                    self._compute_cross_block_fixed(k)
                elif self.options.tt_method == "random":
                    self._compute_cross_block_random(k)
                elif self.options.tt_method == "amen":
                    self._compute_cross_block_amen(k)

            n_iter += 1
            finished = self._is_finished(n_iter, inds)
            
            if finished:
                if self.options.verbose > 1:
                    msg = f"Building block {self.dim} / {self.dim}..."
                    als_info(msg, end="\r")
                self._compute_final_block()
            
            self._compute_rel_error()
            if self.options.verbose > 0:
                self._print_info(n_iter, inds)

            if finished:
                if self.options.verbose > 0:
                    als_info("ALS complete.")
                if self.options.verbose > 1:
                    ranks = "-".join([str(int(r)) for r in self.rank])
                    msg = f"Final TT ranks: {ranks}."
                    als_info(msg)
                return
            
            self.tt_data._reverse_direction()
    

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