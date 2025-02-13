from typing import Callable, Tuple
import warnings

import torch
from torch import Tensor

from .approx_bases import ApproxBases
from .directions import Direction
from .input_data import InputData
from .tt_data import TTData
from ..options import TTOptions
from ..polynomials import Basis1D, Piecewise, Spectral
from ..tools import deim, maxvol
from ..tools.printing import als_info


MAX_COND = 1.0e+5


class TTFunc():

    def __init__(
        self, 
        target_func: Callable[[Tensor], Tensor], 
        bases: ApproxBases, 
        options: TTOptions, 
        input_data: InputData,
        tt_data: TTData|None = None
    ):
        """A functional tensor-train approximation for a function 
        mapping from some subset of R^d to R.

        Parameters
        ----------
        target_func:
            Maps an n * d matrix containing samples from the local 
            domain to an n-dimensional vector containing the values of 
            the target function at each sample.
        bases:
            The bases associated with the approximation domain.
        options:
            Options used when constructing the FTT approximation to
            the target function.
        input_data:
            Data used for initialising and evaluating the quality of 
            the FTT approximation to the target function.
        tt_data:
            Data used to construct the FTT approximation to the target
            function.

        """
        
        self.target_func = target_func
        self.bases = bases 
        self.dim = bases.dim
        self.options = options
        self.input_data = input_data
        self.data = TTData() if tt_data is None else tt_data

        self.input_data.set_samples(self.bases, self.sample_size)
        if self.input_data.is_debug:
            self.input_data.set_debug(self.target_func, self.bases)

        # if isinstance(arg, ApproxFunc):
        #     self.options = arg.options
        
        self.num_eval = 0
        self.errors = torch.zeros(self.dim)
        self.l2_err = torch.inf
        self.linf_err = torch.inf
        return
        
    @property 
    def rank(self) -> Tensor:
        """The ranks of each tensor core."""
        return self.data.rank

    @property
    def use_amen(self) -> bool:
        """Whether to use AMEN."""
        return self.options.tt_method.lower() == "amen"
        
    @property
    def sample_size(self):
        """An upper bound on the total number of samples required to 
        construct a FTT approximation to the target function.
        """
        n = (self.options.init_rank 
             + self.options.kick_rank * (1+self.options.max_als))
        n *= self.dim
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
    def batch_mul(A: Tensor, B: Tensor) -> Tensor:
        """Batch-multiplies two sets of tensors together.
        """
        return torch.einsum("...ij, ...jk", A, B)

    @staticmethod
    def unfold_left(H: Tensor) -> Tensor:
        """Forms the left unfolding matrix associated with a tensor.
        """
        r_p, n_k, r_k = H.shape
        H = H.reshape(r_p * n_k, r_k)
        return H
    
    @staticmethod 
    def unfold_right(H: Tensor) -> Tensor:
        """Forms the (transpose of the) right unfolding matrix 
        associated with a tensor.
        """
        r_p, n_k, r_k = H.shape
        H = H.swapdims(0, 2).reshape(n_k * r_k, r_p)
        return H
    
    @staticmethod 
    def unfold(H: Tensor, direction: Direction) -> Tensor:
        """Unfolds a tensor.
        """
        if direction == Direction.FORWARD:
            H = TTFunc.unfold_left(H)
        else: 
            H = TTFunc.unfold_right(H)
        return H
    
    @staticmethod 
    def fold_left(H: Tensor, newshape: Tuple) -> Tensor:
        """Computes the inverse of the unfold_left operation.
        """
        H = H.reshape(*newshape)
        return H
    
    @staticmethod 
    def fold_right(H: Tensor, newshape: Tuple) -> Tensor:
        """Computes the inverse of the unfold_right operation.
        """
        H = H.reshape(*reversed(newshape)).swapdims(0, 2)
        return H

    @staticmethod
    def eval_core_213(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
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
            A matrix of dimension n_{k} * r_{k-1} * r_{k}, 
            corresponding to evaluations the kth core at each value 
            of ls.
        
        """
        r_p, n_k, r_k = A.shape
        n_ls = ls.numel()
        coeffs = A.permute(1, 0, 2).reshape(n_k, r_p * r_k)
        Gs = poly.eval_radon(coeffs, ls).reshape(n_ls, r_p, r_k)
        return Gs

    @staticmethod
    def eval_core_213_deriv(poly: Basis1D, A: Tensor, ls: Tensor) -> Tensor:
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

    def initialise_cores(self) -> None:
        """Initialises the cores and interpolation points in each 
        dimension.
        """

        for k in range(self.dim):

            core_shape = [
                1 if k == 0 else self.options.init_rank, 
                self.bases.polys[k].cardinality,
                1 if k == self.dim-1 else self.options.init_rank
            ]

            self.data.cores[k] = torch.zeros(core_shape)

            samples = self.input_data.get_samples(self.options.init_rank)
            self.data.interp_ls[k] = samples[:, k:]

        self.data.interp_ls[-1] = torch.tensor([])
        self.data.interp_ls[self.dim] = torch.tensor([])

        return

    def _initialise_res_x(self) -> None:
        """Initialises the residual coordinates for AMEN."""

        for k in range(self.dim-1, -1, -1):
            samples = self.input_data.get_samples(self.options.kick_rank)
            if self.data.direction == Direction.FORWARD:
                self.data.res_x[k] = samples[:, k:]
            else:
                self.data.res_x[k] = samples[:, :(k+1)]

        self.data.res_x[-1] = torch.tensor([])
        self.data.res_x[self.dim] = torch.tensor([])
        return
    
    def _initialise_res_w(self) -> None:
        """Initialises the residual blocks for AMEN."""

        if self.data.direction == Direction.FORWARD:
            
            core_0 = self.data.cores[0]
            shape_0 = (self.options.kick_rank, core_0.shape[-1])
            self.data.res_w[0] = torch.ones(shape_0)
            
            for k in range(1, self.dim):
                core_k = self.data.cores[k].shape[0]
                shape_k = (core_k, self.options.kick_rank)
                self.data.res_w[k] = torch.ones(shape_k)

        else:

            for k in range(self.dim-1):
                core_k = self.data.cores[k]
                shape_k = (self.options.kick_rank, core_k.shape[-1])
                self.data.res_w[k] = torch.ones(shape_k)

            core_d = self.data.cores[self.dim-1]
            shape_d = (core_d.shape[0], self.options.kick_rank)
            self.data.res_w[self.dim-1] = torch.ones(shape_d)

        self.data.res_w[-1] = torch.tensor([[1.0]])
        self.data.res_w[self.dim] = torch.tensor([[1.0]])
        return

    def initialise_amen(self) -> None:
        """Initialises the residual coordinates and residual blocks 
        for AMEN.
        """
        if self.data.res_x == {}:
            self._initialise_res_x()
        if self.data.res_w == {}:
            self._initialise_res_w()
        return

    def _print_info_header(self) -> None:

        info_headers = [
            "ALS", 
            "Max Local Error", 
            "Mean Local Error", 
            "Max Rank", 
            "Func Evals"
        ]
        
        if self.input_data.is_debug:
            info_headers += [
                "Max Debug Error", 
                "Mean Debug Error"
            ]

        als_info(" | ".join(info_headers))
        return

    def _print_info(self, als_iter: int, indices: Tensor) -> None:
        """Prints some diagnostic information about the current ALS 
        iteration.
        """

        diagnostics = [
            f"{als_iter:=3}", 
            f"{torch.max(self.errors[indices]):=15.5e}",
            f"{torch.mean(self.errors[indices]):=16.5e}",
            f"{torch.max(self.rank):=8}",
            f"{self.num_eval:=10}"
        ]

        if self.input_data.is_debug:
            diagnostics += [
                f"{self.linf_err:=15.5e}",
                f"{self.l2_err:=16.5e}"
            ]

        als_info(" | ".join(diagnostics))
        return

    def _select_points_piecewise(
        self,
        U: Tensor,
        poly: Piecewise
    ) -> Tuple[Tensor, Tensor, Tensor]:

        if self.options.int_method == "qdeim":
            raise NotImplementedError()
        elif self.options.int_method == "deim":
            inds, B = deim(U)
            U_interp = U[inds]
        elif self.options.int_method == "maxvol":
            inds, B = maxvol(U)
            U_interp = U[inds]
        
        if (cond := torch.linalg.cond(U_interp)) > MAX_COND:
            msg = f"Poor condition number in interpolation: {cond}."
            warnings.warn(msg)

        return inds, B, U_interp
    
    def _select_points_spectral(
        self,
        U: Tensor,
        poly: Spectral
    ) -> Tuple[Tensor, Tensor, Tensor]:

        n_k = poly.cardinality
        r_p = torch.tensor(U.shape[0] / n_k).round().int()
        
        nodes = poly.basis2node @ U.T.reshape(-1, n_k).T
        nodes = nodes.T.reshape(-1, n_k * r_p).T

        if self.options.int_method == "qdeim":
            raise NotImplementedError()
        elif self.options.int_method == "deim":
            msg = "DEIM is not supported for spectral polynomials."
            raise Exception(msg)
        elif self.options.int_method == "maxvol":
            inds, _ = maxvol(nodes)
            U_interp = nodes[inds]
            B = U @ torch.linalg.inv(U_interp)
        
        if (cond := torch.linalg.cond(U_interp)) > 1e+5:
            msg = f"Poor condition number in interpolation ({cond})."
            warnings.warn(msg)

        return inds, B, U_interp

    def select_points(self, U: Tensor, k: int) -> Tuple[Tensor, Tensor, Tensor]:
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

    def compute_relative_error(self) -> None:
        """Computes the relative error between the value of the FTT 
        approximation to the target function and the true value for the 
        set of debugging samples.
        """

        if not self.input_data.is_debug:
            return
        
        ps_approx = self.eval_local(self.input_data.ls_debug, self.data.direction)
        ps_approx = ps_approx.flatten()
        self.l2_err, self.linf_err = self.input_data.relative_error(ps_approx)
        return

    def build_block_local(
        self, 
        ls_left: Tensor,
        ls_right: Tensor,
        k: int
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
            An r_{k+1} * {k+1} matix containing a set of interpolation 
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

        poly = self.bases.polys[k]
        nodes = poly.nodes[:, None]

        r_p = 1 if ls_left.numel() == 0 else ls_left.shape[0]
        r_k = 1 if ls_right.numel() == 0 else ls_right.shape[0]
        n_k = poly.cardinality

        # Form the Cartesian product of the index sets and the nodes
        # corresponding to the basis of the current dimension
        if ls_left.numel() == 0:

            ls = torch.hstack((
                nodes.repeat_interleave(r_k, dim=0),
                ls_right.repeat(n_k, 1)
            ))

        elif ls_right.numel() == 0:

            ls = torch.hstack((
                ls_left.repeat_interleave(n_k, dim=0),
                nodes.repeat(r_p, 1)
            ))

        else:

            ls = torch.hstack((
                ls_left.repeat_interleave(n_k * r_k, dim=0),
                nodes.repeat_interleave(r_k, dim=0).repeat(r_p, 1),
                ls_right.repeat(r_p * n_k, 1)
            ))
        
        H = self.target_func(ls).reshape(r_p, n_k, r_k)

        # TODO: could be a separate method eventually
        if isinstance(poly, Spectral): 
            H = torch.einsum("jl, ilk", poly.node2basis, H)

        self.num_eval += ls.shape[0]
        return H

    def truncate_local(self, H: Tensor, k: int) -> Tuple[Tensor, Tensor, int]:
        """Truncates the SVD for a given TT block, F.

        Parameters
        ----------
        H:
            The unfolding matrix of evaluations of the target function 
            evaluated at a set of interpolation points.
        k:
            The index of the current dimension.
        
        Returns
        -------
        U:
            Matrix containing the left singular vectors of F after 
            truncation.
        sVh: 
            Matrix containing the transpose of the product of the 
            singular values and the right-hand singular vectors after
            truncation. 
        rank:
            The number of singular values of F that were retained.

        """
            
        U, s, Vh = torch.linalg.svd(H, full_matrices=False)
            
        energies = torch.flip(s**2, dims=(0,)).cumsum(dim=0)
        tol = 0.1 * energies[-1] * self.options.local_tol ** 2
        
        rank = torch.sum(energies > tol)
        rank = torch.clamp(rank, 1, self.options.max_rank)

        U = U[:, :rank]
        sVh = (s[:rank] * Vh[:rank].T).T
 
        return U, sVh, rank

    def apply_mass_R(self, poly: Basis1D, H: Tensor) -> Tensor:

        # Mass matrix for spectral polynomials is the identity
        if isinstance(poly, Spectral):
            return H
        
        nr_k = H.shape[0]
        H = poly.mass_R @ H.T.reshape(-1, poly.cardinality).T
        H = H.T.reshape(-1, nr_k).T
        return H

    def apply_mass_R_inv(self, poly: Basis1D, U: Tensor) -> Tensor:
        
        # Mass matrix for spectral polynomials is the identity
        if isinstance(poly, Spectral):
            return U

        nr_k = U.shape[0]
        U = U.T.reshape(-1, poly.cardinality).T
        U = torch.linalg.solve(poly.mass_R, U)
        U = U.T.reshape(-1, nr_k).T
        return U
    
    def build_basis_svd(self, H: Tensor, k: Tensor|int) -> None:
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

        k = int(k)
        k_prev = int(k - self.data.direction.value)
        k_next = int(k + self.data.direction.value)
        
        poly = self.bases.polys[k]
        interp_ls_prev = self.data.interp_ls[k_prev]
        A_next = self.data.cores[k_next]

        r_p, n_k, r_k = H.shape 
        r_p_next, _, r_k_next = A_next.shape

        H = TTFunc.unfold(H, self.data.direction)
        H = self.apply_mass_R(poly, H)
        U, sVh, rank = self.truncate_local(H, k)
        U = self.apply_mass_R_inv(poly, U)

        inds, B, U_interp = self.select_points(U, k)
        interp_ls = self.get_local_index(poly, interp_ls_prev, inds)

        couple = U_interp @ sVh

        # Form current coefficient tensor and update dimensions of next one
        if self.data.direction == Direction.FORWARD:
            A = TTFunc.fold_left(B, (r_p, n_k, rank))
            couple = couple[:, :r_p_next]
            A_next = torch.einsum("il, ljk", couple, A_next)
        else:
            A = TTFunc.fold_right(B, (rank, n_k, r_k))
            couple = couple[:, :r_k_next]
            A_next = torch.einsum("kl, ijl", couple, A_next)

        self.data.cores[k] = A
        self.data.cores[k_next] = A_next
        self.data.interp_ls[k] = interp_ls 
        return
    
    def build_basis_amen(
        self, 
        F: Tensor,
        F_res: Tensor,
        F_up: Tensor,
        k: Tensor|int
    ) -> None:
        """TODO: finish"""
        
        k = int(k)
        k_prev = int(k - self.data.direction.value)
        k_next = int(k + self.data.direction.value)
        
        poly = self.bases.polys[k]
        interp_x_prev = self.data.interp_ls[k_prev]
        res_x_prev = self.data.res_x[k_prev]

        res_w_prev = self.data.res_w[k-1]
        res_w_next = self.data.res_w[k+1]

        core_next = self.data.cores[k_next]

        n_left, n_nodes, n_right = F.shape
        n_r_left, _, n_r_right = F_res.shape
        r_0_next, n_nodes_next, r_1_next = core_next.shape

        F = TTFunc.unfold(F, self.data.direction)
        F_up = TTFunc.unfold(F_up, self.data.direction)

        if self.data.direction == Direction.FORWARD:
            r_prev = n_left 
            r_next = r_0_next
        else:
            r_prev = n_right 
            r_next = r_1_next

        F = self.apply_mass_R(poly, F)
        B, A, rank = self.truncate_local(F, k)
        B = self.apply_mass_R_inv(poly, B)

        if self.data.direction == Direction.FORWARD:

            temp_l = TTFunc.fold_left(B, (r_prev, n_nodes, rank))
            temp_l = torch.einsum("il, ljk", res_w_prev, temp_l)
            
            temp_r = A @ res_w_next
            F_up -= B @ temp_r

            F_res -= torch.einsum("ijl, lk", temp_l, temp_r)
            F_res = TTFunc.unfold_left(F_res)
        
        else: 
            
            # for the right projection
            temp_r = TTFunc.fold_right(B, (rank, n_nodes, r_prev))
            temp_r = torch.einsum("ijl, lk", temp_r, res_w_next)
            temp_r = TTFunc.unfold_right(temp_r)

            tmp_lt = A @ res_w_prev.T
            # Fu is aligned with F
            F_up -= B @ tmp_lt

            # align Fr as rnew (nrleft), nodes, rold (nrright), m
            F_res -= TTFunc.fold_right(temp_r @ tmp_lt, (n_r_left, n_nodes, n_r_right))
            F_res = TTFunc.unfold_right(F_res)
        
        # Enrich basis
        T = torch.cat((B, F_up), dim=1)

        if isinstance(poly, Piecewise):
            T = T.T.reshape(-1, poly.cardinality) @ poly.mass_R.T
            T = T.reshape(-1, B.shape[0]).T
            Q, R = torch.linalg.qr(T)
            B = torch.linalg.solve(poly.mass_R, Q.T.reshape(-1, poly.cardinality).T)
            B = B.T.reshape(-1, Q.shape[0]).T

        else:
            B, R = torch.linalg.qr(T)

        r_new = B.shape[-1]

        indices, core, interp_atx = self.select_points(B, k)
        couple = interp_atx @ R[:r_new, :rank] @ A

        interp_x = self.get_local_index(poly, interp_x_prev, indices)
        
        # TODO: it might be a good idea to add the error tolerance as an argument to this function.
        Qr = self.truncate_local(F_res, k)[0]

        indices_r = self.select_points(Qr, k)[0]
        res_x = self.get_local_index(poly, res_x_prev, indices_r)

        if self.data.direction == Direction.FORWARD:
            
            core = TTFunc.fold_left(core, (r_prev, n_nodes, r_new))

            temp = torch.einsum("il, ljk", res_w_prev, core)
            temp = TTFunc.unfold_left(temp)
            res_w = temp[indices_r, :]

            couple = couple[:, :r_next]
            core_next = torch.einsum("il, ljk", couple, core_next)

        else:
            
            core = TTFunc.fold_right(core, (r_new, n_nodes, r_prev))

            temp = torch.einsum("ijl, lk", core, res_w_next)
            temp = TTFunc.unfold_right(temp)
            res_w = temp[indices_r, :].T

            couple = couple[:, :r_next]
            core_next = torch.einsum("ijl, kl", core_next, couple)

        self.data.cores[k] = core 
        self.data.cores[k_next] = core_next
        self.data.interp_ls[k] = interp_x
        self.data.res_w[k] = res_w 
        self.data.res_x[k] = res_x
        return

    def get_error_local(self, H: Tensor, k: Tensor) -> Tensor:
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
        core = self.data.cores[int(k)]
        return (core-H).abs().max() / H.abs().max()

    def get_local_index(
        self,
        poly: Basis1D, 
        interp_ls_prev: Tensor,
        inds: Tensor
    ) -> Tensor:
        """Updates the set of interpolation points for the current 
        dimension.
        
        Parameters
        ----------
        poly:
            The polynomial basis for the current dimension of the 
            approximation.
        interp_ls_prev: 
            The previous set of interpolation points.
        inds:
            The set of indices of the maximum-volume submatrix of the 
            current (unfolded) tensor core.
        
        Returns
        -------
        interp_ls:
            The set of updated interpolation points for the current 
            dimension.
            
        """

        if interp_ls_prev.numel() == 0:
            interp_ls = poly.nodes[inds][:, None]
            return interp_ls

        n_k = poly.cardinality

        ls_prev = interp_ls_prev[inds // n_k]
        ls_nodes = poly.nodes[inds % n_k][:, None]

        if self.data.direction == Direction.FORWARD:
            interp_ls = torch.hstack((ls_prev, ls_nodes))
        else:
            interp_ls = torch.hstack((ls_nodes, ls_prev))

        return interp_ls

    def is_finished(self, als_iter: int, indices: Tensor) -> bool:
        """Returns True if the maximum number of ALS iterations has 
        been reached or the desired error tolerance is met, and False 
        otherwise.
        """
        
        max_iters = als_iter == self.options.max_als
        max_error_tol = torch.max(self.errors[indices]) < self.options.als_tol
        # TODO: check where l2_err actually gets updated
        l2_error_tol = self.l2_err < self.options.als_tol

        return max_iters or max_error_tol or l2_error_tol

    def _compute_cross_iter_fixed_rank(self, indices: Tensor) -> None:

        for k in indices:
                    
            ls_left = self.data.interp_ls[int(k-1)]
            ls_right = self.data.interp_ls[int(k+1)]
            
            H = self.build_block_local(ls_left, ls_right, k) 
            self.errors[k] = self.get_error_local(H, k)
            self.build_basis_svd(H, k)

        return
    
    def _compute_cross_iter_random(self, indices: Tensor) -> None:
        
        for k in indices:
            
            ls_left = self.data.interp_ls[int(k-1)].clone()
            ls_right = self.data.interp_ls[int(k+1)].clone()
            enrich = self.input_data.get_samples(self.options.kick_rank)

            F_k = self.build_block_local(ls_left, ls_right, k)
            self.errors[k] = self.get_error_local(F_k, k)

            if self.data.direction == Direction.FORWARD:
                F_enrich = self.build_block_local(ls_left, enrich[:, k+1:], k)
                F_full = torch.concatenate((F_k, F_enrich), dim=2)
            else:
                F_enrich = self.build_block_local(enrich[:, :k], ls_right, k)
                F_full = torch.concatenate((F_k, F_enrich), dim=0)

            self.build_basis_svd(F_full, k)

        return
    
    def _compute_cross_iter_amen(self, indices: Tensor) -> None:
        
        for k in indices:
            
            ls_left = self.data.interp_ls[int(k-1)]
            ls_right = self.data.interp_ls[int(k+1)]
            r_left = self.data.res_x[int(k-1)]
            r_right = self.data.res_x[int(k+1)]

            # Evaluate the interpolant function at x_k nodes
            F = self.build_block_local(ls_left, ls_right, k)
            self.errors[k] = self.get_error_local(F, k)

            # Evaluate residual function at x_k nodes
            F_res = self.build_block_local(r_left, r_right, k)

            if self.data.direction == Direction.FORWARD and k > 0:
                F_up = self.build_block_local(ls_left, r_right, k)
            elif self.data.direction == Direction.BACKWARD and k < self.dim-1: 
                F_up = self.build_block_local(r_left, ls_right, k)
            else:
                F_up = F_res.clone()

            self.build_basis_amen(F, F_res, F_up, k)

        return 

    def _compute_final_block(self) -> None:
        """Computes the final block of the FTT approximation to the 
        target function.
        """

        if self.data.direction == Direction.FORWARD:
            k = self.dim-1 
        else:
            k = 0

        ls_left = self.data.interp_ls[int(k-1)]
        ls_right = self.data.interp_ls[int(k+1)]
        H = self.build_block_local(ls_left, ls_right, k)
        self.errors[k] = self.get_error_local(H, k)
        self.data.cores[k] = H
        return

    def cross(self) -> None:
        """Builds the FTT using cross iterations.
        """

        self._print_info_header()
        als_iter = 0

        if self.data.cores == {}:
            self.data.direction = Direction.FORWARD 
            self.initialise_cores()
        else:
            # Prepare for the next iteration
            self.data.reverse_direction()
        
        if self.use_amen:
            self.initialise_amen()

        while True:

            if self.data.direction == Direction.FORWARD:
                indices = torch.arange(self.dim-1)
            else:
                indices = torch.arange(self.dim-1, 0, -1)
            
            if self.options.tt_method == "fixed_rank":
                self._compute_cross_iter_fixed_rank(indices)
            elif self.options.tt_method == "random":
                self._compute_cross_iter_random(indices)
            elif self.options.tt_method == "amen":
                self._compute_cross_iter_amen(indices)

            als_iter += 1
            if (finished := self.is_finished(als_iter, indices)):
                self._compute_final_block()

            self.compute_relative_error()
            self._print_info(als_iter, indices)

            if finished:
                als_info(f"ALS complete.")
                als_info(f"Final TT ranks: {[int(r) for r in self.rank]}.")
                return
            else:
                self.data.reverse_direction()

    def round(self):
        """Rounds the TT cores. Applies double rounding to get back to 
        the starting direction.
        """

        for _ in range(2):
            
            self.data.reverse_direction()

            if self.data.direction == Direction.FORWARD:
                indices = torch.arange(self.dim-1)
            else:
                indices = torch.arange(self.dim-1, 0, -1)

            for k in indices:
                self.build_basis_svd(self.data.cores[int(k)], k)

        return
    
    def _eval_local_forward(self, ls: Tensor) -> Tensor:
        """Evaluates the FTT approximation to the target function for 
        the first k variables.
        """

        n_ls, d_ls = ls.shape
        polys = self.bases.polys
        cores = self.data.cores
        Gs_prod = torch.ones((n_ls, 1))

        for k in range(d_ls):
            Gs = self.eval_core_213(polys[k], cores[k], ls[:, k])
            Gs_prod = torch.einsum("il, ilk -> ik", Gs_prod, Gs)
        
        return Gs_prod
    
    def _eval_local_backward(self, ls: Tensor) -> Tensor:
        """Evaluates the FTT approximation to the target function for 
        the last k variables.
        """

        n_ls, d_ls = ls.shape
        polys = self.bases.polys 
        cores = self.data.cores
        Gs_prod = torch.ones((n_ls, 1))
        
        for i, k in enumerate(range(self.dim-1, self.dim-d_ls-1, -1), start=1):
            Gs = self.eval_core_231(polys[k], cores[k], ls[:, -i])
            Gs_prod = torch.einsum("il, ilk -> ik", Gs_prod, Gs)
        
        return Gs_prod

    def eval_local(self, ls: Tensor, direction: Direction) -> Tensor:
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
        self._check_sample_dim(ls, self.dim, strict=True)
        ls = self.bases.approx2local(xs)[0]
        gs = self.eval_local(ls, self.data.direction).flatten()
        return gs

    def grad_reference(self, zs: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates the gradient of the approximation to the target 
        function for a set of reference variables.
        """
        raise NotImplementedError()

    def grad(self, xs: Tensor) -> Tensor:
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
        gxs:
            TODO: finish this once grad_reference is done.

        """
        zs, dzdxs = self.bases.approx2local(xs)
        gzs, fxs = self.grad_reference(self, zs)
        gxs = gzs * dzdxs
        return gxs, fxs
    
    def int_reference(self):
        """Integrates the approximation to the target function over the 
        reference domain (TODO: check this).
        """
        raise NotImplementedError()