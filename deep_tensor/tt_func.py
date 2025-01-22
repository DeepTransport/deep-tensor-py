from typing import Callable, Tuple
import warnings

import torch

from .approx_bases import ApproxBases
from .directions import Direction
from .input_data import InputData
from .options import TTOptions
from .polynomials import Basis1D, Piecewise, Spectral
from .tools import deim, maxvol, reshape_matlab
from .tt_data import TTData
from .utils import als_info

MAX_COND = 1.0e+5


class TTFunc():

    def __init__(
        self, 
        target_func: Callable[[torch.Tensor], torch.Tensor], 
        bases: ApproxBases, 
        options: TTOptions, 
        input_data: InputData,
        tt_data: TTData|None=None
    ):
        """A functional tensor-train approximation for a function 
        mapping from some subset of R^d to some subset of R.

        Parameters
        ----------
        target_func:
            Maps an n * d matrix containing samples from the local 
            domain ([-1, 1]^d) to an n-dimensional vector containing 
            the values of the target function at each sample.
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
    def rank(self) -> torch.Tensor:
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

    def initialise_cores(self) -> None:
        """Initialises the cores and interpolation points in each 
        dimension.
        """

        for k in range(self.dim):

            core_shape = [
                1 if k == 0 else self.options.init_rank, 
                self.bases.polys[k].cardinality,
                self.options.init_rank if k != self.dim-1 else 1
            ]

            self.data.cores[k] = torch.rand(core_shape)

            samples = self.input_data.get_samples(self.options.init_rank)
            self.data.interp_x[k] = samples[:, k:]

        self.data.interp_x[-1] = torch.tensor([])
        self.data.interp_x[self.dim] = torch.tensor([])

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

        self.data.res_w[-1] = torch.tensor([1.0])
        self.data.res_w[self.dim] = torch.tensor([1.0])
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

    def _print_info(
        self,
        als_iter: int,
        indices: torch.Tensor
    ) -> None:
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

    def _compute_cross_iter_fixed_rank(
        self, 
        target_func: Callable,
        indices: torch.Tensor
    ) -> None:

        for k in indices:
                    
            x_left = self.data.interp_x[int(k-1)]
            x_right = self.data.interp_x[int(k+1)]
            
            F_k = self.build_block_local(target_func, x_left, x_right, k) 
            self.errors[k] = self.get_error_local(F_k, k)
            self.build_basis_svd(F_k, k)

        return
    
    def _compute_cross_iter_random(
        self,
        target_func: Callable[[torch.Tensor], torch.Tensor],
        indices: torch.Tensor
    ) -> None:
        
        for k in indices:
            
            x_left = self.data.interp_x[int(k-1)].clone()
            x_right = self.data.interp_x[int(k+1)].clone()
            enrich = self.input_data.get_samples(self.options.kick_rank)

            F_k = self.build_block_local(target_func, x_left, x_right, k)
            self.errors[k] = self.get_error_local(F_k, k)

            if self.data.direction == Direction.FORWARD:
                F_enrich = self.build_block_local(target_func, x_left, enrich[:, k+1:], k)
                F_full = torch.concatenate((F_k, F_enrich), dim=2)
            else:
                F_enrich = self.build_block_local(target_func, enrich[:, :k], x_right, k)
                F_full = torch.concatenate((F_k, F_enrich), dim=0)

            self.build_basis_svd(F_full, k)

        return None
    
    def _compute_cross_iter_amen(
        self, 
        func: Callable, 
        indices: torch.Tensor
    ):
        
        for k in indices:
            
            x_left = self.data.interp_x[int(k-1)]
            x_right = self.data.interp_x[int(k+1)]
            r_left = self.data.res_x[int(k-1)]
            r_right = self.data.res_x[int(k+1)]

            # Evaluate the interpolant function at x_k nodes
            F = self.build_block_local(func, x_left, x_right, k)
            self.errors[k] = self.get_error_local(F, k)

            # Evaluate residual function at x_k nodes
            F_res = self.build_block_local(func, r_left, r_right, k)

            if self.data.direction == Direction.FORWARD and k > 0:
                F_up = self.build_block_local(func, x_left, r_right, k)
            elif self.data.direction == Direction.BACKWARD and k < self.dim-1: 
                F_up = self.build_block_local(func, r_left, x_right, k)
            else:
                F_up = F_res.clone()

            self.build_basis_amen(F, F_res, F_up, k)

        return 

    def _compute_final_block(self, func: Callable) -> None:

        if self.data.direction == Direction.FORWARD:
            k = self.dim-1 
        else:
            k = 0

        x_left = self.data.interp_x[int(k-1)]
        x_right = self.data.interp_x[int(k+1)]
        self.data.cores[k] = self.build_block_local(func, x_left, x_right, k)
        return

    def _select_points_piecewise(
        self,
        H: torch.Tensor,
        poly: Piecewise
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.options.int_method == "qdeim":
            raise NotImplementedError()
            # indices, core = qdeim(H)
            # interp_atx = H[indices]

        elif self.options.int_method == "deim":
            indices, A = deim(H)
            interp_atx = H[indices]

        elif self.options.int_method == "maxvol":
            indices, A = maxvol(H)
            interp_atx = H[indices]
        
        if (cond := torch.linalg.cond(interp_atx)) > MAX_COND:
            msg = f"Poor condition number in interpolation: {cond}."
            warnings.warn(msg)

        return indices, A, interp_atx
    
    def _select_points_spectral(
        self,
        H: torch.Tensor,
        poly: Spectral
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # TODO: this is probably available somewhere else.
        rank_prev = torch.tensor(H.shape[0] / poly.cardinality)
        rank_prev = rank_prev.round().int()
        
        nodes = poly.basis2node @ reshape_matlab(H, (poly.cardinality, -1))
        nodes = reshape_matlab(nodes, (poly.cardinality * rank_prev, -1))

        if self.options.int_method == "qdeim":
            raise NotImplementedError()
        
        elif self.options.int_method == "deim":
            msg = "DEIM is not supported for spectral polynomials."
            raise Exception(msg)

        elif self.options.int_method == "maxvol":
            indices, _ = maxvol(nodes)
            interp_atx = nodes[indices]
            core = H @ torch.linalg.inv(interp_atx)
        
        if (cond := torch.linalg.cond(interp_atx)) > 1e+5:
            msg = f"Poor condition number in interpolation ({cond})."
            warnings.warn(msg)

        return indices, core, interp_atx

    def compute_relative_error(self) -> None:
        """TODO: write docstring."""

        if not self.input_data.is_debug:
            return
        
        ps_approx = self.eval_local(self.input_data.ls_debug)
        self.l2_err, self.linf_err = self.input_data.relative_error(ps_approx)
        return

    def build_block_local(
        self, 
        target_func: Callable[[torch.Tensor], torch.Tensor], 
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """Evaluates the function being approximated at a (reduced) set 
        of interpolation points, and returns the corresponding
        local coefficient matrix.

        Parameters
        ----------
        target_func: 
            The function being approximated.
        x_left:
            An r_{k-1} * {k-1} matrix containing a set of interpolation
            points for dimensions 1, ..., {k-1}.
        x_right:
            An r_{k+1} * {k+1} matix containing a set of interpolation 
            points for dimensions {k+1}, ..., d.
        k:
            The dimension in which interpolation is being carried out.

        Returns
        -------
        F_k: 
            An r_{k-1} * n_{k} * r_{k} tensor containing the values of 
            the function evaluated at each interpolation point.

        References
        ----------
        Cui and Dolgov (2022). Deep composition of tensor-trains using 
        squared inverse Rosenblatt transports.
        
        """

        num_left = 1 if x_left.numel() == 0 else x_left.shape[0]
        num_right = 1 if x_right.numel() == 0 else x_right.shape[0]

        poly = self.bases.polys[k]
        nodes = poly.nodes[:, None]

        # Form the Cartesian product of the index sets and the nodes
        # corresponding to the basis of the current dimension
        if x_left.numel() == 0:

            params = torch.hstack((
                nodes.repeat_interleave(num_right, dim=0),
                x_right.repeat(poly.cardinality, 1)
            ))

        elif x_right.numel() == 0:

            params = torch.hstack((
                x_left.repeat_interleave(poly.cardinality, dim=0),
                nodes.repeat(num_left, 1)
            ))

        else:

            params = torch.hstack((
                x_left.repeat_interleave(poly.cardinality * num_right, dim=0),
                nodes.repeat_interleave(num_right, dim=0).repeat(num_left, 1),
                x_right.repeat(num_left * poly.cardinality, 1)
            ))
        
        F_k = target_func(params).reshape(num_left, poly.cardinality, num_right)

        # TODO: could be a separate method eventually
        if isinstance(poly, Spectral): 
            F_k = F_k.permute(2, 0, 1).reshape(num_left * num_right, -1).T
            F_k = poly.node2basis @ F_k
            F_k = F_k.T.reshape(num_right, num_left, -1).permute(1, 2, 0)

        self.num_eval += params.shape[0]
        return F_k

    def truncate_local(
        self,
        F: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Truncates the SVD for a given TT block, F.

        Parameters
        ----------
        F:
            An r_{k_next} * (n_{k} * r_{k_prev}) matrix containing 
            the coefficients associated with the kth tensor core.
        k:
            The index of the current dimension.
        
        Returns
        -------
        B:
            Matrix containing the left singular vectors of F after 
            truncation.
        A: 
            Matrix containing the transpose of the product of the 
            singular values and the right-hand singular vectors after
            truncation. 
        rank:
            The number of singular values of F that were retained.

        """

        poly = self.bases.polys[k]
        if isinstance(poly, Piecewise):
            nr_prev = F.shape[0]
            F = poly.mass_R @ F.T.reshape(-1, poly.cardinality).T
            F = F.T.reshape(-1, nr_prev).T
            
        U, s, Vh = torch.linalg.svd(F, full_matrices=False)
            
        energies = torch.flip(s**2, dims=(0,)).cumsum(dim=0)
        tol = 0.1 * energies[-1] * self.options.local_tol ** 2
        
        rank = torch.sum(energies > tol)
        rank = torch.clamp(rank, 1, self.options.max_rank)

        B = U[:, :rank]
        A = (s[:rank] * Vh[:rank].T).T

        if isinstance(poly, Piecewise):
            B = B.T.reshape(-1, poly.cardinality).T
            B = torch.linalg.solve(poly.mass_R, B)
            B = B.T.reshape(-1, nr_prev).T
 
        return B, A, rank

    def build_basis_svd(
        self, 
        F_k: torch.Tensor, 
        k: torch.Tensor|int
    ) -> None:
        """TODO: write docstring...
        
        Parameters
        ----------
        F_k:
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
        interp_x_prev = self.data.interp_x[k_prev]
        core_next = self.data.cores[k_next]

        n_left, n_nodes, n_right = F_k.shape 
        r_0_next, n_nodes_next, r_1_next = core_next.shape

        if self.data.direction == Direction.FORWARD:
            F_k = F_k.permute(2, 0, 1).reshape(n_right, n_left * n_nodes).T
            r_prev = n_left
        else: 
            F_k = F_k.permute(0, 2, 1).reshape(n_left, n_nodes * n_right).T
            r_prev = n_right

        B, A, rank = self.truncate_local(F_k, k)

        indices, core, interp_atx = self.select_points(B, k)

        couple = reshape_matlab(interp_atx @ A, (rank, -1))
        interp_x = self.get_local_index(poly, interp_x_prev, indices)

        if self.data.direction == Direction.FORWARD:
            
            core = core.T.reshape(rank, r_prev, n_nodes).permute(1, 2, 0)

            couple = couple[:, :r_0_next].T.reshape(r_0_next, -1).T
            
            core_next = couple @ core_next.permute(2, 1, 0).reshape(-1, r_0_next).T
            core_next = reshape_matlab(core_next, (rank, n_nodes_next, r_1_next))

        else:
            
            core = reshape_matlab(core, (n_nodes, r_prev, rank))
            core = core.permute(2, 0, 1)

            couple = couple[:, :r_1_next].T
            couple = reshape_matlab(couple, (r_1_next, -1))

            core_next = reshape_matlab(core_next, (-1, r_1_next))
            core_next = core_next @ couple 
            core_next = reshape_matlab(core_next, (r_0_next, n_nodes_next, rank))

        self.data.cores[k] = core
        self.data.interp_x[k] = interp_x 
        self.data.cores[k_next] = core_next

        return
    
    def build_basis_amen(
        self, 
        F: torch.Tensor,
        F_res: torch.Tensor,
        F_up: torch.Tensor,
        k: torch.Tensor|int
    ) -> None:
        """TODO: finish"""
        
        k = int(k)
        k_prev = int(k - self.data.direction.value)
        k_next = int(k + self.data.direction.value)
        
        poly = self.bases.polys[k]
        interp_x_prev = self.data.interp_x[k_prev]
        res_x_prev = self.data.res_x[k_prev]

        res_w_prev = self.data.res_w[k-1]
        res_w_next = self.data.res_w[k+1]

        core_next = self.data.cores[k_next]

        n_left, n_nodes, n_right = F.shape
        n_r_left, _, n_r_right = F_res.shape
        r_0_next, _, r_1_next = core_next.shape

        if self.data.direction == Direction.FORWARD:
            F = F.permute(2, 0, 1).reshape(n_right, n_left * n_nodes).T
            F_up = F_up.permute(2, 0, 1).reshape(n_r_right, n_left * n_nodes).T
            r_prev = n_left 
            r_next = r_0_next
        else:
            F = F.permute(0, 2, 1).reshape(n_left, n_nodes * n_right).T
            F_up = F_up.permute(0, 2, 1).reshape(n_left, n_nodes * n_right).T
            r_prev = n_right 
            r_next = r_1_next

        B, A, rank = self.truncate_local(F, k)

        if self.data.direction == Direction.FORWARD:
            
            temp_r = A @ res_w_next
            F_up -= B @ temp_r

            temp_l = reshape_matlab(B, (n_nodes, r_prev, rank)).permute(1, 0, 2)
            temp_l = reshape_matlab(temp_l, (r_prev, -1))
            
            temp_l = res_w_prev @ temp_l
            temp_l = reshape_matlab(temp_l, (n_r_left * n_nodes, rank))

            F_res = reshape_matlab(F_res, (n_r_left, n_nodes, n_r_right)) - reshape_matlab(temp_l @ temp_r, (n_r_left, n_nodes, n_r_right))
            F_res = reshape_matlab(F_res.permute(1, 0, 2), (n_nodes * n_r_left, -1))
            r_r_prev = n_r_left
        
        else: 
            raise NotImplementedError()
        
        # Enrich basis
        T = torch.cat((B, F_up), dim=1)

        if isinstance(poly, Piecewise):
            T = T.T.reshape(-1, poly.cardinality) @ poly.mass_R.T
            T = T.reshape(-1, B.shape[0]).T
            Q, R = torch.linalg.qr(T)
            B = torch.linalg.solve(poly.mass_R, Q.T.reshape(-1, poly.cardinality).T)
            B = reshape_matlab(B, (Q.shape[0], -1))

        else:
            B, R = torch.linalg.qr(T)

        r_new = B.shape[-1]

        indices, core, interp_atx = self.select_points(B, k)
        couple = reshape_matlab(interp_atx @ (R[:r_new, :rank] @ A), (r_new, r_next))

        interp_x = self.get_local_index(poly, interp_x_prev, indices)
        
        # TODO: it might be a good idea to add the error tolerance as an argument to this function.
        Qr = self.truncate_local(F_res, k)[0]

        indices_r = self.select_points(Qr, k)[0]
        res_x = self.get_local_index(poly, res_x_prev, indices_r)

        if self.data.direction == Direction.FORWARD:
            
            core = reshape_matlab(core, (n_nodes, r_prev, r_new))
            core = core.permute(1, 0, 2)

            couple = couple[:, :r_next]
            couple = reshape_matlab(couple, (-1, r_next))
            core_next = couple @ reshape_matlab(core_next, (r_next, -1))
            core_next = reshape_matlab(core_next, (r_new, n_nodes, r_1_next))

            temp = res_w_prev @ reshape_matlab(core, (r_prev, n_nodes * r_new))
            temp = reshape_matlab(temp, (r_r_prev, n_nodes, r_new))
            temp = temp.permute(1, 0, 2)
            temp = reshape_matlab(temp, (-1, r_new))
            res_w = temp[indices_r, :]

        else:
            raise NotImplementedError()

        self.data.cores[k] = core 
        self.data.cores[k_next] = core_next
        self.data.interp_x[k] = interp_x
        self.data.res_w[k] = res_w 
        self.data.res_x[k] = res_x

        return

    def get_error_local(
        self,
        F: torch.Tensor,
        k: torch.Tensor
    ) -> torch.Tensor:
        """Returns the ..."""
        core = self.data.cores[int(k)]
        return (core-F).abs().max() / F.abs().max()

    def get_local_index(
        self,
        poly: Basis1D, 
        interp_x_prev: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Builds the local nested index set.
        
        TODO: write docstring for this.
        """

        if interp_x_prev.numel() == 0:
            interp_x = poly.nodes[indices][:, None]
            return interp_x

        rank_prev = interp_x_prev.shape[0]

        # Form the full Cartesian product of something...
        i_pair = torch.vstack((
            torch.arange(rank_prev).repeat_interleave(poly.cardinality),
            torch.arange(poly.cardinality).repeat(rank_prev)
        ))
        i_select = i_pair[:, indices]

        if self.data.direction == Direction.FORWARD:
            interp_x = torch.hstack((
                interp_x_prev[i_select[0]],
                poly.nodes[i_select[1]][:, None]
            ))
        else:
            interp_x = torch.hstack((
                poly.nodes[i_select[1]][:, None],
                interp_x_prev[i_select[0]]
            ))

        return interp_x

    @staticmethod
    def eval_oned_core_213(
        poly_k: Basis1D, 
        A_k: torch.Tensor, 
        ls: torch.Tensor 
    ) -> torch.Tensor:
        """Evaluates the kth tensor core at a given set of values.

        Parameters
        ----------
        poly_k:
            The basis functions associated with the current dimension.
        A_k:
            The coefficient tensor associated with the current core.
        ls: 
            A vector of points at which to evaluate the current core.

        Returns
        -------
        G_k:
            A matrix of dimension r_{k-1}n_{k} * r_{k}, corresponding 
            to evaluations of the kth core at each value of ls stacked 
            on top of one another.
        
        """
        
        r_p, n_k, r_k = A_k.shape
        n_l = ls.numel()

        coeffs = A_k.permute(2, 0, 1).reshape(r_k * r_p, n_k).T

        G_k = (poly_k.eval_radon(coeffs, ls).T
               .reshape(r_k, r_p, n_l)
               .swapdims(1, 2)
               .reshape(r_k, r_p * n_l).T)
        return G_k

    def eval_oned_core_213_deriv(
        self, 
        poly: Basis1D, 
        core: torch.Tensor, 
        xs: torch.Tensor 
    ) -> torch.Tensor:

        raise NotImplementedError()

    @staticmethod
    def eval_oned_core_231(
        poly: Basis1D, 
        A_k: torch.Tensor, 
        ls: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the kth tensor core at a given set of values.

        Parameters
        ----------
        poly_k:
            The basis functions associated with the current dimension.
        A_k:
            The coefficient tensor associated with the current core.
        ls: 
            A vector of points at which to evaluate the current core.

        Returns
        -------
        G_k:
            A matrix of dimension r_{k}n_{k} * r_{k-1}, corresponding 
            to evaluations of the kth core at each value of ls stacked 
            on top of one another.
        
        """
        
        r_p, n_k, r_k = A_k.shape
        n_l = ls.numel()

        coeffs = A_k.swapdims(1, 2).reshape(r_p * r_k, n_k).T

        G_k = (poly.eval_radon(coeffs, ls).T
               .reshape(r_p, r_k, n_l)
               .swapdims(1, 2)
               .reshape(r_p, r_k * n_l).T)
        return G_k
    
    def eval_oned_core_231_deriv(
        self, 
        poly: Basis1D,
        core: torch.Tensor,
        xs: torch.Tensor
    ) -> torch.Tensor:
        """TODO: write docstring."""

        raise NotImplementedError()

    def eval(self, xs: torch.Tensor) -> torch.Tensor:
        """Evaluates the approximated function at a set of points in 
        the approximation domain.
        
        Parameters
        ----------
        xs:
            A matrix containing n sets of d-dimensional input 
            variables in the approximation domain. Each row contains a
            single input variable.
            
        Returns
        -------
        fxs:
            An n-dimensional vector containing the values of the 
            function at each x value.
        """
        ls = self.bases.approx2local(xs)[0]
        fxs = self.eval_local(ls)
        return fxs

    def eval_local(
        self, 
        ls: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the approximation to the target function for a set 
        of samples in the local domain ([-1, 1]^d).

        Parameters
        ----------
        ls:
            An n * d matrix containing samples in the local domain.
        
        Returns
        -------
        ps:
            An n-dimensional vector containing the result of evaluating
            the target function at each element in ls. 
            
        """
        ps = self.eval_block(ls, self.data.direction)
        return ps
    
    def _eval_block_forward(self, ls: torch.Tensor) -> torch.Tensor:
        """Evaluates the FTT approximation to the target function for 
        the first k variables.
        """

        n_l, dim_l = ls.shape
        fls = torch.ones((n_l, 1))

        for k in range(min(dim_l, self.dim)):

            r_p = self.data.cores[k].shape[0]

            G_k = self.eval_oned_core_213(
                self.bases.polys[k],
                self.data.cores[k],
                ls[:, k]
            )
            
            ii = torch.arange(n_l).repeat(r_p)
            jj = (torch.arange(r_p * n_l)
                    .reshape(n_l, r_p).T
                    .flatten())
            indices = torch.vstack((ii[None, :], jj[None, :]))
            size = (n_l, r_p * n_l)
            B = torch.sparse_coo_tensor(indices, fls.T.flatten(), size)

            fls = B @ G_k

        return fls.squeeze()
    
    def _eval_block_backward(self, ls: torch.Tensor) -> torch.Tensor:
        """Evaluates the FTT approximation to the target function for 
        the last k variables.
        """

        n_l, dim_l = ls.shape
        fls = torch.ones((n_l, 1))

        x_inds = torch.arange(dim_l-1, -1, -1)
        t_inds = torch.arange(self.dim-1, -1, -1)
        
        for i in range(min(dim_l, self.dim)):
            
            j = int(t_inds[i])
            
            r_j = self.data.cores[j].shape[-1]

            G_k = self.eval_oned_core_231(
                self.bases.polys[j],
                self.data.cores[j],
                ls[:, x_inds[i]]
            )

            ii = torch.arange(n_l * r_j)
            jj = torch.arange(n_l).repeat_interleave(r_j)
            
            indices = torch.vstack((ii[None, :], jj[None, :]))
            size = (r_j * n_l, n_l)
            B = torch.sparse_coo_tensor(indices, fls.T.flatten(), size)

            fls = G_k.T @ B

        return fls.squeeze()

    def eval_block(
        self, 
        ls: torch.Tensor,
        direction: Direction
    ) -> torch.Tensor:
        """Evaluates the functional tensor train approximation to the 
        target function for either the first or last k variables, for a 
        set of points in the local domain ([-1, 1]).
        
        Parameters
        ----------
        ls:
            A set of points in the local domain.
        
        Returns
        -------
        fls:
            The value of the FTT approximation to the target function
            at each point in ls.
            
        """
        if direction == Direction.FORWARD:
            fls = self._eval_block_forward(ls)
        else: 
            fls = self._eval_block_backward(ls)
        return fls
    
    def grad_reference(
        self, 
        zs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the gradient of the approximation to the target 
        function for a set of reference variables.
        """
        raise NotImplementedError()

    def grad(self, xs: torch.Tensor) -> torch.Tensor:
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

    def round(self):
        """Rounds the TT cores.
        
        TODO: finish docstring.
        """

        # Apply double rounding to get back to the starting direction
        for _ in range(2):
            
            self.data.reverse_direction()

            if self.data.direction == Direction.FORWARD:
                indices = torch.arange(self.dim-1)
            else:
                indices = torch.arange(self.dim-1, 0, -1)

            for k in indices:
                self.build_basis_svd(self.data.cores[int(k)], k)

        return

    def select_points(
        self,
        H: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds the cross indices.

        Parameters
        ----------
        H:
            TODO: write this
        k: 
            The index of the current dimension.
        
        Returns
        -------
        indices:
            The set of indices of the (approximate) maximum volume 
            submatrix of H.
        core:
            The corresponding (unfolded) tensor core.
        interp_atx:
            The nodes of the basis of the current dimension 
            corresponding to the set of indices of the maximum volume
            submatrix.
        
        """

        poly = self.bases.polys[k]

        if isinstance(poly, Piecewise):
            return self._select_points_piecewise(H, poly)
        elif isinstance(poly, Spectral):
            return self._select_points_spectral(H, poly)
    
        raise Exception("Unknown polynomial encountered.")

    def is_finished(
        self, 
        als_iter: int,
        indices: torch.Tensor
    ) -> bool:
        """Returns True if the maximum number of ALS iterations has 
        been reached or the desired error tolerance is met, and False 
        otherwise.
        """
        
        max_iters = als_iter == self.options.max_als
        max_error_tol = torch.max(self.errors[indices]) < self.options.als_tol
        l2_error_tol = self.l2_err < self.options.als_tol # TODO: check where l2_err actually gets updated.

        return max_iters or max_error_tol or l2_error_tol

    def cross(
        self, 
        #target_func: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """Cross iterations for building the tensor train.

        Parameters
        ----------
        func: 
            Returns the target function evaluated at a given point in 
            the reference domain.
        
        Returns 
        -------
        None
        
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
                self._compute_cross_iter_fixed_rank(self.target_func, indices)
            
            elif self.options.tt_method == "random":
                self._compute_cross_iter_random(self.target_func, indices)

            elif self.options.tt_method == "amen":
                self._compute_cross_iter_amen(self.target_func, indices)
            
            else: 
                raise Exception("Unknown TT method.")

            als_iter += 1
            finished = self.is_finished(als_iter, indices)

            if finished: 
                self._compute_final_block(self.target_func)

            self.compute_relative_error()

            if finished:
                self._print_info(als_iter, indices)
                als_info(f"ALS complete.")
                als_info(f"Final TT ranks: {[int(r) for r in self.rank]}.")
                return

            else:
                self._print_info(als_iter, indices)
                self.data.reverse_direction()